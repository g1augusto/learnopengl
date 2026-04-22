'''
OpenGL in Python with ModernGL
Based on learngopengl.com
https://learnopengl.com/Getting-started/Hello-Triangle
Vertex Input
Vertex Buffer Object (VBO)
Vertex Shader
Fragment Shader
Vertex Array Object (VAO)
+ Element Buffer Object (EBO) aka Index Buffer Object (IBO older term)

Excercise:  Instancing 
            NOTE:   To achieve instancing while retaining the previous implementation of model loading with pyASSIMP I have modified a little from learnopengl implementation
                    - Model class now has a DYNAMIC VBO where we reserve the necessary data based on the number of instances to render
                        -   it is only at creation time, it could be dynamic or set on the fly but for now is ok (setting instances parameter >1)
                        -   the dynamic VBO is passed to the mesh processing, if a mesh identifies that the model class called with an actual VBO for multiple instances then it adds
                            the necessary parameters to its own VBO
                            -   the logic is that each mesh has its own VBO as before but in case of multiple instances they all refer to the MODEL VBO (only a single VBO for all meshes)
                                where instances data is going to be passed (dynamic VBO host no actual data at creation time, just reserved)
                        -   To populate the model dynamic instances VBO is necessary to call the "writeInstancesData" method passing a pre-populated list of glm.mat4 model matrices
                        -   there is a dedicated shader for multiple instances (progInstances) and model and mesh class can be used for single rendering and multiple rendering with instancing
                            -   is necessary at creation time and at rendering time to set the correct number of instances and the correct shader

                        
                        Controls
                        ----------------------------------------


                        F2: start/stop spinning light
                        F3: increase source light ambient 
                        F4: decrease source light ambient
                        F5: increase source light diffuse
                        F6: decrease source light diffuse
                        F3 + CTRL: Increase constant attenuation
                        F4 + CTRL: Decrease constant attenuation
                        F5 + CTRL: Increase linear attenuation
                        F6 + CTRL: Decrease linear attenuation
                        F7 + CTRL: Increase quadric attenuation
                        F8 + CTRL: Decrease quadric attenuation                    
                        Mouse wheel + SPACE: Increase/decrease spotlight cone
                        F11: Release mouselook
'''
from enum import Enum # to define movement enum class
import math
import ctypes # for pyglm bytes conversion
import struct
import sys
import pygame
import moderngl
import glm
from dataclasses import dataclass
import os
# pyassimp requires assimp.dll to be on the os PATH, this ensures that the local folder is on the OS PATH
new_path = os.getcwd() + os.pathsep + os.environ['PATH']
os.environ['PATH'] = new_path
import pyassimp
from itertools import zip_longest # used for mesh data in processMesh
import random


################# Text Renderer Class to show FPS


class TextAlign (Enum):
    CENTER = 1
    LEFT = 2
    RIGHT = 3
    CENTER_TOP = 4
    CENTER_BOTTOM = 5
    LEFT_TOP = 6
    LEFT_BOTTOM = 7
    RIGHT_TOP = 8
    RIGHT_BOTTOM = 9


class TextRenderer():
    _unit_quad_vbo = None
    _unit_quad_vao = None
    _shader_program = None # Stores the shader program shared by all instances

    @classmethod
    def initialize_shared_resources(cls, ctx: moderngl.Context, shader_program: moderngl.Program):
        """
        Initializes OpenGL resources shared by all TextRenderer instances.
        Call this ONCE after context and shader program are ready.
        """
        cls._shader_program = shader_program
        if cls._unit_quad_vbo is None:
            # Define vertices for a unit quad (0,0) at bottom-left to (1,1) at top-right
            # This matches how u_text_offset + (unit_pos * dimensions) would typically work.
            # UVs (0,0) for bottom-left of texture, (1,1) for top-right.
            # Assumes texture data is flipped if coming from Pygame default surface.
            unit_vertices = [
                # Pos (X,Y)  UV (S,T)
                0.0, 0.0,    0.0, 0.0,  # Bottom-left
                1.0, 0.0,    1.0, 0.0,  # Bottom-right
                0.0, 1.0,    0.0, 1.0,  # Top-left

                0.0, 1.0,    0.0, 1.0,  # Top-left
                1.0, 0.0,    1.0, 0.0,  # Bottom-right
                1.0, 1.0,    1.0, 1.0   # Top-right
            ]
            vertex_data_bytes = struct.pack(f"{len(unit_vertices)}f", *unit_vertices)
            cls._unit_quad_vbo = ctx.buffer(vertex_data_bytes)
            cls._unit_quad_vao = ctx.vertex_array(
                cls._shader_program,
                [(cls._unit_quad_vbo, "2f 2f", "in_unit_vertex_pos", "in_uv")]
            )
        print("TextRenderer shared resources initialized.")

    @classmethod
    def release_shared_resources(cls):
        """Releases shared OpenGL resources. Call ONCE at application shutdown."""
        if cls._unit_quad_vao:
            cls._unit_quad_vao.release()
            cls._unit_quad_vao = None
        if cls._unit_quad_vbo:
            cls._unit_quad_vbo.release()
            cls._unit_quad_vbo = None
        # Note: The shader program is typically released by whoever created it,
        # but if TextRenderer exclusively "owns" it, it could be released here.
        # For this example, we assume shader program lifecycle is managed outside.
        cls._shader_program = None
        print("TextRenderer shared resources released.")

    def __init__(self, ctx: moderngl.Context, initial_text: str, font_path: str, font_size: int,
                 text_rgb_color: tuple, position: glm.vec2,bold:bool = False,italic:bool = False,visible:bool = True,
                 alignment:TextAlign=TextAlign.LEFT_TOP): # Using glm.vec2 for position
        if TextRenderer._shader_program is None or TextRenderer._unit_quad_vao is None:
            raise RuntimeError("TextRenderer shared resources not initialized. Call TextRenderer.initialize_shared_resources() first.")

        self.ctx = ctx
        try:
            if os.path.exists(font_path):
                self.pygame_font = pygame.font.Font(font_path, font_size)
                self.pygame_font.bold = bold
                self.pygame_font.italic = italic
            else:
                self.pygame_font = pygame.font.SysFont(font_path, font_size,bold=bold,italic=italic)
        except pygame.error as e:
            print(f"Warning: Font '{font_path}' not found or error: {e}. Using system default.")
            self.pygame_font = pygame.font.SysFont(None, font_size) # Fallback

        self.text_rgb_color = text_rgb_color # e.g., (255, 255, 255)
        self.position = None # This will be the bottom-left position of the text quad
        self.actual_position = None
        
        self.text_texture = None
        self.text_width = 0.0
        self.text_height = 0.0
        self.visible:bool = visible
        self.alignment:TextAlign = alignment
        self.text = initial_text

        self.update_text(initial_text)
        self.set_position(position)

    def update_text(self, new_text: str):
        textLines = new_text.split('\n')
        currentPosition = glm.vec2(0,0)
        lineSpacing = self.pygame_font.get_linesize()
        # Pygame renders with (0,0) at top-left.
        # We want text_color to be just RGB, alpha comes from convert_alpha()

        textLinesList = [self.pygame_font.render(textLine, True, self.text_rgb_color).convert_alpha() for textLine in textLines]
        textMaxHeight = len(textLinesList) * lineSpacing
        textMaxWidth = 0
        for textLine in textLinesList:
            currentWidth = textLine.get_width()
            if currentWidth > textMaxWidth:
                textMaxWidth = currentWidth
        text_surface_pygame = pygame.Surface((textMaxWidth, textMaxHeight), pygame.SRCALPHA)
        
        for textLine in textLinesList:
            text_surface_pygame.blit(textLine,currentPosition)
            currentPosition.y += lineSpacing

        self.text_width = float(text_surface_pygame.get_width())
        self.text_height = float(text_surface_pygame.get_height())
        
        # image_data is flipped vertically to match OpenGL's texture coord convention (0,0 at bottom-left)
        image_data = pygame.image.tostring(text_surface_pygame, "RGBA", False) 

        if self.text_texture:
            self.text_texture.release()
        
        if self.text_width > 0 and self.text_height > 0:
            self.text_texture = self.ctx.texture(
                (int(self.text_width), int(self.text_height)), 4, image_data
            )
            self.text_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.text_texture.use(location=0)

        else:
            self.text_texture = None # Handle empty string case

    def set_position(self, new_position: glm.vec2):
        self.position = new_position
        actual_position:glm.vec2 = glm.vec2(0,0)

        match self.alignment:
            case TextAlign.CENTER:
                actual_position = self.position - (glm.vec2(self.text_width,self.text_height) / 2)
            case TextAlign.LEFT:
                actual_position.x = self.position.x
                actual_position.y = self.position.y - (self.text_height / 2)
            case TextAlign.RIGHT:
                actual_position.x = self.position.x - self.text_width
                actual_position.y = self.position.y - (self.text_height / 2)
            case TextAlign.CENTER_TOP:
                actual_position.x = self.position.x - (self.text_width / 2)
                actual_position.y = self.position.y
            case TextAlign.CENTER_BOTTOM:
                actual_position.x = self.position.x - (self.text_width / 2)
                actual_position.y = self.position.y - self.text_height
            case TextAlign.LEFT_TOP:
                actual_position = self.position   
            case TextAlign.LEFT_BOTTOM:
                actual_position.x = self.position.x
                actual_position.y = self.position.y - self.text_height
            case TextAlign.RIGHT_TOP:
                actual_position.x = self.position.x - self.text_width
                actual_position.y = self.position.y
            case TextAlign.RIGHT_BOTTOM:
                actual_position.x = self.position.x - self.text_width
                actual_position.y = self.position.y - self.text_height
        
        self.actual_position = actual_position

    def render(self):
        if not self.text_texture or TextRenderer._unit_quad_vao is None or TextRenderer._shader_program is None:
            return



        # Set uniforms
        TextRenderer._shader_program['u_text_dimensions'].value = (self.text_width, self.text_height) # pylint: disable=unsubscriptable-object
        TextRenderer._shader_program['u_text_offset'].value = tuple(self.actual_position) # pylint: disable=unsubscriptable-object

        self.text_texture.use(location=0) # Bind to texture unit 0
        TextRenderer._shader_program['u_texture'].value = 0 # pylint: disable=unsubscriptable-object # Tell sampler to use texture unit 0

        # Render the shared VAO
        self.ctx.enable(moderngl.BLEND)
        TextRenderer._unit_quad_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.disable(moderngl.BLEND)



### Mesh data structure
@dataclass
class Vertex:
    Position: glm.vec3
    #Normal: glm.vec3
    TexCoords: glm.vec2

    def size(self):
        #size = ( glm.sizeof(self.Position) + glm.sizeof(self.Normal) + glm.sizeof(self.TexCoords) ) // 4
        size = ( glm.sizeof(self.Position) + glm.sizeof(self.TexCoords) ) // 4
        return size  

@dataclass
class Texture:
    id: moderngl.Texture
    type: str
    path: str

class Mesh():
    def __init__(self,context:moderngl.Context,program:moderngl.Program,vertices:list[Vertex],indices:list[int],textures:list[Texture],vbo_instances:moderngl.Context.buffer=None):
        self.context = context # in ModernGL is necessary to create a VAO
        self.program = program # in ModernGL is necessary to create a VAO
        self.vertices = vertices
        self.indices = indices
        self.textures = textures
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.vbo_instances = vbo_instances

        self.setupMesh()

    def Draw(self, shader: moderngl.Program,instances=1):
        '''
        Uniforms in the shader follows a specific naming convention:\n
        material.texture_diffuseX or material.texture_specularY\n
        where X and Y is a sequential number increased at each type of texture processed\n
        Each texture in the list is first bound with the .use method to a texture unit\n
        (Texture.id is a moderngl texture object)\n
        after identified the uniform name, sets the shader sampler uniform to the texture unit 
        Lastly it verifies if the VAO is already in use and finally render the mesh
        '''
        diffuseNr = 0
        specularNr = 0
        for texture_unit,texture in enumerate(self.textures):
            uniform_name = "" # to avoid any exception later when used
            texture.id.use(location=texture_unit) # texture.id is a moderngl context.texture object
            if texture.type == "texture_diffuse":
                diffuseNr += 1
                uniform_name  = f"{texture.type}{diffuseNr}"
            elif texture.type == "texture_specular":
                specularNr += 1
                uniform_name  = f"{texture.type}{specularNr}"
            # check if the uniform name assembled is an available uniform in the shader
            # if yes assign the uniform to use the texture unit previously selected
            if uniform_name in shader:
                shader[uniform_name] = texture_unit
        

        # Render the mesh VAO with the passed shader
        if instances >1:
            self.vao.render(instances=instances)
        else:
            self.vao.render()


    def setupMesh(self):
        ## Create VBO data
        # create a mutable bytearray to contain each vertex data in bytes, also calculate the format of ech vertex (8 floats)
        vertices_binaryformat_array = bytearray()
        vertex_format = f"{self.vertices[0].size()}f"
        # iterate through each vertex and convert the glm vectors to list elements and concatenate them in a single list
        # the single list of all elements of a single vertex is then converted to byte and concatenated to the initial bytearray
        for vertex in self.vertices:
            #components = list(vertex.Position) + list(vertex.Normal) + list(vertex.TexCoords)
            components = list(vertex.Position) + list(vertex.TexCoords)
            packed_vertex = struct.pack(vertex_format, *components)
            vertices_binaryformat_array.extend(packed_vertex) # Append bytes to bytearray
        vertices_binaryformat = bytes(vertices_binaryformat_array) # convert mutable byterray to immutable bytes
        self.vbo = self.context.buffer(vertices_binaryformat)

        ## Create EBO data
        indices_binaryformat = struct.pack(f"{len(self.indices)}I",*self.indices)
        self.ebo = self.context.buffer(indices_binaryformat)

        ## Create VAO
        #vbo_string = f"{glm.sizeof(self.vertices[0].Position)//4}f {glm.sizeof(self.vertices[0].Normal)//4}f {glm.sizeof(self.vertices[0].TexCoords)//4}f"
        # NOTE Create VBO parameters without aNormal (not needed in this example)
        #vbo_parameters = [(self.vbo,vbo_string,"aPos","aNormal","aTexCoords")]
        vbo_string_no_normal = f"{glm.sizeof(self.vertices[0].Position)//4}f {glm.sizeof(self.vertices[0].TexCoords)//4}f"
        vbo_parameters_no_normal = [(self.vbo,vbo_string_no_normal,"aPos","aTexCoords")]
        # NOTE: this is where we take into consideration if we have multiple instances being requested by the model calling class
        # 16f refers to a glm.mat4 matrix which has 4x4 floats = 16f
        # if the model calling class has no instancing, the mesh can render regularly (assumed the right shader is set)
        if self.vbo_instances:
            vbo_parameters_no_normal.append(
                (self.vbo_instances,"16f /i","instanceMatrix")
            )
        self.vao = self.context.vertex_array(self.program,vbo_parameters_no_normal,self.ebo)

class Model():
    def __init__(self,context:moderngl.Context,program:moderngl.Program,path:str,flipTexture=True,instances:int=1):
        self.context = context
        self.program = program
        self.meshes:list[Mesh] = []
        self.directory = ""
        self.path = path
        self.textures_loaded:list[Texture] = []
        self.flipTexture = flipTexture # flip or not Mesh Textures 
        self.instances = instances
        self.vbo_instances = None
        self.setInstances(self.instances)
        self.loadModel(path=self.path)
    
    def setInstances(self,instances):
        '''
        This method prepares the dynamic VBO to host model matrices data in case the model is created with instancing set
        '''
        if instances > 1:
            # Total Bytes = (Number of Instances) × (Floats per Matrix) × (Bytes per Float)
            # Floats per Matrix: A mat4 is a 4x4 matrix, so it contains 4 * 4 = 16
            # Bytes per Float: A standard single-precision float (the type used by default in GLSL and pyglm) is 4 bytes
            TotalBytesMat4 = instances * 16 * 4
            self.vbo_instances = self.context.buffer(reserve=TotalBytesMat4,dynamic=True)
        
    def writeInstanceData(self,matrices:list[glm.mat4]):
        '''
        This method writes actual data to the dynamc VBO created for instancing\n
        At creation time the dynamic VBO is empty, this method takes a list of mat4\n
        converts them and write them to the dynamic VBO
        '''
        # --- Step 1: Flatten the list of matrices into a generator of floats ---
        # This is the core logic. It iterates through each matrix, then each
        # column in the matrix, then each component (float) in the column.
        # glm matrices are column-major by default, which is what OpenGL expects.
        matrices_flat_format = (
            component
            for matrix in matrices    # For each mat4 in the list
            for column in matrix      # For each vec4 column in the mat4
            for component in column   # For each float in the vec4
        )       
        # --- Step 2: Create the format string for struct.pack ---
        # Each mat4 has 16 floats.
        format_string = f"{len(matrices)*16}f"
        
        matrices_binaryformat = struct.pack(format_string,*matrices_flat_format)
        self.vbo_instances.write(matrices_binaryformat)

    def loadModel(self,path:str=""):
        self.meshes.clear()
        if path == "":
            path = self.path
        with pyassimp.load(filename=path,processing=pyassimp.postprocess.aiProcess_Triangulate | pyassimp.postprocess.aiProcess_FlipUVs | pyassimp.postprocess.aiProcess_GenSmoothNormals) as scene:
            if scene:
                self.directory = os.path.dirname(path)
            self.processNode(node=scene.rootnode)
            
    
    def Draw(self,shader:moderngl.Program,instances=1):
        for mesh in self.meshes:
            mesh.Draw(shader=shader,instances=instances) # NOTE here we are passing the number of instances

    def processNode(self,node:pyassimp.structs.Node):
        for mesh in node.meshes:
            self.meshes.append(self.processMesh(mesh))
            print(f"added mesh: {len(self.meshes)}")
        for node in node.children:
            self.processNode(node=node)
        
    
    def processMesh(self,mesh:pyassimp.structs.Mesh) -> Mesh:
        vertices:list[Vertex] = []
        indices:list[int] = []
        textures:list[Texture] = []
        # retrieve Vertices data
        # zip_longest allows to iterate until the longest data and by default sets shorter data to None
        # used because there may be less textures
        # texture coordinates are in a sub list of texturecoords at element 0 but it is possible
        # that this element is not present ( no texture coords) so to keep the zip function consistent 
        # we catch the specific IndexError exception mesh.texturecoords is empty
        try:
            iterate_texturecoords = mesh.texturecoords[0]
        except IndexError:
            iterate_texturecoords = []
        for position,normal,textureCoords in zip_longest(mesh.vertices,mesh.normals,iterate_texturecoords):
            new_position = glm.vec3(position)
            #new_normal = glm.vec3(normal)
            if textureCoords:
                new_TexCoords = glm.vec2(textureCoords)
            else:
                new_TexCoords = glm.vec2(0,0)
            #vertices.append(Vertex(Position=new_position,Normal=new_normal,TexCoords=new_TexCoords))
            vertices.append(Vertex(Position=new_position,TexCoords=new_TexCoords))
        
        # retrieve incides data and flatten the faces NxN list into a 1xN list
        indices = [item for sublist in mesh.faces for item in sublist]

        # retrieve textures (or materials)
        # there is only one material for each mesh and in case the model has meshes with
        # multiple material, assimp split that mesh into multiple logical meshes, to keep
        # a 1:1 mesh:material ratio
        if mesh.materialindex >= 0:
            for key,value in mesh.material.properties.items():
                if key == "file":
                    # clarify that "value" is the texture image path (for readability)
                    path = f"{self.directory}/{value}"
                    # look if the texture was already loaded
                    found_texture = next((texture for texture in self.textures_loaded if texture.path == path),None)
                    if found_texture:
                        textures.append(found_texture)
                    else:
                        Mapimage = pygame.image.load(path)
                        Mapimage_data = pygame.image.tobytes(Mapimage,"RGBA",self.flipTexture)
                        MapTexture = self.context.texture(Mapimage.get_size(),4,Mapimage_data)
                        texture_type = None
                        if "diffuse" in path:
                            # load diffuse texture
                            texture_type = "texture_diffuse"
                        elif "specular" in path:
                            texture_type = "texture_specular"
                        elif "normal" in path:
                            texture_type = "texture_normal"
                        else:
                            texture_type = "unknown"
                        new_texture = Texture(MapTexture,texture_type,path)
                        # check that we have identified a supported texture map format
                        if texture_type:
                            textures.append(new_texture)
                            self.textures_loaded.append(new_texture)
        new_mesh = Mesh(self.context,self.program,vertices,indices,textures,self.vbo_instances) # here we pass the dynamic VBO attribute reference, if not null it will create and instancing mesh
        return new_mesh


    


class Camera():
    '''
    Camera Class:
    Update first the vectors with updateCameraVectors()\n
    Assign the VIEW matrix with the return value of GetViewMatrix()
    '''
    YAW = -90.0
    PITCH = 0.0
    SPEED = 0.05
    TURNSPEED = 1.0
    SENSITIVITY = 0.1
    ZOOM = 45.0


    def __init__(self,cameraPos = glm.vec3(0.0, 0.0, 0.0), cameraUp = glm.vec3(0.0, 1.0, 0.0), yaw = YAW, pitch = PITCH):
        self.cameraTarget = glm.vec3(0.0, 0.0, -1.0)
        self.MovementSpeed = Camera.SPEED
        self.TurnSpeed = Camera.TURNSPEED
        self.MouseSensitivity = Camera.SENSITIVITY
        self.zoom = Camera.ZOOM
        self.cameraPos = cameraPos
        self.cameraUp = cameraUp
        self.yaw = yaw
        self.pitch = pitch

    def GetViewMatrix(self):
        return glm.lookAt(self.cameraPos,self.cameraPos + self.cameraTarget, self.cameraUp)
    
    def updateCameraVectors(self):
        # calculate the new Target vector
        direction = glm.vec3()
        direction.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        direction.y = math.sin(glm.radians(self.pitch))
        direction.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.cameraTarget = glm.normalize(direction)


    class Movement(Enum):
        '''
        Movement Enum subclass
        Allows for the definition of specific readable format constants to use in movement
        it's defined as a subclass of Camera since there is no need to use it outside
        it cam be accessed like the following:\n
        Camera.Movement.FORWARD
        '''
        FORWARD = 1
        BACKWARD = 2
        TURN_LEFT = 3
        TURN_RIGHT = 4
        STRIFE_LEFT = 5
        STRIFE_RIGHT = 6
        LOOK_UP = 7
        LOOK_DOWN = 8
    
    def ProcessKeyboard(self,direction,deltaTime):
        if not isinstance(direction,Camera.Movement): # check type
            raise ValueError
        else:
            velocity = self.MovementSpeed * deltaTime
            turnVelocity = self.TurnSpeed * deltaTime
        if direction == Camera.Movement.FORWARD:
            self.cameraPos += self.cameraTarget * velocity
        elif direction == Camera.Movement.BACKWARD:
            self.cameraPos -= self.cameraTarget * velocity
        elif direction == Camera.Movement.TURN_RIGHT:
            self.yaw += turnVelocity * deltaTime
        elif direction == Camera.Movement.TURN_LEFT:
            self.yaw -= turnVelocity * deltaTime
        elif direction == Camera.Movement.STRIFE_RIGHT:
            self.cameraPos += glm.normalize(glm.cross(self.cameraTarget,self.cameraUp)) * velocity * deltaTime
        elif direction == Camera.Movement.STRIFE_LEFT:
            self.cameraPos -= glm.normalize(glm.cross(self.cameraTarget,self.cameraUp)) * velocity * deltaTime
        elif direction == Camera.Movement.LOOK_UP:
            self.pitch += turnVelocity * deltaTime
        elif direction == Camera.Movement.LOOK_DOWN:
            self.pitch -= turnVelocity * deltaTime

    def ProcessMouseMovement(self,xoffset,yoffset,deltaTime,constrainPitch=True):
        xoffset *= self.MouseSensitivity * deltaTime
        yoffset *= self.MouseSensitivity * deltaTime
        self.yaw += xoffset
        self.pitch -= yoffset
        if constrainPitch:
            if(self.pitch > 89.0):
                self.pitch =  89.0
            if(self.pitch < -89.0):
                self.pitch = -89.0      

    def ProcessMouseScroll(self,yoffset,deltaTime):
        self.zoom -= yoffset * deltaTime
        if self.zoom < 1.0:
            self.zoom = 1.0
        if self.zoom > 45.0:
            self.zoom = 45.0


class CameraFPS(Camera):
    def __init__(self, cameraPos=glm.vec3(0, 0, 0), cameraUp=glm.vec3(0, 1, 0), yaw=Camera.YAW, pitch=Camera.PITCH):
        super().__init__(cameraPos, cameraUp, yaw, pitch)
    

    def ProcessKeyboard(self,direction,deltaTime):
        super().ProcessKeyboard(direction=direction,deltaTime=deltaTime)
        # make sure the user stays at the ground level
        self.cameraPos.y = 0.0 # <-- this one-liner keeps the user at the ground level (xz plane)

windowed_size = (800,600)
vsync = False

pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE)

# Create and initializize display
screen_flags = pygame.OPENGL | pygame.RESIZABLE | pygame.DOUBLEBUF
screen_display = pygame.display.set_mode(windowed_size,flags=screen_flags,vsync=vsync)



### OpenGL section

# ModernGL create a context : a state machine or a container for OpenGL
context = moderngl.create_context()

### Enable DEPTH TESTING
# When depth testing is enabled, OpenGL (and thus ModernGL) uses a depth buffer to determine which fragments (pixels) should be drawn on the screen.
# Each fragment has a depth value, which represents its distance from the viewer.   
# https://moderngl.readthedocs.io/en/latest/reference/context.html#Context.enable
context.enable(moderngl.DEPTH_TEST)
depth_test = True

# Define Vertex Shader and Fragment Shader in ModernGL (GLSL language)
# ModernGL abstracts vertex and fragment shader as specific parameter of the context program method
# NOTE: This shader is used to render the models on scene and another shader will be used to
#       render the light source cube (the only difference is in the fragment shader)
prog = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
//layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    TexCoords = aTexCoords;    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D texture_diffuse1;

void main()
{
    FragColor = texture(texture_diffuse1, TexCoords);
}
''')


progInstances = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in mat4 instanceMatrix;

out vec2 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    gl_Position = projection * view * instanceMatrix * vec4(aPos, 1.0); 
    TexCoords = aTexCoords;
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D texture_diffuse1;

void main()
{
    FragColor = texture(texture_diffuse1, TexCoords);
}
''')

# This is the definition of the shaders for the light source cube.
# in this example we represent the light source as a cube for demonstrative reasons but we
# don't want that cube to be affected by the transformations that will be done on the models
# (including light source reflections) so we use a different shader sets where the fragment
# shader is unaffected by light sources
progLight = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // note that we read the multiplication from right to left
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0); // set all 4 vector values to 1.0
}
''')

### TEXT RENDERING SHADER
program_text = context.program(
vertex_shader='''
#version 330 core
layout (location = 0) in vec2 in_unit_vertex_pos; // Unit quad (0,0 to 1,1)
layout (location = 1) in vec2 in_uv;

uniform mat4 projection;
uniform vec2 u_text_dimensions;   // Actual width and height of the text
uniform vec2 u_text_offset;       // Screen position (e.g., top-left or bottom-left)

out vec2 v_uv;

void main() {
    vec2 scaled_pos = in_unit_vertex_pos * u_text_dimensions;
    vec2 world_pos = scaled_pos + u_text_offset; // Assuming u_text_offset is bottom-left
    gl_Position = projection * vec4(world_pos, 0.0, 1.0);
    v_uv = in_uv;
}
''',
fragment_shader = '''
#version 330 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_texture;

void main() {
    out_color = texture(u_texture, v_uv);
}
'''
)


# vertices
#  NOTE: A normal vector is a (unit) vector that is perpendicular to the surface of a vertex.
#        Since a vertex by itself has no surface (it's just a single point in space) we retrieve a
#        normal vector by using its surrounding vertices to figure out the surface of the vertex.
#        We can use a little trick to calculate the normal vectors for all the cube's vertices by 
#        using the cross product, but since a 3D cube is not a complicated shape we can simply manually 
#        add them to the vertex data
#        To measure the angle between the light ray and the fragment we use something called a normal vector,
#        that is a vector perpendicular to the fragment's surface
vertices = [
    # position           # normal vector       # Texture
    #                      for each vertex       coordinates
    #                      (precalculated)
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0,  1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0,  0.0,

        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0,  1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0,  0.0,

        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  1.0,  1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0,  1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  0.0,  0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0,  0.0,

         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0,  1.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0,  1.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0,  0.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0,  0.0,

        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0,  1.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0,  1.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0,  0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0,  1.0,

        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0,  1.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0,  0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0,  1.0
]

vertices_lightbox = [
    # positions          
    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5,  0.5, -0.5,
     0.5,  0.5, -0.5,
    -0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5,

    -0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    -0.5, -0.5,  0.5,

    -0.5,  0.5,  0.5,
    -0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5,
    -0.5, -0.5,  0.5,
    -0.5,  0.5,  0.5,

     0.5,  0.5,  0.5,
     0.5,  0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,

    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,
    -0.5, -0.5, -0.5,

    -0.5,  0.5, -0.5,
     0.5,  0.5, -0.5,
     0.5,  0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    -0.5,  0.5, -0.5,
]

# let's define a translation vector for each cube that specifies its position in world space. We'll define 10 cube positions 
# we will iterate over these positions for the MODEL view translation (MODEL view is related to the original object position)
cubePositions = [
    glm.vec3( 0.0,  0.0,  0.0), 
    glm.vec3( 2.0,  5.0, -15.0), 
    glm.vec3(-1.5, -2.2, -2.5),  
    glm.vec3(-3.8, -2.0, -12.3),  
    glm.vec3( 2.4, -0.4, -3.5),  
    glm.vec3(-1.7,  3.0, -7.5),  
    glm.vec3( 1.3, -2.0, -2.5),  
    glm.vec3( 1.5,  2.0, -2.5), 
    glm.vec3( 1.5,  0.2, -1.5), 
    glm.vec3(-1.3,  1.0, -1.5)     
]

# Point lights positions
pointLightPositions = [
	glm.vec3( 0.7,  0.2,  2.0),
	glm.vec3( 2.3, -3.3, -4.0),
	glm.vec3(-4.0,  2.0, -12.0),
	glm.vec3( 0.0,  0.0, -3.0)   
]
PointLights = len(pointLightPositions)

# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '32f': This is the format string. It specifies that we want to pack 32 floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
vertices_binaryformat = struct.pack(f"{len(vertices)}f",*vertices)
# now for the vertices of the lightbox
vertices_lightbox_binaryformat = struct.pack(f"{len(vertices_lightbox)}f",*vertices_lightbox)


# Define VBO (Vertex Buffer Object) containing vertex data
vbo = context.buffer(vertices_binaryformat)

# Define VBO for the lightbox VAO
vboLightbox = context.buffer(vertices_lightbox_binaryformat)




# now for the lightbox
vboLightbox_parameters = [
    (vboLightbox,"3f","aPos")
]

# Define an additional VAO For the light reusing the VBO data from the box VBO
# so we define a light source with the same model for simplicity but we attach it to another
# VAO so we don't propagate changes on the boxes models also on the light source
lightvao = context.vertex_array(progLight,vboLightbox_parameters)


def matrix_bytes(matrix:glm.mat4):
    '''
    Function to convert a glm matrix into a GLSL readable stream of bytes to pass as a uniform
    '''
    ptr = glm.value_ptr(matrix)
    matrix_size = matrix.length() * matrix.length()
    float_array = (ctypes.c_float * matrix_size).from_address(ctypes.addressof(ptr.contents))
    matrix_bytes_output = bytes(float_array)
    return matrix_bytes_output


### Camera Object
cam = Camera(glm.vec3(0.0, 0.0, 55.0))


# Reference variables for Delta time
FRAMERATE_REFERENCE = 60
FRAMERATE = 0 # Set framerate to 0 to let it run as fast as possible, otherwise set it to the desired frame rate (e.g., 60)


amount:int = 20000
# Start with a moving light
moveLight = False
pygame.display.set_caption("Click on the window to enable mouselook")

# this model will be rendered normally
PlanetModelFile = "./model3d/planet.obj"
PlanetModel = Model(context,prog,PlanetModelFile)

# this model will be instanced
RockModelFile = "./model3d/rock.obj"
RockModel = Model(context,progInstances,RockModelFile,instances=amount)

# here we create the model matrices list
modelMatrices:list[glm.mat4] = []
radius:float = 50.0
offset:float = 2.5
for _ in range(amount):
    model = glm.mat4(1.0)
    # 1. translation: displace along circle with 'radius' in range [-offset, offset]
    angle = _/amount * 360.0
    displacement = (random.randint(0,32767) % int(2 * offset * 100)) / 100.0 - offset
    x = math.sin(angle) * radius + displacement
    displacement = (random.randint(0,32767) % int(2 * offset * 100)) / 100.0 - offset
    y = displacement * 0.4
    displacement = (random.randint(0,32767) % int(2 * offset * 100)) / 100.0 - offset
    z = math.cos(angle) * radius + displacement
    model = glm.translate(model,glm.vec3(x,y,z))

    # 2. scale: scale between 0.05 and 0.25f
    scale = (random.randint(0,32767) % 20) / 100.0 + 0.05
    model = glm.scale(model,glm.vec3(scale))

    # 3. rotation: add random rotation around a (semi)randomly picked rotation axis vector
    rotAngle = random.randint(0,32767) % 360
    model = glm.rotate(model,rotAngle,glm.vec3(0.4,0.6,0.8))

    # 4. now add to list of matrices
    modelMatrices.append(model)

# we are passing the list of matrices to the dynamic VBO for the instancing model
RockModel.writeInstanceData(modelMatrices)

clock = pygame.time.Clock()

# this just creates an useful object to visualize the FPS on screen, to understand performance impact of instancing
TextRenderer.initialize_shared_resources(context, program_text)

fps = TextRenderer(context, f"FPS {clock.get_fps()}:.2f", "", 70, (255, 255, 0), glm.vec2(0,0),bold=False,italic=False,visible=True,alignment=TextAlign.LEFT_TOP)

while True:
    # calculate the normalized delta time to affect movement consistently regardless FPS
    NormalizedDeltaTime = clock.tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE
    fps.update_text(f"FPS {clock.get_fps():.2f}")

    # Text requires an orthogonal projection matrix since it's on a 2D space
    TextProjection = glm.ortho(0.0,pygame.display.get_window_size()[0],pygame.display.get_window_size()[1],0)
    program_text["projection"].write(matrix_bytes(TextProjection))

    ## Matrices 
    # NOTE: View and Projection matrices needs to be updated at every loop iteration
    # View Matrix
    # NOTE: now the view matrix will be influenced by the "camera" 
    #       This is all managed now inside the Camera Class
    cam.updateCameraVectors()
    view = cam.GetViewMatrix()

    # Projection Matrix
    # NOTE: here we refer to the camera zoom property to influence the projection matrix 
    projection = glm.perspective(glm.radians(cam.zoom),windowed_size[0] / windowed_size[1], 0.1, 200.0)    

    # here we pass the values for view and projection matrix uniforms
    prog["view"].write(matrix_bytes(view))
    prog["projection"].write(matrix_bytes(projection))

    # ALSO set uniforms for the INSTANCING shader
    progInstances["view"].write(matrix_bytes(view))
    progInstances["projection"].write(matrix_bytes(projection))

    # Model matrix is necessary to be calculated in the main loop if the object is ANIMATED
    # in this case it could be outside the loop but I kept it here for future reference
    model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    model = glm.translate(model,glm.vec3(0.0, -3.0, 0.0))
    model = glm.scale(model, glm.vec3(4.0, 4.0, 4.0))
    prog["model"].write(matrix_bytes(model))

    # render Planet model
    PlanetModel.Draw(prog)

    # render meteorites (WITH instancing)
    RockModel.Draw(progInstances,amount)

    fps.render()
    pygame.display.flip()
    context.clear(color=(0.0, 0.0, 0.0), depth=1.0) # clears the framebuffer (Necessary and also best practice) AND clears the z-buffer setting it to the max
    keys = pygame.key.get_pressed()
    if keys[pygame.K_d]:
        cam.ProcessKeyboard(Camera.Movement.STRIFE_RIGHT,NormalizedDeltaTime)
    if keys[pygame.K_a]:
        cam.ProcessKeyboard(Camera.Movement.STRIFE_LEFT,NormalizedDeltaTime)
    if keys[pygame.K_w]:
        cam.ProcessKeyboard(Camera.Movement.FORWARD,NormalizedDeltaTime)
    if keys[pygame.K_s]:
        cam.ProcessKeyboard(Camera.Movement.BACKWARD,NormalizedDeltaTime)
    if keys[pygame.K_q]:
        cam.ProcessKeyboard(Camera.Movement.LOOK_UP,NormalizedDeltaTime)
    if keys[pygame.K_e]:
        cam.ProcessKeyboard(Camera.Movement.LOOK_DOWN,NormalizedDeltaTime)
    if keys[pygame.K_RIGHT]:
        cam.ProcessKeyboard(Camera.Movement.TURN_RIGHT,NormalizedDeltaTime)
    if keys[pygame.K_LEFT]:
        cam.ProcessKeyboard(Camera.Movement.TURN_LEFT,NormalizedDeltaTime)    
    for event in pygame.event.get():
        if  event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.VIDEORESIZE:
                windowed_size = (event.w, event.h)
                pygame.display.set_mode(windowed_size, flags=screen_flags, vsync=vsync)
                context.viewport = (0, 0, windowed_size[0], windowed_size[1])
        elif event.type == pygame.MOUSEBUTTONDOWN: # when a mouse button is clicked on the window
            if event.button == 1:  # Left mouse button
                # set the mouse invisible and grab the mouse movement (virtual mouse pointer)
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                pygame.display.set_caption("Mouselook enabled - F11 to release")
        elif event.type == pygame.MOUSEWHEEL: # event to capture the mouse wheel
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                lightPos.y += (event.y / 100) # move vertically the light
            if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                lightRadius += (event.y / 10) # change the radius of the light rotation
            if keys[pygame.K_SPACE]:
                cutOffAngle += event.y
                outerCutOffAngle += event.y

            else:
                cam.ProcessMouseScroll(event.y,NormalizedDeltaTime) # event.y is the amount of scroll (up or down)
        elif event.type == pygame.MOUSEMOTION:
            if pygame.event.get_grab():
                relative_x, relative_y = event.rel
                cam.ProcessMouseMovement(relative_x,relative_y,NormalizedDeltaTime)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_F2:
                # start/stop spinning light
                moveLight = not moveLight
                print(f"camera move: {moveLight}")
            elif event.key == pygame.K_F3:
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # increase light Constant attentuation
                    lightConstant += 0.1
                    print(f"light constant attenuation: {lightConstant}")
                else:
                    # increase ambient light
                    if ambientStrength<1.0: ambientStrength += 0.1
                    print(f"ambient light {ambientStrength}")
            elif event.key == pygame.K_F4:
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # decrease light Constant attentuation
                    lightConstant -= 0.1
                    print(f"light constant attenuation: {lightConstant}")
                else:
                    # decrease ambient light
                    if ambientStrength>0.0: ambientStrength -= 0.1
                    print(f"ambient light {ambientStrength}")
            elif event.key == pygame.K_F5:
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # increase light Linear attentuation
                    lightLinear += 0.1
                    print(f"light linear attenuation: {lightLinear}")
                else:
                    # increase diffuse light
                    if diffuseStrength<1.0: diffuseStrength += 0.1
                    print(f"diffuse light: {diffuseStrength}")
            elif event.key == pygame.K_F6:
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # decrease light Linear attentuation
                    lightLinear -= 0.1
                    print(f"light linear attenuation: {lightLinear}")
                else:
                    # decrease diffuse light
                    if diffuseStrength>0.0: diffuseStrength -= 0.1
                    print(f"diffuse light: {diffuseStrength}")
            elif event.key == pygame.K_F7:
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # increase light quadratic attentuation
                    lightQuadratic += 0.1
                    print(f"light quadratic attenuation: {lightQuadratic}")
                else:
                    # increase specular light
                    if specularStrength<1.0: specularStrength += 0.1
                    print(f"specular light: {specularStrength}")
            elif event.key == pygame.K_F8:
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # increase light quadratic attentuation
                    lightQuadratic -= 0.1
                    print(f"light quadratic attenuation: {lightQuadratic}")
                else:
                    # decrease specular light
                    if specularStrength>0.0: specularStrength -= 0.1
                    print(f"specular light: {specularStrength}")
            elif event.key == pygame.K_F10:
                context.wireframe = not context.wireframe
            elif event.key == pygame.K_F11:
                # release the mouse and keyboard and make the mouse visible
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
                pygame.display.set_caption("Click on the window to enable mouselook")
            elif event.key == pygame.K_F12:
                # reload the model and invert the flip texture option
                flip = not ourModel.flipTexture
                del PlanetModel
                del RockModel
                PlanetModel = Model(context,prog,PlanetModelFile,flipTexture=flip)
                RockModel = Model(context,prog,RockModelFile,flipTexture=flip)
            elif event.key == pygame.K_F9:
                if depth_test:
                    context.disable(moderngl.DEPTH_TEST)
                    depth_test = False
                else:
                    context.enable(moderngl.DEPTH_TEST)
                    depth_test = True
