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

Excercise:  Model loading with pyAssimp + multiple light sources
            NOTE: Lights are not included in this scenario
            NOTE: Due to OpenGL shader optimization, aNormal (vertex normals) were not used
                  this made necessary to comment out all references about the normals in the 
                  Vertex Data structure and VBO creation in the following portions:
                  Vertex dataclass
                  Mesh.setupMesh
                  Model.processMesh
            Excercise:  
                        
                        Controls
                        ----------------------------------------
                        F2: start/stop spinning light
                        F3: increase source light ambient 
                        F4: decrease source light ambient
                        F5: increase source light diffuse
                        F6: decrease source light diffuse
                        F7: increase specular light
                        F8: decrease specular light
                        F3 + CTRL: Increase constant attenuation
                        F4 + CTRL: Decrease constant attenuation
                        F5 + CTRL: Increase linear attenuation
                        F6 + CTRL: Decrease linear attenuation
                        F7 + CTRL: Increase quadric attenuation
                        F8 + CTRL: Decrease quadric attenuation  
                        F9: enable/disable depth test
                        F10: enable/disable wireframe                  
                        F11: Release mouselook
                        F12: Flip textures                               
                        Mouse wheel + SPACE: Increase/decrease spotlight cone
                        Mouse wheel + SHIFT: Increase/decrease light Y position
                        Mouse wheel + CTRL: Increase/decrease light rotation radius
                        0: restore normal framebuffer view
                        1: framebuffer effect: inversion shader
                        2: framebuffer effect: grayscale shader
                        3: framebuffer effect: kernel base shader
                        4: framebuffer effect: blur shader
                        5: framebuffer effect: edges shader
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


### Mesh data structure
@dataclass
class Vertex:
    Position: glm.vec3
    Normal: glm.vec3
    TexCoords: glm.vec2

    def size(self):
        size = ( glm.sizeof(self.Position) + glm.sizeof(self.Normal) + glm.sizeof(self.TexCoords) ) // 4
        #size = ( glm.sizeof(self.Position) + glm.sizeof(self.TexCoords) ) // 4
        return size

@dataclass
class Texture:
    id: moderngl.Texture
    type: str
    path: str

class Mesh():
    def __init__(self,context:moderngl.Context,program:moderngl.Program,vertices:list[Vertex],indices:list[int],textures:list[Texture]):
        self.context = context # in ModernGL is necessary to create a VAO
        self.program = program # in ModernGL is necessary to create a VAO
        self.vertices = vertices
        self.indices = indices
        self.textures = textures
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.setupMesh()

    def Draw(self, shader: moderngl.Program):
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
                uniform_name  = f"material.diffuse"
            elif texture.type == "texture_specular":
                specularNr += 1
                uniform_name  = f"material.specular"
            # check if the uniform name assembled is an available uniform in the shader
            # if yes assign the uniform to use the texture unit previously selected
            if uniform_name in shader:
                shader[uniform_name] = texture_unit
            
        
        prog["material.shininess"] = 32.0
        # Render the mesh VAO with the passed shader
        self.vao.render()


    def setupMesh(self):
        ## Create VBO data
        # create a mutable bytearray to contain each vertex data in bytes, also calculate the format of ech vertex (8 floats)
        vertices_binaryformat_array = bytearray()
        vertex_format = f"{self.vertices[0].size()}f"
        # iterate through each vertex and convert the glm vectors to list elements and concatenate them in a single list
        # the single list of all elements of a single vertex is then converted to byte and concatenated to the initial bytearray
        for vertex in self.vertices:
            components = list(vertex.Position) + list(vertex.Normal) + list(vertex.TexCoords)
            #components = list(vertex.Position) + list(vertex.TexCoords)
            packed_vertex = struct.pack(vertex_format, *components)
            vertices_binaryformat_array.extend(packed_vertex) # Append bytes to bytearray
        vertices_binaryformat = bytes(vertices_binaryformat_array) # convert mutable byterray to immutable bytes
        self.vbo = self.context.buffer(vertices_binaryformat)

        ## Create EBO data
        indices_binaryformat = struct.pack(f"{len(self.indices)}I",*self.indices)
        self.ebo = self.context.buffer(indices_binaryformat)

        ## Create VAO
        vbo_string = f"{glm.sizeof(self.vertices[0].Position)//4}f {glm.sizeof(self.vertices[0].Normal)//4}f {glm.sizeof(self.vertices[0].TexCoords)//4}f"
        vbo_parameters = [(self.vbo,vbo_string,"aPos","aNormal","aTexCoords")]
        self.vao = self.context.vertex_array(self.program,vbo_parameters,self.ebo)
        # NOTE Create VBO parameters without aNormal (not needed in this example)
        #vbo_string_no_normal = f"{glm.sizeof(self.vertices[0].Position)//4}f {glm.sizeof(self.vertices[0].TexCoords)//4}f"
        #vbo_parameters_no_normal = [(self.vbo,vbo_string_no_normal,"aPos","aTexCoords")]
        #self.vao = self.context.vertex_array(self.program,vbo_parameters_no_normal,self.ebo)

class Model():
    def __init__(self,context:moderngl.Context,program:moderngl.Program,path:str,flipTexture=True):
        self.context = context
        self.program = program
        self.meshes:list[Mesh] = []
        self.directory = ""
        self.path = path
        self.textures_loaded:list[Texture] = []
        self.flipTexture = flipTexture # flip or not Mesh Textures 
        self.loadModel(path=self.path)
        
        

    def loadModel(self,path:str=""):
        self.meshes.clear()
        if path == "":
            path = self.path
        with pyassimp.load(filename=path,processing=pyassimp.postprocess.aiProcess_Triangulate | pyassimp.postprocess.aiProcess_FlipUVs | pyassimp.postprocess.aiProcess_GenSmoothNormals) as scene:
            if scene:
                self.directory = os.path.dirname(path)
            self.processNode(node=scene.rootnode)
            
    
    def Draw(self,shader:moderngl.Program):
        for mesh in self.meshes:
            mesh.Draw(shader=shader)

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
            new_normal = glm.vec3(normal)
            if textureCoords:
                new_TexCoords = glm.vec2(textureCoords)
            else:
                new_TexCoords = glm.vec2(0,0)
            vertices.append(Vertex(Position=new_position,Normal=new_normal,TexCoords=new_TexCoords))
            #vertices.append(Vertex(Position=new_position,TexCoords=new_TexCoords))
        
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
        new_mesh = Mesh(self.context,self.program,vertices,indices,textures)
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
# Depth buffer bits
pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
# Stencil buffer bits <-- IMPORTANT
pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)

# Create and initializize display
screen_flags = pygame.OPENGL | pygame.RESIZABLE | pygame.DOUBLEBUF
screen_display = pygame.display.set_mode(windowed_size,flags=screen_flags,vsync=vsync)
### Multisampling 
# This tells Pygame (and underlying SDL) to enable multisample buffers. A value of 1 requests their availability
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
# This specifies the desired number of samples for MSAA (e.g., N could be 2, 4, 8, etc.). The actual number of samples you get depends on your GPU's capabilities.
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES,4)


### OpenGL section

# ModernGL create a context : a state machine or a container for OpenGL
context = moderngl.create_context()

### Enable DEPTH TESTING
# When depth testing is enabled, OpenGL (and thus ModernGL) uses a depth buffer to determine which fragments (pixels) should be drawn on the screen.
# Each fragment has a depth value, which represents its distance from the viewer.   
# https://moderngl.readthedocs.io/en/latest/reference/context.html#Context.enable
context.enable(moderngl.DEPTH_TEST)
depth_test = True
# Enable/disable multisample mode (GL_MULTISAMPLE)
multisample = True
context.multisample = True
num_fbo_msaa_samples = 4

# Define Vertex Shader and Fragment Shader in ModernGL (GLSL language)
# ModernGL abstracts vertex and fragment shader as specific parameter of the context program method
# NOTE: This shader is used to render the models on scene and another shader will be used to
#       render the light source cube (the only difference is in the fragment shader)
prog = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords; // Texture coordinate for the diffuse sampler to pass to the fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix; // used with the normal vectors

out vec3 Normal;  // normal vector used to calculate the angle of diffuse light per fragment
out vec3 FragPos; // position of the fragment in the world space (retrieved by multiplying the vertex position with model matrix)
out vec2 TexCoords; // Texture coordinate for the diffuse sampler to pass to the fragment shader

void main()
{
    // note that we read the multiplication from right to left
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    Normal = normalMatrix * aNormal;
    FragPos = vec3(model * vec4(aPos,1.0)); // Fragment position in the world calculated by the model matrix
    TexCoords = aTexCoords; // Texture coordinate for the diffuse sampler to pass to the fragment shader
}
''',

    fragment_shader='''
#version 330 core
struct Material {
    sampler2D diffuse;  // diffuse is a diffuse map texture
    sampler2D specular; // specular is now a specular map texture
    float     shininess;
}; 

// multiple lights
struct DirLight {
    vec3 direction;
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};  
uniform DirLight dirLight;
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir);  

struct PointLight {    
    vec3 position;
    
    float constant;
    float linear;
    float quadratic;  

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};  
#define NR_POINT_LIGHTS 4
uniform PointLight pointLights[NR_POINT_LIGHTS];
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);  

struct SpotLight {
    vec3  position;
    vec3  direction;
    float cutOff;       // inner spotlight cone
    float outerCutOff;  // outer spotlight cone
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    // attenuation values 
    float constant;
    float linear;
    float quadratic;
};
uniform SpotLight spotLight;
vec3 CalcSpotLight(SpotLight light, vec3 normalized, vec3 FragPos, vec3 viewDirection);

out vec4 FragColor;

in vec3 Normal;  // normal vector used to calculate the angle of diffuse light per fragment
in vec3 FragPos; // This in variable will be interpolated from the 3 world position vectors of the triangle to form the FragPos vector that is the per-fragment world position
in vec2 TexCoords; // Texture coordinate for the diffuse sampler received from the vertex shader

uniform vec3 viewPos; // position of the viewer (camera)

uniform Material material; // contain data for the material surface
uniform bool spotLightOn;
void main()
{

        // properties
        vec3 normalized = normalize(Normal);
        vec3 viewDirection = normalize(viewPos - FragPos);

        // phase 1: Directional lighting
        vec3 result = CalcDirLight(dirLight, normalized, viewDirection);
        // phase 2: Point lights
        for(int i = 0; i < NR_POINT_LIGHTS; i++)
            result += CalcPointLight(pointLights[i], normalized, FragPos, viewDirection); 
        if (spotLightOn)
        {   
            // phase 3: Spot light
            result += CalcSpotLight(spotLight, normalized, FragPos, viewDirection);    
        }
        FragColor = vec4(result, 1.0);
}

vec3 CalcDirLight(DirLight light, vec3 normalized, vec3 viewDirection)
{
    // Ambient Lighting
    // Normalized vector is calculated out of this function
    vec3 ambientComponent = light.ambient * vec3(texture(material.diffuse, TexCoords)); // using diffuse material color from the texture sampler
    
    // Diffuse Lighting
    vec3 lightDirection = normalize(-light.direction);
    float diffuseImpact = max(dot(normalized,lightDirection),0.0);
    vec3 diffuseComponent = light.diffuse * diffuseImpact * vec3(texture(material.diffuse, TexCoords)); // we are using a diffuse map to sample the color

    // Specular lighting    
    // view direction is calculated out of this function
    vec3 reflectDirection = reflect(-lightDirection, normalized); 
    float specular = pow(max(dot(viewDirection, reflectDirection), 0.0), material.shininess);
    vec3 specularComponent = light.specular * specular * vec3(texture(material.specular,TexCoords)); // we are using a specular map now instead of a single vector3

    vec3 result = ambientComponent + diffuseComponent + specularComponent;
    return result;
}

vec3 CalcPointLight(PointLight light, vec3 normalized, vec3 FragPos, vec3 viewDirection)
{   
    // Ambient Lighting
    vec3 ambientComponent = light.ambient * vec3(texture(material.diffuse, TexCoords)); // using diffuse material color from the texture sampler

    // Diffuse Lighting
    vec3 lightDirection = normalize(light.position - FragPos);
    float diffuseImpact = max(dot(normalized,lightDirection),0.0);
    vec3 diffuseComponent = light.diffuse * diffuseImpact * vec3(texture(material.diffuse, TexCoords)); // we are using a diffuse map to sample the color
    
    // Specular lighting

    vec3 reflectDirection = reflect(-lightDirection, normalized); 
    float specular = pow(max(dot(viewDirection, reflectDirection), 0.0), material.shininess);
    vec3 specularComponent = light.specular * specular * vec3(texture(material.specular,TexCoords)); // we are using a specular map now instead of a single vector3

    // calculate the attenuation
    float distance    = length(light.position - FragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    ambientComponent  *= attenuation; 
    diffuseComponent  *= attenuation;
    specularComponent *= attenuation;  

    // Calculate the result by merging ambient and diffuse lights
    vec3 result = ambientComponent + diffuseComponent + specularComponent;
    return result;
}

vec3 CalcSpotLight(SpotLight light, vec3 normalized, vec3 FragPos, vec3 viewDirection)
{   

    // Ambient Lighting
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords)); // using diffuse material color from the texture sampler

    // Diffuse Lighting
    // When calculating lighting we usually do not care about the magnitude of a vector or their position, we only care about their direction
    // calculations are done with unit vectors since it simplifies most calculations
    //e first negate the light.direction vector. The lighting calculations we used so far expect the light direction to be a
    // direction from the fragment towards the light source, but people generally prefer to specify a directional light as
    // a global direction pointing from the light source
    vec3 lightDirection = normalize(light.position - FragPos);
    // calculate the diffuse impact of the light on the current fragment by taking the dot product between the norm and lightDir vectors
    // The resulting value is then multiplied with the light's color to get the diffuse component
    float diffuseImpact = max(dot(normalized,lightDirection),0.0);
    vec3 diffuseComponent = light.diffuse * diffuseImpact * vec3(texture(material.diffuse, TexCoords)); // we are using a diffuse map to sample the color
    
    // Specular lighting
    vec3 reflectDirection = reflect(-lightDirection, normalized); 
    float spec = pow(max(dot(viewDirection, reflectDirection), 0.0), material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular,TexCoords)); // we are using a specular map now instead of a single vector3

    // spotlight calculations : ambient light is excluded to have always some little light in the scene
    float theta = dot(lightDirection, normalize(-light.direction));
    float epsilon   = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    diffuseComponent *= intensity;
    specular *= intensity;
    
    // calculate the attenuation
    float distance    = length(light.position - FragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    ambient  *= attenuation; 
    diffuseComponent  *= attenuation;
    specular *= attenuation;  

    // Calculate the result by merging ambient and diffuse lights (object color is removed, color is provided by the material properties)
    vec3 result = ambient + diffuseComponent + specular;
    return result;
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
cam = Camera(glm.vec3(0.0, 0.0, 3.0))


# Reference variables for Delta time
FRAMERATE_REFERENCE = 60
FRAMERATE = 60



# Start with a moving light
moveLight = True
pygame.display.set_caption("Click on the window to enable mouselook")

#modelFile = "./model3d/cottage_blender3.obj"
modelFile = "./model3d/backpack.obj"
ourModel = Model(context,prog,modelFile)

############ LIGHT TEST
# Directional Light parameters
DirectionalAmbientStrength = 0.1
DirectionalDiffuseStrength = 0.1
DirectionalSpecularStrength = 0.1

# Point light parameters
PointAmbientStrength = 0.1
PointDiffuseStrength = 0.8
PointSpecularStrength = 1.0
lightYdelta = 0
lightRadiusDelta = 0
# attenuation parameters
PointLightConstant = 1.0
PointLightLinear = 0.09
PointLightQuadratic = 0.032

# Spot Light  parameters
SpotAmbientStrength = 0.0
SpotDiffuseStrength = 1.0
SpotSpecularStrength = 1.0
SpotLightConstant = 1.0
SpotLightLinear = 0.09
SpotLightQuadratic = 0.032
SpotCutOffAngle = 12.5
SpotOuterCutOffAngle = 17
spotLightOn = False
#######################


####################### FRAMEBUFFER routine
# Framebuffers requires at least one buffer between:
# color     -> MANDATORY, 2D Array containing color information (including Alpha)
# depth     -> 2D array containing depth value, to handle occlusion, not creating it makes depth testing lost
# stencil   -> 2D array typicall of 8 bits, used to control drawing operations
# We have to attach at least one buffer (color, depth or stencil buffer).
# There should be at least one color attachment.
# All attachments should be complete as well (reserved memory).
# Each buffer should have the same number of samples.
# Depth and stencil buffer can be combined together and GPUs are optimized to handle it <-- at this time modernGL cannot do it
# look at : https://github.com/moderngl/moderngl/pull/725 support for stencil will be added!
# 
offscreen_texture = context.texture(windowed_size, 4)
offscreen_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
# single depth buffer 
# stencil support will be added soon to modernGL
offscreen_depth = context.depth_texture(windowed_size)
framebuffer_object = context.framebuffer(color_attachments=[offscreen_texture],depth_attachment=offscreen_depth)

# 1. Create attachments for a MULTISAMPLED FBO
msaa_color_texture = context.texture(windowed_size, 4, samples=num_fbo_msaa_samples)
msaa_depth_attachment = context.depth_texture(windowed_size, samples=num_fbo_msaa_samples)

# 2. Create the MULTISAMPLED FBO (Anti Aliasing will be rendered here)
msaa_framebuffer_object = context.framebuffer(color_attachments=[msaa_color_texture],depth_attachment=msaa_depth_attachment)

fbo_vertex_shader = '''
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); 
    TexCoords = aTexCoords;
}  
'''
fbo_fragment_shader='''
#version 330 core
out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D screenTexture;

void main()
{ 
    FragColor = texture(screenTexture, TexCoords);
}
'''

fbo_fragment_shader_inversion='''
#version 330 core
out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D screenTexture;

void main()
{ 
    FragColor = vec4(vec3(1.0 - texture(screenTexture, TexCoords)), 1.0);
}
'''

fbo_fragment_shader_grayscale = '''
#version 330 core
out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D screenTexture;

void main()
{
    FragColor = texture(screenTexture, TexCoords);
    float average = 0.2126 * FragColor.r + 0.7152 * FragColor.g + 0.0722 * FragColor.b;
    FragColor = vec4(average, average, average, 1.0);
} 
'''

fbo_fragment_shader_kernel = '''
#version 330 core
out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D screenTexture;

const float offset = 1.0 / 300.0;  

void main()
{
    vec2 offsets[9] = vec2[](
        vec2(-offset,  offset), // top-left
        vec2( 0.0f,    offset), // top-center
        vec2( offset,  offset), // top-right
        vec2(-offset,  0.0f),   // center-left
        vec2( 0.0f,    0.0f),   // center-center
        vec2( offset,  0.0f),   // center-right
        vec2(-offset, -offset), // bottom-left
        vec2( 0.0f,   -offset), // bottom-center
        vec2( offset, -offset)  // bottom-right    
    );

    float kernel[9] = float[](
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    );
    
    vec3 sampleTex[9];
    for(int i = 0; i < 9; i++)
    {
        sampleTex[i] = vec3(texture(screenTexture, TexCoords.st + offsets[i]));
    }
    vec3 col = vec3(0.0);
    for(int i = 0; i < 9; i++)
        col += sampleTex[i] * kernel[i];
    
    FragColor = vec4(col, 1.0);
}
'''

fbo_fragment_shader_kernel_blur = '''
#version 330 core
out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D screenTexture;

const float offset = 1.0 / 300.0;  

void main()
{
    vec2 offsets[9] = vec2[](
        vec2(-offset,  offset), // top-left
        vec2( 0.0f,    offset), // top-center
        vec2( offset,  offset), // top-right
        vec2(-offset,  0.0f),   // center-left
        vec2( 0.0f,    0.0f),   // center-center
        vec2( offset,  0.0f),   // center-right
        vec2(-offset, -offset), // bottom-left
        vec2( 0.0f,   -offset), // bottom-center
        vec2( offset, -offset)  // bottom-right    
    );

    float kernel[9] = float[](
        1.0 / 16, 2.0 / 16, 1.0 / 16,
        2.0 / 16, 4.0 / 16, 2.0 / 16,
        1.0 / 16, 2.0 / 16, 1.0 / 16  
    );
    
    vec3 sampleTex[9];
    for(int i = 0; i < 9; i++)
    {
        sampleTex[i] = vec3(texture(screenTexture, TexCoords.st + offsets[i]));
    }
    vec3 col = vec3(0.0);
    for(int i = 0; i < 9; i++)
        col += sampleTex[i] * kernel[i];
    
    FragColor = vec4(col, 1.0);
}
'''

fbo_fragment_shader_kernel_edges = '''
#version 330 core
out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D screenTexture;

const float offset = 1.0 / 300.0;  

void main()
{
    vec2 offsets[9] = vec2[](
        vec2(-offset,  offset), // top-left
        vec2( 0.0f,    offset), // top-center
        vec2( offset,  offset), // top-right
        vec2(-offset,  0.0f),   // center-left
        vec2( 0.0f,    0.0f),   // center-center
        vec2( offset,  0.0f),   // center-right
        vec2(-offset, -offset), // bottom-left
        vec2( 0.0f,   -offset), // bottom-center
        vec2( offset, -offset)  // bottom-right    
    );

    float kernel[9] = float[](
        1.0 , 1.0 , 1.0,
        1.0 , -8.0 , 1.0,
        1.0 , 1.0 , 1.0  
    );
    
    vec3 sampleTex[9];
    for(int i = 0; i < 9; i++)
    {
        sampleTex[i] = vec3(texture(screenTexture, TexCoords.st + offsets[i]));
    }
    vec3 col = vec3(0.0);
    for(int i = 0; i < 9; i++)
        col += sampleTex[i] * kernel[i];
    
    FragColor = vec4(col, 1.0);
}
'''

prog_fbo = context.program(vertex_shader=fbo_vertex_shader,fragment_shader=fbo_fragment_shader)
prog_fbo_inversion = context.program(vertex_shader=fbo_vertex_shader,fragment_shader=fbo_fragment_shader_inversion)
prog_fbo_grayscale = context.program(vertex_shader=fbo_vertex_shader,fragment_shader=fbo_fragment_shader_grayscale)
prog_fbo_kernel = context.program(vertex_shader=fbo_vertex_shader,fragment_shader=fbo_fragment_shader_kernel)
prog_fbo_kernel_blur = context.program(vertex_shader=fbo_vertex_shader,fragment_shader=fbo_fragment_shader_kernel_blur)
prog_fbo_kernel_edges = context.program(vertex_shader=fbo_vertex_shader,fragment_shader=fbo_fragment_shader_kernel_edges)


# Full-screen quad vertices (x, y) and UV coordinates (u, v)
quad_buffer_data = [
    # positions   # texCoords
    -1.0,  1.0,  0.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,

    -1.0,  1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0
]

# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '24f': This is the format string. It specifies that we want to pack 24 floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
fbo_vertices_binaryformat = struct.pack(f"{len(quad_buffer_data)}f",*quad_buffer_data)

# Define VBO (Vertex Buffer Object) containing vertex data
fbo_vbo = context.buffer(fbo_vertices_binaryformat)


# VBO parameters to be passed to the VAO
# This is what in modernGL is defined as "multiple buffers for all input variables"
# meaning that each VBO buffer is described as a tuple in a list
# elements of the tuple describes
# 1) Vertex Buffer Object in input
# 2) type of input parameters (3f in this case corresponds to a 3vec input) defined in shaders
# 3) name of the input parameter in the related shader (aPos in this case)
# NOTE: These parameters are the same also for the light source VAO
fbo_vbo_parameters = [
    (fbo_vbo,"2f 2f","aPos","aTexCoords")
]

# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
# NOTE: ebo with indices is not used in this example
fbo_vao = context.vertex_array(prog_fbo,fbo_vbo_parameters)
fbo_vao_inversion = context.vertex_array(prog_fbo_inversion,fbo_vbo_parameters)
fbo_vao_grayscale = context.vertex_array(prog_fbo_grayscale,fbo_vbo_parameters)
fbo_vao_kernel = context.vertex_array(prog_fbo_kernel,fbo_vbo_parameters)
fbo_vao_kernel_blur = context.vertex_array(prog_fbo_kernel_blur,fbo_vbo_parameters)
fbo_vao_kernel_edges = context.vertex_array(prog_fbo_kernel_edges,fbo_vbo_parameters)



fbo_current_vao = fbo_vao
fbo_current_program = prog_fbo
#######################

while True:
    ### FRAMEBUFFER object set in use and clear (THIS TIME WE USE THE MultiSampled Anti Aliasing MSAA FBO)
    # Later we will copy the MSAA FBO into a standard FBO that supports postprocessing
    # MultiSampled Framebuffers and Textures cannot be passed to a shader for postprocessing
    msaa_framebuffer_object.use()
    # clear the framebuffer
    msaa_framebuffer_object.clear(color=(1, 1, 1, 1.0), depth=1.0)

    
    # Directional Light (1 for the whole scene)
    prog["dirLight.direction"] = glm.vec3(-0.2, -1.0, -0.3)
    prog["dirLight.ambient"] = glm.vec3(DirectionalAmbientStrength, DirectionalAmbientStrength, DirectionalAmbientStrength)
    prog["dirLight.diffuse"] = glm.vec3(DirectionalDiffuseStrength, DirectionalDiffuseStrength, DirectionalDiffuseStrength)
    prog["dirLight.specular"] = glm.vec3(DirectionalSpecularStrength, DirectionalSpecularStrength, DirectionalSpecularStrength)

    # Point lights
    for _ in range(PointLights):
        #prog[f"pointLights[{_}].position"] = pointLightPositions[_] # we are setting the light position in the lightbox movement portion
        prog[f"pointLights[{_}].ambient"] = glm.vec3(PointAmbientStrength, PointAmbientStrength, PointAmbientStrength)
        prog[f"pointLights[{_}].diffuse"] = glm.vec3(PointDiffuseStrength, PointDiffuseStrength, PointDiffuseStrength)
        prog[f"pointLights[{_}].specular"] = glm.vec3(PointSpecularStrength, PointSpecularStrength, PointSpecularStrength)
        # attenuation values for light point
        prog[f"pointLights[{_}].constant"] = PointLightConstant
        prog[f"pointLights[{_}].linear"] = PointLightLinear
        prog[f"pointLights[{_}].quadratic"] = PointLightQuadratic

    # Spot Light
    # Pass light source parameters to the shaders
    prog["spotLight.ambient"] = glm.vec3(SpotAmbientStrength, SpotAmbientStrength, SpotAmbientStrength)
    prog["spotLight.diffuse"] = glm.vec3(SpotDiffuseStrength, SpotDiffuseStrength, SpotDiffuseStrength)
    prog["spotLight.specular"] = glm.vec3(SpotSpecularStrength, SpotSpecularStrength, SpotSpecularStrength)
    # attenuation values for light point
    prog["spotLight.constant"] = SpotLightConstant
    prog["spotLight.linear"] = SpotLightLinear
    prog["spotLight.quadratic"] = SpotLightQuadratic
    prog["spotLight.position"] = cam.cameraPos
    prog["spotLight.direction"] = cam.cameraTarget
    prog["spotLight.cutOff"] = glm.cos(glm.radians(SpotCutOffAngle))
    prog["spotLight.outerCutOff"] = glm.cos(glm.radians(SpotOuterCutOffAngle))

    # calculate the normalized delta time to affect movement consistently regardless FPS
    NormalizedDeltaTime = pygame.time.Clock().tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE

    ## Matrices 
    # NOTE: View and Projection matrices needs to be updated at every loop iteration
    # View Matrix
    # NOTE: now the view matrix will be influenced by the "camera" 
    #       This is all managed now inside the Camera Class
    cam.updateCameraVectors()
    view = cam.GetViewMatrix()

    # Projection Matrix
    # NOTE: here we refer to the camera zoom property to influence the projection matrix 
    projection = glm.perspective(glm.radians(cam.zoom),windowed_size[0] / windowed_size[1], 0.1, 100.0)    

    # here we pass the values for view and projection matrix uniforms
    prog["view"].write(matrix_bytes(view))
    prog["projection"].write(matrix_bytes(projection))

    # load the viewPos uniform with the position of the camera to be used in specular lighting calculations
    prog["viewPos"].value = cam.cameraPos



    model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    model = glm.translate(model,glm.vec3(0.0, 0.0, 0.0))
    model = glm.scale(model, glm.vec3(1.0, 1.0, 1.0))
    prog["model"].write(matrix_bytes(model))

    ## -> calculate the Normal Matrix
    # NOTE: Normal matrix needs to be calculated before rendering the models for EACH model
    #       the normal matrix calculation and update before rendering each cube.
    #       This ensures the correct normal transformation is applied to the current cube’s model matrix.
    normalMatrix = glm.mat3(glm.transpose(glm.inverse(model)))
    # load the normal matrix calculated before
    # NOTE: This way we calculate one normal matrix for each object rather than for each vertex as it is in learnopengl
    prog["normalMatrix"].write(matrix_bytes(normalMatrix))

    prog["spotLightOn"] = spotLightOn

    # render model
    ourModel.Draw(prog)


    # Here we pass matrices data for the light box rendering (view and projection matrices)
    progLight["view"].write(matrix_bytes(view))
    progLight["projection"].write(matrix_bytes(projection))
    # here we move the lightbox in circle if enabled by a radius (lightRadius) defined out of the loop
    # and controllable via user input
    for _ in range(PointLights):
        if moveLight:
            lightRadius = glm.length(glm.vec2(pointLightPositions[_].x, pointLightPositions[_].z)) + lightRadiusDelta
            lightX = math.sin(pygame.time.get_ticks()/1000) * lightRadius
            lightZ = math.cos(pygame.time.get_ticks()/1000) * lightRadius
            lightY = pointLightPositions[_].y + lightYdelta
            # Now pass the value of the light position vector as uniform
            lightPos = glm.vec3(lightX,lightY,lightZ)
        # light position is used to calculate the direction of the light in diffuse lighting and the
        # distance of an object from the light for the attenuation
        else:
            lightPos = pointLightPositions[_]
        prog[f"pointLights[{_}].position"] = lightPos
        modelLight = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
        modelLight = glm.translate(modelLight,lightPos) # put the light source in position
        modelLight = glm.scale(modelLight, glm.vec3(0.2))
        progLight["model"].write(matrix_bytes(modelLight))
        # render the lightbox
        lightvao.render()


    # 2. RESOLVE MULTISAMPLED FBO TO REGULAR FBO
    # This transfers the antialiased image from msaa_framebuffer_object to resolve_fbo (and thus resolve_texture).
    context.copy_framebuffer(dst=framebuffer_object, src=msaa_framebuffer_object)

    ### FRAMEBUFFER render back to screen with default framebuffer
    # context.screen is the default framebuffer
    context.screen.use()
    context.viewport = (0, 0, windowed_size[0], windowed_size[1])
    # clear default framebuffer to avoid previous artifacts
    context.screen.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
    offscreen_texture.use(location=0)
    fbo_current_program["screenTexture"].value = 0
    # disable Depth test for the framebuffer object to screen render and re-enable it after the rendering
    context.disable(moderngl.DEPTH_TEST)
    fbo_current_vao.render()
    if depth_test : 
        context.enable(moderngl.DEPTH_TEST)

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
                lightYdelta += (event.y / 10) # move vertically the light
            elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                lightRadiusDelta += (event.y / 10) # change the radius of the light rotation
            elif keys[pygame.K_SPACE]:
                SpotCutOffAngle += event.y
                SpotOuterCutOffAngle += event.y

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
            elif event.key == pygame.K_F9:
                if depth_test:
                    context.disable(moderngl.DEPTH_TEST)
                    depth_test = False
                else:
                    context.enable(moderngl.DEPTH_TEST)
                    depth_test = True
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
                del ourModel
                ourModel = Model(context,prog,modelFile,flipTexture=flip)
            elif event.key == pygame.K_l:
                spotLightOn = not spotLightOn
            elif event.key == pygame.K_0:
                fbo_current_vao = fbo_vao
                fbo_current_program = prog_fbo
            elif event.key == pygame.K_1:
                fbo_current_vao = fbo_vao_inversion
                fbo_current_program = prog_fbo_inversion
            elif event.key == pygame.K_2:
                fbo_current_vao = fbo_vao_grayscale
                fbo_current_program = prog_fbo_grayscale
            elif event.key == pygame.K_3:
                fbo_current_vao = fbo_vao_kernel
                fbo_current_program = prog_fbo_kernel
            elif event.key == pygame.K_4:
                fbo_current_vao = fbo_vao_kernel_blur
                fbo_current_program = prog_fbo_kernel_blur
            elif event.key == pygame.K_5:
                fbo_current_vao = fbo_vao_kernel_edges
                fbo_current_program = prog_fbo_kernel_edges
            elif event.key == pygame.K_6:
                multisample = not multisample
                context.multisample = multisample
                print(f"Multisample: {multisample}")