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

Excercise:  Multiple lights
            In the previous chapters we learned a lot about lighting in OpenGL.
            We learned about Phong shading, materials, lighting maps and different types
            of light casters. In this chapter we're going to combine all the previously
            obtained knowledge by creating a fully lit scene with 6 active light sources.
            We are going to simulate a sun-like light as a directional light source,
            4 point lights scattered throughout the scene and we'll be adding a flashlight as well.
            To use more than one light source in the scene we want to encapsulate the lighting calculations into GLSL functions.

            Excercise:  
                        
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

message = "-- c to enable/disable FACE CULLING"
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


# set face culling enabled
context.enable(moderngl.CULL_FACE)
face_culling = True

# Set OpenGL Blending 
context.enable(moderngl.BLEND)
context.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)



# Define Vertex Shader and Fragment Shader in ModernGL (GLSL language)
# ModernGL abstracts vertex and fragment shader as specific parameter of the context program method
# NOTE: This shader is used to render the models on scene and another shader will be used to
#       render the light source cube (the only difference is in the fragment shader)
prog = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

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

uniform sampler2D texture1;

void main()
{             
    FragColor = texture(texture1, TexCoords);
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
cubeVertices = [
    # Back face
    -0.5, -0.5, -0.5,  0.0, 0.0, # Bottom-left
     0.5,  0.5, -0.5,  1.0, 1.0, # top-right
     0.5, -0.5, -0.5,  1.0, 0.0, # bottom-right         
     0.5,  0.5, -0.5,  1.0, 1.0, # top-right
    -0.5, -0.5, -0.5,  0.0, 0.0, # bottom-left
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    # Front face
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
     0.5, -0.5,  0.5,  1.0, 0.0, # bottom-right
     0.5,  0.5,  0.5,  1.0, 1.0, # top-right
     0.5,  0.5,  0.5,  1.0, 1.0, # top-right
    -0.5,  0.5,  0.5,  0.0, 1.0, # top-left
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left
    # Left face
    -0.5,  0.5,  0.5,  1.0, 0.0, # top-right
    -0.5,  0.5, -0.5,  1.0, 1.0, # top-left
    -0.5, -0.5, -0.5,  0.0, 1.0, # bottom-left
    -0.5, -0.5, -0.5,  0.0, 1.0, # bottom-left
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-right
    -0.5,  0.5,  0.5,  1.0, 0.0, # top-right
    # Right face
     0.5,  0.5,  0.5,  1.0, 0.0, # top-left
     0.5, -0.5, -0.5,  0.0, 1.0, # bottom-right
     0.5,  0.5, -0.5,  1.0, 1.0, # top-right         
     0.5, -0.5, -0.5,  0.0, 1.0, # bottom-right
     0.5,  0.5,  0.5,  1.0, 0.0, # top-left
     0.5, -0.5,  0.5,  0.0, 0.0, # bottom-left     
    # Bottom face
    -0.5, -0.5, -0.5,  0.0, 1.0, # top-right
     0.5, -0.5, -0.5,  1.0, 1.0, # top-left
     0.5, -0.5,  0.5,  1.0, 0.0, # bottom-left
     0.5, -0.5,  0.5,  1.0, 0.0, # bottom-left
    -0.5, -0.5,  0.5,  0.0, 0.0, # bottom-right
    -0.5, -0.5, -0.5,  0.0, 1.0, # top-right
    # Top face
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
     0.5,  0.5,  0.5,  1.0, 0.0, # bottom-right
     0.5,  0.5, -0.5,  1.0, 1.0, # top-right     
     0.5,  0.5,  0.5,  1.0, 0.0, # bottom-right
    -0.5,  0.5, -0.5,  0.0, 1.0, # top-left
    -0.5,  0.5,  0.5,  0.0, 0.0  # bottom-left        
]


# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '32f': This is the format string. It specifies that we want to pack 32 floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
cubeVertices_binaryformat = struct.pack(f"{len(cubeVertices)}f",*cubeVertices)
# now for the vertices of the floor



# Define VBO (Vertex Buffer Object) containing vertex data
vbo_cubeVertices = context.buffer(cubeVertices_binaryformat)


### load image and create texture

# Load image
cubeTexture   = pygame.image.load("./assets/marble.jpg")

# Convert image into a stream of bytes with 4 components (RGBA) and flip the image
# (OpenGL expects flipped coordinates) compared to a normal image
cubeTexture_data = pygame.image.tobytes(cubeTexture,"RGBA",True)

# load the texture within the OpenGL context
cubeTexture_Texture = context.texture(cubeTexture.get_size(),4,cubeTexture_data)

# Access the uniform variable and assign it a texture unit (0) (it will be then used by the texture binding)
prog["texture1"] = 0






# VBO parameters to be passed to the VAO
# This is what in modernGL is defined as "multiple buffers for all input variables"
# meaning that each VBO buffer is described as a tuple in a list
# elements of the tuple describes
# 1) Vertex Buffer Object in input
# 2) type of input parameters (3f in this case corresponds to a 3vec input) defined in shaders
# 3) name of the input parameter in the related shader (aPos in this case)
# NOTE: These parameters are the same also for the light source VAO
vbo_cubeVertices_parameters = [
    (vbo_cubeVertices,"3f 2f","aPos","aTexCoords")
]

# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
# NOTE: ebo with indices is not used in this example
vao_cubeVertices = context.vertex_array(prog,vbo_cubeVertices_parameters)



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



pygame.display.set_caption(f"Click on the window to enable mouselook {message}")


while True:

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

    ## render cube
    # bind the texture to the texture unit related to the uniform variable
    cubeTexture_Texture.use(location=0)
    model:glm.mat4 = glm.mat4(1.0)
    model = glm.translate(model, glm.vec3(-1.0, 0.0, -1.0))
    prog["model"].write(matrix_bytes(model))
    vao_cubeVertices.render(moderngl.TRIANGLES)

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
                pygame.display.set_caption(f"Mouselook enabled - F11 to release {message}")
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
                pygame.display.set_caption(f"Click on the window to enable mouselook {message}")
            elif event.key == pygame.K_F9:
                if depth_test:
                    context.disable(moderngl.DEPTH_TEST)
                    depth_test = False
                else:
                    context.enable(moderngl.DEPTH_TEST)
                    depth_test = True
            elif event.key == pygame.K_c:
                if face_culling:
                    context.disable(moderngl.CULL_FACE)
                    face_culling = False
                else:
                    context.enable(moderngl.CULL_FACE)
                    face_culling = True
