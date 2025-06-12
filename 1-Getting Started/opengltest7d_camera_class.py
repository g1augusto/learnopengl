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

Excercise:  Walk around (MADE WITH CAMERA CLASS)
            Swinging the camera around a scene is fun, but it's more fun to do all the movement ourselves!
            NOTE: we have
            - keyboard look with
                WASD + QE and left and right cursors
            - Mouselook with
                Catch the mouse by clicking on the window
                mouse is invisible -> this trigger the use of virtual mouse
                virtual mouse can move beyond window's constarints 
                F11 to release the mouse
'''

from enum import Enum # to define movement enum class
import math
import ctypes # for pyglm bytes conversion
import struct
import sys
import pygame
import moderngl
import glm

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
            self.yaw += turnVelocity * NormalizedDeltaTime
        elif direction == Camera.Movement.TURN_LEFT:
            self.yaw -= turnVelocity * NormalizedDeltaTime
        elif direction == Camera.Movement.STRIFE_RIGHT:
            self.cameraPos += glm.normalize(glm.cross(self.cameraTarget,self.cameraUp)) * velocity
        elif direction == Camera.Movement.STRIFE_LEFT:
            self.cameraPos -= glm.normalize(glm.cross(self.cameraTarget,self.cameraUp)) * velocity
        elif direction == Camera.Movement.LOOK_UP:
            self.pitch += turnVelocity * NormalizedDeltaTime
        elif direction == Camera.Movement.LOOK_DOWN:
            self.pitch -= turnVelocity * NormalizedDeltaTime

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
# NOTE: In this shader we are injecting an uniform 4x4 matrix (mat4)
prog = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec3 ourColor;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // note that we read the multiplication from right to left
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;

  
in vec3 ourColor;
in vec2 TexCoord;

uniform sampler2D ourTexture;
uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);
}
''')

# vertices
vertices = [
    # positions          # texture coords
    -0.5, -0.5, -0.5,  0.0, 0.0,
     0.5, -0.5, -0.5,  1.0, 0.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
    -0.5,  0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 0.0,

    -0.5, -0.5,  0.5,  0.0, 0.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 1.0,
     0.5,  0.5,  0.5,  1.0, 1.0,
    -0.5,  0.5,  0.5,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,

    -0.5,  0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5, -0.5,  1.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,
    -0.5,  0.5,  0.5,  1.0, 0.0,

     0.5,  0.5,  0.5,  1.0, 0.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5,  0.5,  0.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 0.0,

    -0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5, -0.5,  1.0, 1.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,

    -0.5,  0.5, -0.5,  0.0, 1.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5,  0.5,  0.5,  1.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5,  0.5,  0.0, 0.0,
    -0.5,  0.5, -0.5,  0.0, 1.0,
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

# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '32f': This is the format string. It specifies that we want to pack 32 floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
vertices_binaryformat = struct.pack(f"{len(vertices)}f",*vertices)

# Define VBO (Vertex Buffer Object) containing vertex data
vbo = context.buffer(vertices_binaryformat)

# EBO indices for "indexed drawing"
# indexed drawing means identifying vertices in an unique index and build shapes by referring to these indexes
indices = [
    0, 1, 3,
    1, 2, 3
]

# we use struct.pack to convert the indices list into binary format
# 6I stands for 6 unsigned int (https://docs.python.org/3.7/library/struct.html#format-characters)
indices_binaryformat = struct.pack(f"{len(indices)}I",*indices)

# Define EBO containing idices
ebo = context.buffer(indices_binaryformat)

### load image and create texture

# Load image
image1 = pygame.image.load("./assets/container.jpg")
image2 = pygame.image.load("./assets/awesomeface.png")

# Convert image into a stream of bytes with 4 components (RGBA) and flip the image
# (OpenGL expects flipped coordinates) compared to a normal image
image_data1 = pygame.image.tobytes(image1,"RGBA",True)
image_data2 = pygame.image.tobytes(image2,"RGBA",True)

# load the texture within the OpenGL context
texture1 = context.texture(image1.get_size(),4,image_data1)
texture2 = context.texture(image2.get_size(),4,image_data2)

# Access the uniform variable and assign it a texture unit (0) (it will be then used by the texture binding)
prog["texture1"] = 0
prog["texture2"] = 1

# bind the texture to the texture unit related to the uniform variable
texture1.use(location=0)
texture2.use(location=1)




# VBO parameters to be passed to the VAO
# This is what in modernGL is defined as "multiple buffers for all input variables"
# meaning that each VBO buffer is described as a tuple in a list
# elements of the tuple describes
# 1) Vertex Buffer Object in input
# 2) type of input parameters (3f in this case corresponds to a 3vec input) defined in shaders
# 3) name of the input parameter in the related shader (aPos in this case)
vbo_parameters = [
    (vbo,"3f 2f","aPos","aTexCoord")
]

# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
# NOTE: ebo with indices is not used in this example
vao = context.vertex_array(prog,vbo_parameters)




# Function to convert a glm matrix into a GLSL readable stream of bytes to pass as a uniform
def matrix_bytes(matrix):
    ptr = glm.value_ptr(matrix)
    matrix_size = matrix.length() * matrix.length()
    float_array = (ctypes.c_float * matrix_size).from_address(ctypes.addressof(ptr.contents))
    matrix_bytes = bytes(float_array)
    return matrix_bytes


### Camera Object
cam = Camera(glm.vec3(0.0, 0.0, 3.0))


# Matrices Parameters
fov = 45.0
height = 800.0
width = 600.0
x = 0.0
y = 0.0
z = -3.0

# Reference variables for Delta time
FRAMERATE_REFERENCE = 60
FRAMERATE = 60

pygame.display.set_caption("Click on the window to enable mouselook")
while True:
    # calculate the normalized delta time to affect movement consistently regardless FPS
    NormalizedDeltaTime = pygame.time.Clock().tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE

    ## Model Matrices 
    # View Matrix
    # NOTE: now the view matrix will be influenced by the "camera" 
    #       This is all managed now inside the Camera Class
    cam.updateCameraVectors()
    view = cam.GetViewMatrix()
    
    # Projection Matrix
    # NOTE: here we refer to the camera zoom property to influence the projection matrix 
    # (besides the size of the screen and the near and far parameters)
    #
    # GROK3 suggestion:
    # it could make sense to make the projection matrix management part of a Display class—or even a subclass of it—depending on your
    # architecture and how tightly coupled you want the projection settings to be with the display (e.g., window or viewport) properties.
    # This approach aligns with the idea that the projection matrix often depends on display-related factors like the aspect ratio,
    # which is derived from the window’s width and height. Let’s explore both options and their logical implications.
    projection = glm.perspective(glm.radians(cam.zoom),height/ width, 0.1, 100.0)

    # here we pass the values for view and projection matrix uniforms
    prog["view"].write(matrix_bytes(view))
    prog["projection"].write(matrix_bytes(projection))

    for position in cubePositions:
        # Model Matrix
        model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
        model = glm.translate(model,position)
        if cubePositions.index(position) % 3 == 0:
            model = glm.rotate(model,pygame.time.get_ticks() / 1000.0 * glm.radians(50.0),glm.vec3(0.5,1.0,0.0)) # For fun, we'll let the cube rotate over time
        prog["model"].write(matrix_bytes(model))
        vao.render()

    pygame.display.flip()
    context.clear(color=(0.0, 0.0, 0.0), depth=1.0) # clears the framebuffer (Necessary and also best practice) AND clears the z-buffer setting it to the max
    for event in pygame.event.get():
        if  event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN: # when a mouse button is clicked on the window
            if event.button == 1:  # Left mouse button
                # set the mouse invisible and grab the mouse movement (virtual mouse pointer)
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                pygame.display.set_caption("Mouselook enabled - F11 to release")
        elif event.type == pygame.MOUSEWHEEL: # event to capture the mouse wheel
            cam.ProcessMouseScroll(event.y,NormalizedDeltaTime) # event.y is the amount of scroll (up or down)
        elif event.type == pygame.MOUSEMOTION:
            if pygame.event.get_grab():
                relative_x, relative_y = event.rel
                cam.ProcessMouseMovement(relative_x,relative_y,NormalizedDeltaTime)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_F10:
                context.wireframe = not context.wireframe
            elif event.key == pygame.K_F11:
                # release the mouse and keyboard and make the mouse visible
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
                pygame.display.set_caption("Click on the window to enable mouselook")
            elif event.key == pygame.K_F9:
                if depth_test:
                    context.disable(moderngl.DEPTH_TEST)
                    depth_test = False
                else:
                    context.enable(moderngl.DEPTH_TEST)
                    depth_test = True
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