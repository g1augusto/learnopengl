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

Excercise:  Diffuse lighting
            Adding ambient lighting to the scene is really easy. We take the light's color,
            multiply it with a small constant ambient factor, multiply this with the object's color,
            and use that as the fragment's color in the cube object's shader
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
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 Normal;  // normal vector used to calculate the angle of diffuse light per fragment
out vec3 FragPos; // position of the fragment in the world space (retrieved by multiplying the vertex position with model matrix)

void main()
{
    // note that we read the multiplication from right to left
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    Normal = aNormal;
    FragPos = vec3(model * vec4(aPos,1.0)); // Fragment position in the world calculated by the model matrix
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;

in vec3 Normal;  // normal vector used to calculate the angle of diffuse light per fragment
in vec3 FragPos; // This in variable will be interpolated from the 3 world position vectors of the triangle to form the FragPos vector that is the per-fragment world position

uniform vec3 objectColor; // used by ambient lighting
uniform vec3 lightColor;  // used by ambient lighting
uniform vec3 lightPos; // position of the light source for diffuse lighting

void main()
{   
    // Ambient Lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse Lighting
    // When calculating lighting we usually do not care about the magnitude of a vector or their position, we only care about their direction
    // calculations are done with unit vectors since it simplifies most calculations
    vec3 normalized = normalize(Normal);
    vec3 lightDirection = normalize(lightPos - FragPos);
    // calculate the diffuse impact of the light on the current fragment by taking the dot product between the norm and lightDir vectors
    // The resulting value is then multiplied with the light's color to get the diffuse component
    float diffuseImpact = max(dot(normalized,lightDirection),0.0);
    vec3 diffuseComponent = diffuseImpact * lightColor;
    
    // Calculate the result by merging ambient and diffuse lights and multiply by the object color
    vec3 result = (ambient + diffuseComponent) * objectColor;
    FragColor = vec4(result, 1.0);
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
    # position         # normal vector for each vertex (precalculated)
    -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
     0.5, -0.5, -0.5,  0.0,  0.0, -1.0, 
     0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
     0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
    -0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
    -0.5, -0.5, -0.5,  0.0,  0.0, -1.0, 

    -0.5, -0.5,  0.5,  0.0,  0.0, 1.0,
     0.5, -0.5,  0.5,  0.0,  0.0, 1.0,
     0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
     0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
    -0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0,  0.0, 1.0,

    -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
    -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
    -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,

     0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
     0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
     0.5,  0.5,  0.5,  1.0,  0.0,  0.0,

    -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
     0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
    -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
    -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,

    -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
     0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
     0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
     0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
    -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
    -0.5,  0.5, -0.5,  0.0,  1.0,  0.0
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





# VBO parameters to be passed to the VAO
# This is what in modernGL is defined as "multiple buffers for all input variables"
# meaning that each VBO buffer is described as a tuple in a list
# elements of the tuple describes
# 1) Vertex Buffer Object in input
# 2) type of input parameters (3f in this case corresponds to a 3vec input) defined in shaders
# 3) name of the input parameter in the related shader (aPos in this case)
# NOTE: These parameters are the same also for the light source VAO
vbo_parameters = [
    (vbo,"3f 3f","aPos","aNormal")
]

# now for the lightbox

vboLightbox_parameters = [
    (vboLightbox,"3f","aPos")
]


# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
# NOTE: ebo with indices is not used in this example
vao = context.vertex_array(prog,vbo_parameters)

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
cam = CameraFPS(glm.vec3(0.0, 0.0, 3.0))


# Matrices Parameters
height = 800.0
width = 600.0

# Reference variables for Delta time
FRAMERATE_REFERENCE = 60
FRAMERATE = 60

# this is the position of the light source in the scene and will be used on the light model matrix
lightPos = glm.vec3(1.2, 1.0, 2.0)

# Here we update the model matrix for both VAOs
# NOTE: in this example these objects are STATIC, meaning that there is no reason to update their
#       model at the rendering loop but if these objects were to move or rotate then yes they would
#       require a loop iteration update
model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
prog["model"].write(matrix_bytes(model))
modelLight = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
modelLight = glm.translate(modelLight,lightPos) # put the light source in position
modelLight = glm.scale(modelLight, glm.vec3(0.2))
progLight["model"].write(matrix_bytes(modelLight))


# Access the uniform variable 
# NOTE: These values are assigned only to the objects shader (prog)
prog["objectColor"].value = glm.vec3(1.0, 0.5, 0.31)
prog["lightColor"].value = glm.vec3(1.0, 1.0, 1.0)
# Now pass the value of the light position vector as uniform
prog["lightPos"].value = lightPos

pygame.display.set_caption("Click on the window to enable mouselook")
while True:
    # calculate the normalized delta time to affect movement consistently regardless FPS
    NormalizedDeltaTime = pygame.time.Clock().tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE

    ## Matrices 
    # NOTE: View and Projecttion matrices needs to be updated at every loop iteration
    # View Matrix
    # NOTE: now the view matrix will be influenced by the "camera" 
    #       This is all managed now inside the Camera Class
    cam.updateCameraVectors()
    view = cam.GetViewMatrix()
    
    # Projection Matrix
    # NOTE: here we refer to the camera zoom property to influence the projection matrix 
    #       
    projection = glm.perspective(glm.radians(cam.zoom),height/ width, 0.1, 100.0)

    # here we pass the values for view and projection matrix uniforms
    # NOTE: we do it for both shader programs
    prog["view"].write(matrix_bytes(view))
    prog["projection"].write(matrix_bytes(projection))
    progLight["view"].write(matrix_bytes(view))
    progLight["projection"].write(matrix_bytes(projection))

    # NOTE: Model Matrices are defined out of the loop becuse objects are static in this example

        
    # here we render both VAOs
    vao.render()
    lightvao.render()

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