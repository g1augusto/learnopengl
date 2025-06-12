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

Excercise:  Directional light
            When a light source is far away the light rays coming from the light source are close to parallel to
            each other. It looks like all the light rays are coming from the same direction, regardless of where
            the object and/or the viewer is. When a light source is modeled to be infinitely far away it is called
            a directional light since all its light rays have the same direction; it is independent of the location
            of the light source.
            We can model such a directional light by defining a light direction vector instead of a position vector.
            The shader calculations remain mostly the same except this time we directly use the light's direction
            vector instead of calculating the lightDir vector using the light's position vector

            NOTE: Light movement and rendering code (VAO,VBO,etc) was removed because not relevant for this sample
            Excercise:  
                        
                        Controls
                        ----------------------------------------
                        SHIFT + mouse scroll: move light on the Y Axis
                        F2: start/stop spinning light
                        F3: increase source light ambient 
                        F4: decrease source light ambient
                        F5: increase source light diffuse
                        F6: decrease source light diffuse
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
// Removed ambient light (will be calculated from the diffuse light)
// added a texture sampler source to calculate the light color on the surface
struct Material {
    sampler2D diffuse;  // diffuse is a diffuse map texture
    sampler2D specular; // specular is now a specular map texture
    float     shininess;
}; 

struct Light {  // Light source data type
    vec3 direction; // this is a directional light not bound to a position (like the sun)
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};  

out vec4 FragColor;

in vec3 Normal;  // normal vector used to calculate the angle of diffuse light per fragment
in vec3 FragPos; // This in variable will be interpolated from the 3 world position vectors of the triangle to form the FragPos vector that is the per-fragment world position
in vec2 TexCoords; // Texture coordinate for the diffuse sampler received from the vertex shader

uniform vec3 objectColor; // used by ambient lighting
uniform vec3 viewPos; // position of the viewer (camera)

uniform Material material; // contain data for the material surface
uniform Light light; // contain data for position and light intensity for ambient/diffuse/specular

void main()
{   
    // Ambient Lighting
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords)); // using diffuse material color from the texture sampler

    // Diffuse Lighting
    // When calculating lighting we usually do not care about the magnitude of a vector or their position, we only care about their direction
    // calculations are done with unit vectors since it simplifies most calculations
    vec3 normalized = normalize(Normal);
    // first negate the light.direction vector. The lighting calculations we used so far expect the light direction to be a
    // direction from the fragment towards the light source, but people generally prefer to specify a directional light as
    // a global direction pointing from the light source
    vec3 lightDirection = normalize(-light.direction);
    // calculate the diffuse impact of the light on the current fragment by taking the dot product between the norm and lightDir vectors
    // The resulting value is then multiplied with the light's color to get the diffuse component
    float diffuseImpact = max(dot(normalized,lightDirection),0.0);
    vec3 diffuseComponent = light.diffuse * diffuseImpact * vec3(texture(material.diffuse, TexCoords)); // we are using a diffuse map to sample the color
    
    // Specular lighting
    vec3 viewDirection = normalize(viewPos - FragPos);
    vec3 reflectDirection = reflect(-lightDirection, normalized); 
    float spec = pow(max(dot(viewDirection, reflectDirection), 0.0), material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular,TexCoords)); // we are using a specular map now instead of a single vector3


    // Calculate the result by merging ambient and diffuse lights (object color is removed, color is provided by the material properties)
    vec3 result = ambient + diffuseComponent + specular;
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

### load image and create texture

# Load image
diffuseMapimage = pygame.image.load("./assets/container2.png")
specularMapimage = pygame.image.load("./assets/container2_specular.png")

# Convert image into a stream of bytes with 4 components (RGBA) and flip the image
# (OpenGL expects flipped coordinates) compared to a normal image
diffuseMapimage_data = pygame.image.tobytes(diffuseMapimage,"RGBA",True)
specularMapimage_data = pygame.image.tobytes(specularMapimage,"RGBA",True)

# load the texture within the OpenGL context
diffuseMapTexture = context.texture(diffuseMapimage.get_size(),4,diffuseMapimage_data)
specularMapTexture = context.texture(specularMapimage.get_size(),4,specularMapimage_data)

# Access the uniform variable and assign it a texture unit (0) (it will be then used by the texture binding)
prog["material.diffuse"] = 0
prog["material.specular"] = 1

# bind the texture to the texture unit related to the uniform variable
diffuseMapTexture.use(location=0)
specularMapTexture.use(location=1)


# VBO parameters to be passed to the VAO
# This is what in modernGL is defined as "multiple buffers for all input variables"
# meaning that each VBO buffer is described as a tuple in a list
# elements of the tuple describes
# 1) Vertex Buffer Object in input
# 2) type of input parameters (3f in this case corresponds to a 3vec input) defined in shaders
# 3) name of the input parameter in the related shader (aPos in this case)
# NOTE: These parameters are the same also for the light source VAO
vbo_parameters = [
    (vbo,"3f 3f 2f","aPos","aNormal","aTexCoords")
]

# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
# NOTE: ebo with indices is not used in this example
vao = context.vertex_array(prog,vbo_parameters)

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

# In this scenario we have directional light with a specific direction applied to all objects (like the sun)
# the coordinates shows that the light is pointing downwards
prog["light.direction"] = glm.vec3(-0.2, -1.0, -0.3)

# Material parameters
prog["material.shininess"] = 32.0

# Light source parameters
ambientStrength = 0.2
diffuseStrength = 0.5
specularStrength = 1.0

# Start with a moving light
moveLight = True
pygame.display.set_caption("Click on the window to enable mouselook")
while True:
    # calculate the normalized delta time to affect movement consistently regardless FPS
    NormalizedDeltaTime = pygame.time.Clock().tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE
    
    # Pass light source parameters to the shaders
    prog["light.ambient"] = glm.vec3(ambientStrength, ambientStrength, ambientStrength)
    prog["light.diffuse"] = glm.vec3(diffuseStrength, diffuseStrength, diffuseStrength)
    prog["light.specular"] = glm.vec3(specularStrength, specularStrength, specularStrength)

    ## Matrices 
    # NOTE: View and Projection matrices needs to be updated at every loop iteration
    # View Matrix
    # NOTE: now the view matrix will be influenced by the "camera" 
    #       This is all managed now inside the Camera Class
    cam.updateCameraVectors()
    view = cam.GetViewMatrix()

    # Projection Matrix
    # NOTE: here we refer to the camera zoom property to influence the projection matrix 
    #       
    projection = glm.perspective(glm.radians(cam.zoom),windowed_size[0] / windowed_size[1], 0.1, 100.0)    

    
    # here we pass the values for view and projection matrix uniforms
    prog["view"].write(matrix_bytes(view))
    prog["projection"].write(matrix_bytes(projection))


    # load the viewPos uniform with the position of the camera
    prog["viewPos"].value = cam.cameraPos
    # Here we update the model matrix for both VAOs
    # NOTE: since we have multiple cubes to draw we run the render() method of the associated VAO
    #       at each loop
    for position in cubePositions:
        # Model Matrix
        model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
        model = glm.translate(model,position)
        if cubePositions.index(position) % 3 == 0:
            model = glm.rotate(model,pygame.time.get_ticks() / 1000.0 * glm.radians(50.0),glm.vec3(0.5,1.0,0.0)) # For fun, we'll let the cube rotate over time
        prog["model"].write(matrix_bytes(model))
        ## -> calculate the Normal Matrix
        # NOTE: Normal matrix needs to be calculated before rendering the models for EACH model
        #       the normal matrix calculation and update before rendering each cube.
        #       This ensures the correct normal transformation is applied to the current cube’s model matrix.
        normalMatrix = glm.mat3(glm.transpose(glm.inverse(model)))
        # load the normal matrix calculated before
        # NOTE: This way we calculate one normal matrix for each object rather than for each vertex as it is in learnopengl
        prog["normalMatrix"].write(matrix_bytes(normalMatrix))
        vao.render()







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
                #lightPos.y += (event.y / 100)
                ...
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
                # increase ambient light
                if ambientStrength<1.0: ambientStrength += 0.1
                print(f"ambient light {ambientStrength}")
            elif event.key == pygame.K_F4:
                # decrease ambient light
                if ambientStrength>0.0: ambientStrength -= 0.1
                print(f"ambient light {ambientStrength}")
            elif event.key == pygame.K_F5:
                # increase diffuse light
                if diffuseStrength<1.0: diffuseStrength += 0.1
                print(f"diffuse light: {diffuseStrength}")
            elif event.key == pygame.K_F6:
                # decrease diffuse light
                if diffuseStrength>0.0: diffuseStrength -= 0.1
                print(f"diffuse light: {diffuseStrength}")
            elif event.key == pygame.K_F7:
                # increase specular light
                if specularStrength<1.0: specularStrength += 0.1
                print(f"specular light: {specularStrength}")
            elif event.key == pygame.K_F8:
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
            elif event.key == pygame.K_F9:
                if depth_test:
                    context.disable(moderngl.DEPTH_TEST)
                    depth_test = False
                else:
                    context.enable(moderngl.DEPTH_TEST)
                    depth_test = True
