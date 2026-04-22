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


simpleDepthShader = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main()
{
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
''',
fragment_shader='''
#version 330 core

void main()
{             
    // gl_FragDepth = gl_FragCoord.z;
}  
'''
)

debugDepthQuad = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos, 1.0);
}
''',
fragment_shader='''
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D depthMap;
uniform float near_plane;
uniform float far_plane;

// required when using a perspective projection matrix
float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));	
}

void main()
{             
    float depthValue = texture(depthMap, TexCoords).r;
    // FragColor = vec4(vec3(LinearizeDepth(depthValue) / far_plane), 1.0); // perspective
    FragColor = vec4(vec3(depthValue), 1.0); // orthographic
}
'''
)

#### NORMALS AND TEX COORDINATES NEEDED TO BE REMOVED SINCE PADDING WAS NOT WORKING 
#### WHEN DEFINING THE VBO PARAMETERS
planeVertices = [
        #// positions        // normals      // texcoords
         25.0, -0.5,  25.0,  0.0, 1.0, 0.0,  25.0,  0.0,
        -25.0, -0.5,  25.0,  0.0, 1.0, 0.0,   0.0,  0.0,
        -25.0, -0.5, -25.0,  0.0, 1.0, 0.0,   0.0, 25.0,

         25.0, -0.5,  25.0,  0.0, 1.0, 0.0,  25.0,  0.0,
        -25.0, -0.5, -25.0,  0.0, 1.0, 0.0,   0.0, 25.0,
         25.0, -0.5, -25.0,  0.0, 1.0, 0.0,  25.0, 25.0
]

vertices = [
            # back face
            -1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 0.0, # bottom-left
             1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 1.0, # top-right
             1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 0.0, # bottom-right         
             1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 1.0, # top-right
            -1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 0.0, # bottom-left
            -1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 1.0, # top-left
            # front face
            -1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 0.0, # bottom-left
             1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 0.0, # bottom-right
             1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 1.0, # top-right
             1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 1.0, # top-right
            -1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 1.0, # top-left
            -1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 0.0, # bottom-left
            # left face
            -1.0,  1.0,  1.0, -1.0,  0.0,  0.0, 1.0, 0.0, # top-right
            -1.0,  1.0, -1.0, -1.0,  0.0,  0.0, 1.0, 1.0, # top-left
            -1.0, -1.0, -1.0, -1.0,  0.0,  0.0, 0.0, 1.0, # bottom-left
            -1.0, -1.0, -1.0, -1.0,  0.0,  0.0, 0.0, 1.0, # bottom-left
            -1.0, -1.0,  1.0, -1.0,  0.0,  0.0, 0.0, 0.0, # bottom-right
            -1.0,  1.0,  1.0, -1.0,  0.0,  0.0, 1.0, 0.0, # top-right
            # right face
             1.0,  1.0,  1.0,  1.0,  0.0,  0.0, 1.0, 0.0, # top-left
             1.0, -1.0, -1.0,  1.0,  0.0,  0.0, 0.0, 1.0, # bottom-right
             1.0,  1.0, -1.0,  1.0,  0.0,  0.0, 1.0, 1.0, # top-right         
             1.0, -1.0, -1.0,  1.0,  0.0,  0.0, 0.0, 1.0, # bottom-right
             1.0,  1.0,  1.0,  1.0,  0.0,  0.0, 1.0, 0.0, # top-left
             1.0, -1.0,  1.0,  1.0,  0.0,  0.0, 0.0, 0.0, # bottom-left     
            # bottom face
            -1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 0.0, 1.0, # top-right
             1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 1.0, 1.0, # top-left
             1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 1.0, 0.0, # bottom-left
             1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 1.0, 0.0, # bottom-left
            -1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 0.0, 0.0, # bottom-right
            -1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 0.0, 1.0, # top-right
            # top face
            -1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 0.0, 1.0, # top-left
             1.0,  1.0 , 1.0,  0.0,  1.0,  0.0, 1.0, 0.0, # bottom-right
             1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 1.0, 1.0, # top-right     
             1.0,  1.0,  1.0,  0.0,  1.0,  0.0, 1.0, 0.0, # bottom-right
            -1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 0.0, 1.0, # top-left
            -1.0,  1.0,  1.0,  0.0,  1.0,  0.0, 0.0, 0.0  # bottom-left 
        ]

quadVertices = [
            #// positions        // texture Coords
            -1.0,  1.0, 0.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0, 0.0,
             1.0,  1.0, 0.0, 1.0, 1.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
]

# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '32f': This is the format string. It specifies that we want to pack 32 floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
cube_vertices_binaryformat = struct.pack(f"{len(vertices)}f",*vertices)

# Define VBO (Vertex Buffer Object) containing vertex data
cubevbo = context.buffer(cube_vertices_binaryformat)
cubevbo_parameters = [
    # to add proper padding to skip 5 float values the 5x4 format needs to be used
    # more info here https://moderngl.readthedocs.io/en/latest/topics/buffer_format.html#buffer-format
    (cubevbo,"3f 5x4","aPos") 
]

cubevao = context.vertex_array(simpleDepthShader,cubevbo_parameters)



plane_vertices_binaryformat = struct.pack(f"{len(planeVertices)}f",*planeVertices)

# Define VBO (Vertex Buffer Object) containing vertex data
planevbo = context.buffer(plane_vertices_binaryformat)
plabevbo_parameters = [
    (planevbo,"3f 5x4","aPos")
]

planevao = context.vertex_array(simpleDepthShader,plabevbo_parameters)

# Floor Load image
floorImage = pygame.image.load("./assets/wood.png")
# Convert image into a stream of bytes with 4 components (RGBA) and flip the image
# (OpenGL expects flipped coordinates) compared to a normal image
floorImage_data = pygame.image.tobytes(floorImage,"RGBA",True)
# load the texture within the OpenGL context
floorTexture = context.texture(floorImage.get_size(),4,floorImage_data)
floorTexture.use(location=0)


quadVertices_binaryformat = struct.pack(f"{len(quadVertices)}f",*quadVertices)

# Define VBO (Vertex Buffer Object) containing vertex data
quadVerticesvbo = context.buffer(quadVertices_binaryformat)
quadVerticesvbo_parameters = [
    (quadVerticesvbo,"3f 2f","aPos","aTexCoords")
]

quadvao = context.vertex_array(debugDepthQuad,quadVerticesvbo_parameters)


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


pygame.display.set_caption("Click on the window to enable mouselook")

depthMap = context.depth_texture(windowed_size)
framebuffer_object = context.framebuffer(depth_attachment=depthMap)

# generic light position
lightPos = glm.vec3(-2.0, 4.0, -1.0);
while True:

    #################### RENDER TO DEPTH MAP FRAMEBUFFER
    near_plane = 1.0
    far_plane = 7.5
    lightProjection = glm.ortho(-10.0, 10.0, -10.0, 10.0, near_plane, far_plane)
    lightView = glm.lookAt(lightPos, glm.vec3(0.0), glm.vec3(0.0, 1.0, 0.0))
    lightSpaceMatrix = lightProjection * lightView
    simpleDepthShader["lightSpaceMatrix"].write(matrix_bytes(lightSpaceMatrix))
    framebuffer_object.use()
    framebuffer_object.clear(depth=1.0)
    # floor
    model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    simpleDepthShader["model"].write(matrix_bytes(model))
    planevao.render()
    # cubes
    model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    model = glm.translate(model,glm.vec3(0.0, 1.5, 0.0))
    model = glm.scale(model, glm.vec3(0.5))
    simpleDepthShader["model"].write(matrix_bytes(model))
    cubevao.render()
    #
    model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    model = glm.translate(model,glm.vec3(2.0, 0.0, 1.0))
    model = glm.scale(model, glm.vec3(0.5))
    simpleDepthShader["model"].write(matrix_bytes(model))
    cubevao.render()
    #
    model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    model = glm.translate(model,glm.vec3(-1.0, 0.0, 2.0))
    model = glm.rotate(model, glm.radians(60.0), glm.normalize(glm.vec3(1.0, 0.0, 1.0)))
    model = glm.scale(model, glm.vec3(0.25))
    simpleDepthShader["model"].write(matrix_bytes(model))
    cubevao.render()

    #################### RENDER TO DEFAULT SCREEN FRAMEBUFFER
    context.viewport = (0,0,windowed_size[0],windowed_size[1])
    ### FRAMEBUFFER object set in use and clear
    context.screen.use()
    #debugDepthQuad["near_plane"] = near_plane
    #debugDepthQuad["far_plane"] = far_plane
    depthMap.use(location=0)
    debugDepthQuad["depthMap"] = 0
    quadvao.render(mode=moderngl.TRIANGLE_STRIP)

    # calculate the normalized delta time to affect movement consistently regardless FPS
    NormalizedDeltaTime = pygame.time.Clock().tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE







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
            elif event.key == pygame.K_l:
                spotLightOn = not spotLightOn

