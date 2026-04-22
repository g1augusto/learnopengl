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
from enum import Enum 
import os
import math
import ctypes
import struct
import sys
import pygame
import moderngl
import glm


new_path = os.getcwd() + os.pathsep + os.environ.get('PATH', '')
os.environ['PATH'] = new_path

from OpenGL.GL import * 

class Camera():
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
        direction = glm.vec3()
        direction.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        direction.y = math.sin(glm.radians(self.pitch))
        direction.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.cameraTarget = glm.normalize(direction)


    class Movement(Enum):
        FORWARD = 1
        BACKWARD = 2
        TURN_LEFT = 3
        TURN_RIGHT = 4
        STRIFE_LEFT = 5
        STRIFE_RIGHT = 6
        LOOK_UP = 7
        LOOK_DOWN = 8
    
    # --- DIDACTIC GUIDE: MOVEMENT MATH & DELTA TIME ---
    # In frame-independent movement, velocity must be multiplied by deltaTime to ensure
    # the camera moves at the same physical speed regardless of the frame rate.
    # Notice that velocity is calculated ONCE at the top. Previously, the code multiplied
    # by deltaTime a second time inside the specific movement branches (e.g., strafing),
    # squaring the fraction and causing the camera to crawl at high frame rates.
    def ProcessKeyboard(self, direction, deltaTime):
        if not isinstance(direction, Camera.Movement): 
            raise ValueError
        
        velocity = self.MovementSpeed * deltaTime
        turnVelocity = self.TurnSpeed * deltaTime
        
        if direction == Camera.Movement.FORWARD:
            self.cameraPos += self.cameraTarget * velocity
        elif direction == Camera.Movement.BACKWARD:
            self.cameraPos -= self.cameraTarget * velocity
        elif direction == Camera.Movement.TURN_RIGHT:
            self.yaw += turnVelocity
        elif direction == Camera.Movement.TURN_LEFT:
            self.yaw -= turnVelocity
        elif direction == Camera.Movement.STRIFE_RIGHT:
            self.cameraPos += glm.normalize(glm.cross(self.cameraTarget, self.cameraUp)) * velocity
        elif direction == Camera.Movement.STRIFE_LEFT:
            self.cameraPos -= glm.normalize(glm.cross(self.cameraTarget, self.cameraUp)) * velocity
        elif direction == Camera.Movement.LOOK_UP:
            self.pitch += turnVelocity
        elif direction == Camera.Movement.LOOK_DOWN:
            self.pitch -= turnVelocity

    # --- DIDACTIC GUIDE: ABSOLUTE VS. RELATIVE INPUT ---
    # Mouse movement (event.rel) and scroll wheel ticks return ABSOLUTE distances (pixels),
    # not continuous speeds. Therefore, deltaTime must NOT be applied here.
    # Multiplying absolute pixel movement by deltaTime causes mouse sensitivity to
    # plummet at high frame rates and spike wildly at low frame rates.
    def ProcessMouseMovement(self, xoffset, yoffset, constrainPitch=True):
        xoffset *= self.MouseSensitivity 
        yoffset *= self.MouseSensitivity 
        
        self.yaw += xoffset
        self.pitch -= yoffset
        
        if constrainPitch:
            if(self.pitch > 89.0):
                self.pitch =  89.0
            if(self.pitch < -89.0):
                self.pitch = -89.0      

    def ProcessMouseScroll(self, yoffset):
        self.zoom -= yoffset 
        if self.zoom < 1.0:
            self.zoom = 1.0
        if self.zoom > 45.0:
            self.zoom = 45.0

class CameraFPS(Camera):
    def __init__(self, cameraPos=glm.vec3(0, 0, 0), cameraUp=glm.vec3(0, 1, 0), yaw=Camera.YAW, pitch=Camera.PITCH):
        super().__init__(cameraPos, cameraUp, yaw, pitch)
    
    def ProcessKeyboard(self,direction,deltaTime):
        super().ProcessKeyboard(direction=direction,deltaTime=deltaTime)
        self.cameraPos.y = 0.0 

windowed_size = (800,600)
vsync = False

pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE)

screen_flags = pygame.OPENGL | pygame.RESIZABLE | pygame.DOUBLEBUF
screen_display = pygame.display.set_mode(windowed_size,flags=screen_flags,vsync=vsync)

context = moderngl.create_context()
context.enable(moderngl.DEPTH_TEST)
depth_test = True

# --- DIDACTIC GUIDE: LAYERED RENDERING & GEOMETRY SHADERS ---
# Traditional shadow mapping uses a 2D texture. Point shadows require a 3D Cubemap.
# Instead of binding the FBO 6 times and issuing 6 draw calls on the CPU (very slow),
# we use a Geometry Shader. The GS intercepts the triangles from the Vertex Shader,
# loops 6 times, and uses the built-in 'gl_Layer' variable to direct the emitted
# triangles to all 6 faces of the bound Cubemap simultaneously in a single pass.
# The Fragment Shader then calculates the true linear distance to the light.
DepthShader = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
void main()
{
    gl_Position = model * vec4(aPos, 1.0);
}
''',
geometry_shader='''
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

uniform mat4 shadowMatrices[6];
out vec4 FragPos; 

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        gl_Layer = face; 
        for(int i = 0; i < 3; ++i) 
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
} 
''',
fragment_shader='''
#version 330 core
in vec4 FragPos;

uniform vec3 lightPos;
uniform float far_plane;

void main()
{
    float lightDistance = length(FragPos.xyz - lightPos);
    lightDistance = lightDistance / far_plane;
    gl_FragDepth = lightDistance;
} 
'''
)

RenderShader = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform bool reverse_normals;

void main()
{
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
    if(reverse_normals) 
        vs_out.Normal = transpose(inverse(mat3(model))) * (-1.0 * aNormal);
    else
        vs_out.Normal = transpose(inverse(mat3(model))) * aNormal;
        
    // --- DIDACTIC GUIDE: TEXTURE SCALE ILLUSION ---
    // UV coordinates typically map from 0.0 to 1.0. Stretching a small texture (like a few
    // floorboards) across a massive 10x10 3D wall creates an optical illusion, tricking
    // the human brain into perceiving the 3D room as microscopic (a "zoomed in" effect).
    // Multiplying the coordinates by 5.0 forces the texture to tile/repeat 5 times,
    // shrinking the visual floorboards to human scale and correcting the perceived FOV.
    // --- TEXTURE SCALE FIX ---
    vs_out.TexCoords = aTexCoords * 5.0; 
    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
''',
fragment_shader='''
#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
} fs_in;

uniform sampler2D diffuseTexture;
uniform samplerCube depthMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform float far_plane;
uniform bool shadows;

vec3 gridSamplingDisk[20] = vec3[]
(
   vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1), 
   vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
   vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
   vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
   vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);

float ShadowCalculation(vec3 fragPos)
{
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight);
    
    // --- DIDACTIC GUIDE: PETER PANNING & LIGHTING ---
    // Shadow Bias: A high bias (e.g., 0.15) pushes the shadow map too far backward,
    // causing the shadow to detach from the object (an artifact known as Peter Panning).
    float shadow = 0.0;
    float bias = 0.05; // --- BIAS FIX ---
    int samples = 20;
    float viewDistance = length(viewPos - fragPos);
    float diskRadius = (1.0 + (viewDistance / far_plane)) / 25.0;
    
    for(int i = 0; i < samples; ++i)
    {
        float closestDepth = texture(depthMap, fragToLight + gridSamplingDisk[i] * diskRadius).r;
        closestDepth *= far_plane;   
        if(currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);
        
    return shadow;
}

void main()
{           
    vec3 color = texture(diffuseTexture, fs_in.TexCoords).rgb;
    vec3 normal = normalize(fs_in.Normal);
    
    // --- DIDACTIC GUIDE: AMBIENT LIGHT LEVELS ---
    // A low base light color (e.g., 0.3) multiplied by an ambient factor (0.3)
    // results in near pitch-black shadows (0.09). Pure black environments hide the
    // soft Percentage-Closer Filtering (PCF) blurs, making shadows look artificially sharp.
    // Setting lightColor to 1.0 properly illuminates the gray penumbra.
    // --- LIGHT COLOR FIX ---
    vec3 lightColor = vec3(1.0); 
    
    // ambient
    vec3 ambient = 0.3 * lightColor;
    // diffuse
    vec3 lightDir = normalize(lightPos - fs_in.FragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    // specular
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.0;
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = spec * lightColor;    
    
    // calculate shadow
    float shadow = shadows ? ShadowCalculation(fs_in.FragPos) : 0.0;                      
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;    
    
    FragColor = vec4(lighting, 1.0);
}
'''
)

vertices = [
            # back face
            -1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 0.0, 
             1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 1.0, 
             1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 0.0,          
             1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 1.0, 
            -1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 0.0, 
            -1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 1.0, 
            # front face
            -1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 0.0, 
             1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 0.0, 
             1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 1.0, 
             1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 1.0, 
            -1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 1.0, 
            -1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 0.0, 
            # left face
            -1.0,  1.0,  1.0, -1.0,  0.0,  0.0, 1.0, 0.0, 
            -1.0,  1.0, -1.0, -1.0,  0.0,  0.0, 1.0, 1.0, 
            -1.0, -1.0, -1.0, -1.0,  0.0,  0.0, 0.0, 1.0, 
            -1.0, -1.0, -1.0, -1.0,  0.0,  0.0, 0.0, 1.0, 
            -1.0, -1.0,  1.0, -1.0,  0.0,  0.0, 0.0, 0.0, 
            -1.0,  1.0,  1.0, -1.0,  0.0,  0.0, 1.0, 0.0, 
            # right face
             1.0,  1.0,  1.0,  1.0,  0.0,  0.0, 1.0, 0.0, 
             1.0, -1.0, -1.0,  1.0,  0.0,  0.0, 0.0, 1.0, 
             1.0,  1.0, -1.0,  1.0,  0.0,  0.0, 1.0, 1.0,          
             1.0, -1.0, -1.0,  1.0,  0.0,  0.0, 0.0, 1.0, 
             1.0,  1.0,  1.0,  1.0,  0.0,  0.0, 1.0, 0.0, 
             1.0, -1.0,  1.0,  1.0,  0.0,  0.0, 0.0, 0.0,     
            # bottom face
            -1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 0.0, 1.0, 
             1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 1.0, 1.0, 
             1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 1.0, 0.0, 
             1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 1.0, 0.0, 
            -1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 0.0, 0.0, 
            -1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 0.0, 1.0, 
            # top face
            -1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 0.0, 1.0, 
             1.0,  1.0 , 1.0,  0.0,  1.0,  0.0, 1.0, 0.0, 
             1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 1.0, 1.0,     
             1.0,  1.0,  1.0,  0.0,  1.0,  0.0, 1.0, 0.0, 
            -1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 0.0, 1.0, 
            -1.0,  1.0,  1.0,  0.0,  1.0,  0.0, 0.0, 0.0  
        ]

cube_vertices_binaryformat = struct.pack(f"{len(vertices)}f",*vertices)

cubevbo = context.buffer(cube_vertices_binaryformat)
cubevbo_parameters_depth = [
    (cubevbo,"3f 5x4","aPos")
]
cubevbo_parameters = [
    (cubevbo,"3f 3f 2f","aPos","aNormal","aTexCoords")
]

cubevao_depth = context.vertex_array(DepthShader,cubevbo_parameters_depth)
cubevao = context.vertex_array(RenderShader,cubevbo_parameters)

try:
    floorImage = pygame.image.load("./assets/wood.png")
    floorImage_data = pygame.image.tobytes(floorImage,"RGBA",True)
    floorTexture = context.texture(floorImage.get_size(),4,floorImage_data)
    floorTexture.use(location=0)
except FileNotFoundError:
    print("Warning: Texture not found, creating dummy texture.")
    floorImage_data = b'\xff\x00\xff\xff' * 16
    floorTexture = context.texture((4, 4), 4, floorImage_data)
    floorTexture.use(location=0)


def matrix_bytes(matrix:glm.mat4):
    ptr = glm.value_ptr(matrix)
    matrix_size = matrix.length() * matrix.length()
    float_array = (ctypes.c_float * matrix_size).from_address(ctypes.addressof(ptr.contents))
    matrix_bytes_output = bytes(float_array)
    return matrix_bytes_output

### Camera Object
cam = Camera(glm.vec3(0.0, 0.0, 3.0))
FRAMERATE_REFERENCE = 60

pygame.display.set_caption("Click on the window to enable mouselook")

cubemap_size = (1024,1024)

# --- DIDACTIC GUIDE: THE HARDWARE DEPTH COMPARISON BUG ---
# By default, ModernGL enables Hardware Depth Comparison on depth textures (compare_func).
# This assumes you want the GPU to automatically return a boolean (1.0 or 0.0) for shadows.
# However, our GLSL shader uses 'samplerCube' to manually read the raw float distances.
# Mixing manual GLSL sampling with hardware depth comparison causes undefined behavior,
# resulting in pitch-black, jagged, curved artifacts clipping through the walls.
# Setting compare_func to an empty string disables the hardware intervention, 
# allowing the shader to receive the raw float values it expects.
# 1. Create the final destination TextureCube
depthCubemapTexture = context.depth_texture_cube(cubemap_size)
# --- HARDWARE COMPARISON FIX ---
depthCubemapTexture.compare_func = ''
depthCubemapTexture.filter = (moderngl.LINEAR, moderngl.LINEAR)
depthCubemapTexture.repeat_x = False
depthCubemapTexture.repeat_y = False
depthCubemapTexture.repeat_z = False

# --- DIDACTIC GUIDE: MODERNGL LIMITATION & PYOPENGL BRIDGE ---
# ModernGL acts as a safe, object-oriented wrapper around raw OpenGL. However, its
# 'context.framebuffer()' strictly enforces that depth attachments must be 2D textures.
# It natively blocks attaching a 6-sided TextureCube for Geometry Shader layered rendering.
# To bypass this, we use a hybrid approach:
# 1. We create a "dummy" 2D texture to satisfy ModernGL's strict FBO creation check.
# 2. We expose the raw OpenGL integer ID of the ModernGL objects using the '.glo' property.
# 3. We use raw PyOpenGL (glBindFramebuffer, glFramebufferTexture) to hot-swap the 
#    dummy texture out and attach the entire Cubemap under the hood.
# 4. We use glDrawBuffer/glReadBuffer(GL_NONE) because shadow maps don't need color data.
# 2. Create a "dummy" 2D texture to satisfy ModernGL's FBO type checks
dummy_depth = context.depth_texture(cubemap_size)
framebuffer_object = context.framebuffer(depth_attachment=dummy_depth)

# 3. Use raw PyOpenGL to swap the dummy texture for the full TextureCube array
glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_object.glo)
glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthCubemapTexture.glo, 0)
glDrawBuffer(GL_NONE) 
glReadBuffer(GL_NONE)
if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    print("Error: Shadow Framebuffer is not complete!")
glBindFramebuffer(GL_FRAMEBUFFER, 0)

lightPos = glm.vec3(0.0, 0.0, 0.0)
clock = pygame.time.Clock()


# --- DIDACTIC GUIDE: DRY PRINCIPLE (DON'T REPEAT YOURSELF) ---
# In a multipass rendering pipeline (e.g., drawing once for the depth map, and
# again for the screen), duplicating the model matrix transformations is dangerous.
# If an object moves in the visual pass but you forget to update the depth pass,
# the shadow will physically detach from the object. Centralizing the draw calls into a
# single function ensures the geometry is 100% synchronized across all passes.
# --- DRY HELPER FUNCTION ---
def renderScene(shader_program, vao):
    # room cube
    model = glm.mat4(1.0)
    model = glm.scale(model, glm.vec3(5.0))
    shader_program["model"].write(matrix_bytes(model))
    
    context.disable(moderngl.CULL_FACE)
    if "reverse_normals" in shader_program:
        shader_program["reverse_normals"].value = True
    vao.render()
    if "reverse_normals" in shader_program:
        shader_program["reverse_normals"].value = False
    context.enable(moderngl.CULL_FACE)
    
    # cubes
    model = glm.mat4(1.0)
    model = glm.translate(model,glm.vec3(4.0, -3.5, 0.0))
    model = glm.scale(model, glm.vec3(0.5))
    shader_program["model"].write(matrix_bytes(model))
    vao.render()
    
    model = glm.mat4(1.0)
    model = glm.translate(model,glm.vec3(2.0, 3.0, 1.0))
    model = glm.scale(model, glm.vec3(0.75))
    shader_program["model"].write(matrix_bytes(model))
    vao.render()
    
    model = glm.mat4(1.0)
    model = glm.translate(model,glm.vec3(-3.0, -1.0, 0.0))
    model = glm.scale(model, glm.vec3(0.5))
    shader_program["model"].write(matrix_bytes(model))
    vao.render()
    
    model = glm.mat4(1.0)
    model = glm.translate(model,glm.vec3(-1.5, 1.0, 1.5))
    model = glm.scale(model, glm.vec3(0.5))
    shader_program["model"].write(matrix_bytes(model))
    vao.render()
    
    model = glm.mat4(1.0)
    model = glm.translate(model,glm.vec3(-1.5, 2.0, -3.0))
    model = glm.rotate(model, glm.radians(60.0), glm.normalize(glm.vec3(1.0, 0.0, 1.0)))
    model = glm.scale(model, glm.vec3(0.75))
    shader_program["model"].write(matrix_bytes(model))
    vao.render()


while True:
    
    # --- ANIMATED LIGHT FIX ---
    time_seconds = pygame.time.get_ticks() / 1000.0
    lightPos.x = 0.0
    lightPos.y = 0.0
    lightPos.z = math.sin(time_seconds * 0.5) * 3.0
    
    #################### RENDER TO DEPTH MAP FRAMEBUFFER
    context.viewport = (0,0,cubemap_size[0],cubemap_size[1])
    aspect = cubemap_size[0] / cubemap_size[1]
    near_plane = 1.0
    far_plane = 25.0
    
    lightProjection = glm.perspective(glm.radians(90.0), aspect, near_plane, far_plane)
    
    shadowTransforms = [
        lightProjection  * glm.lookAt(lightPos, lightPos + glm.vec3( 1.0, 0.0, 0.0), glm.vec3(0.0,-1.0, 0.0)),
        lightProjection  * glm.lookAt(lightPos, lightPos + glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0.0,-1.0, 0.0)),
        lightProjection  * glm.lookAt(lightPos, lightPos + glm.vec3( 0.0, 1.0, 0.0), glm.vec3(0.0, 0.0, 1.0)),
        lightProjection  * glm.lookAt(lightPos, lightPos + glm.vec3( 0.0,-1.0, 0.0), glm.vec3(0.0, 0.0,-1.0)),
        lightProjection  * glm.lookAt(lightPos, lightPos + glm.vec3( 0.0, 0.0, 1.0), glm.vec3(0.0,-1.0, 0.0)),
        lightProjection  * glm.lookAt(lightPos, lightPos + glm.vec3( 0.0, 0.0,-1.0), glm.vec3(0.0,-1.0, 0.0)),
    ]
    
    packed_data = b''.join(matrix_bytes(m) for m in shadowTransforms)
    
    framebuffer_object.use()
    framebuffer_object.clear(depth=1.0) 
    
    DepthShader["shadowMatrices"].write(packed_data)
    DepthShader["far_plane"].value = far_plane
    DepthShader["lightPos"].value = tuple(lightPos)
    
    # Render scene for depth map
    renderScene(DepthShader, cubevao_depth)


    # --- DIDACTIC GUIDE: HIGH-DPI VIEWPORT SCALING ---
    # Hardcoding the viewport to the logical window size (e.g., 800x600) breaks on
    # High-DPI monitors (Retina, 4K, or OS-level UI scaling), where the physical pixel
    # buffer might be 1600x1200. This squashes the scene into the corner of the window.
    # 'context.screen.viewport' automatically queries the OS for the true physical
    # dimensions, ensuring the OpenGL canvas maps 1:1 with the window pixels.
    #################### RENDER TO DEFAULT SCREEN FRAMEBUFFER
    # --- VIEWPORT FIX ---
    context.viewport = context.screen.viewport
    context.screen.use()
    context.clear(color=(0.1, 0.1, 0.1), depth=1.0) # Made slightly brighter like C++ base color
    
    cam.updateCameraVectors()
    view = cam.GetViewMatrix()
    projection = glm.perspective(glm.radians(cam.zoom),windowed_size[0] / windowed_size[1], 0.1, 100.0)   
    
    RenderShader["projection"].write(matrix_bytes(projection))
    RenderShader["view"].write(matrix_bytes(view))
    RenderShader["viewPos"].value = tuple(cam.cameraPos)
    RenderShader["lightPos"].value = tuple(lightPos)
    RenderShader["shadows"].value = True
    RenderShader["far_plane"].value = far_plane
    
    depthCubemapTexture.use(location=0)
    floorTexture.use(location=1)
    RenderShader["depthMap"] = 0
    RenderShader["diffuseTexture"] = 1
    
    # Render scene for screen
    renderScene(RenderShader, cubevao)
    
    NormalizedDeltaTime = clock.tick(0) * 0.001 * FRAMERATE_REFERENCE
    pygame.display.flip()

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
        elif event.type == pygame.MOUSEBUTTONDOWN: 
            if event.button == 1: 
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                pygame.display.set_caption("Mouselook enabled - F11 to release")
                pygame.mouse.get_rel() 
        elif event.type == pygame.MOUSEWHEEL: 
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                pass 
            elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                pass
            elif keys[pygame.K_SPACE]:
                pass
            else:
                cam.ProcessMouseScroll(event.y) 
        elif event.type == pygame.MOUSEMOTION:
            if pygame.event.get_grab():
                relative_x, relative_y = event.rel
                cam.ProcessMouseMovement(relative_x, relative_y) 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
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
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
                pygame.display.set_caption("Click on the window to enable mouselook")