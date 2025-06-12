import pygame
import moderngl
import glm
from enum import Enum # to define movement enum class
import sys
import struct
import ctypes
from functools import singledispatch # for function overloading
from dataclasses import dataclass
import time
import random

# GLOBAL CONSTANTS
FRAMERATE = 60                          # Framerate to use in clock tick method
FRAMERATE_REFERENCE = 60                # Reference from Framerate Target to be achieved
WIDTH = 800
HEIGHT = 600
VSYNC = False
PLAYER_SIZE = glm.vec2(100.0,20)
PLAYER_VELOCITY = 10.0
INITIAL_BALL_VELOCITY = glm.vec2(5.0,-5.0)
BALL_RADIUS = 12.5



############### SHADER DEFINITION
# main program shaders
program_vs='''
// Note that we store both the position and texture-coordinate data in a single vec4 variable.
//Because both the position and texture coordinates contain two floats, we can combine them in a single vertex attribute.
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 projection;

void main()
{
    TexCoords = vertex.zw;
    gl_Position = projection * model * vec4(vertex.xy, 0.0, 1.0);
}
'''

program_fs='''
// The fragment shader is relatively straightforward as well.
// We take a texture and a color vector that both affect the final color of the fragment.
// By having a uniform color vector, we can easily change the color of sprites from the game-code
// NOTE: Added a discard option for alpha transparency (color.a is the alpha channel)
// TODO: Include several level of transparency for fancy bricks
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D image;
uniform vec3 spriteColor;

void main()
{    
    vec4 color = vec4(spriteColor, 1.0) * texture(image, TexCoords);
    if (color.a < 0.1)
       discard;
    FragColor = color;
}  
'''

# particles shaders
particles_vs='''
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>

out vec2 TexCoords;
out vec4 ParticleColor;

uniform mat4 projection;
uniform vec2 offset;
uniform vec4 color;

void main()
{
    float scale = 10.0f;
    TexCoords = vertex.zw;
    ParticleColor = color;
    gl_Position = projection * vec4((vertex.xy * scale) + offset, 0.0, 1.0);
}
'''

particles_fs='''
#version 330 core
in vec2 TexCoords;
in vec4 ParticleColor;
out vec4 color;

uniform sampler2D sprite;

void main()
{
    color = (texture(sprite, TexCoords) * ParticleColor);
}
'''

# postprocessing shaders
postProcessing_vs='''
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>

out vec2 TexCoords;

uniform bool  chaos;
uniform bool  confuse;
uniform bool  shake;
uniform float time;

void main()
{
    gl_Position = vec4(vertex.xy, 0.0f, 1.0f); 
    vec2 texture = vertex.zw;
    if (chaos)
    {
        float strength = 0.3;
        vec2 pos = vec2(texture.x + sin(time) * strength, texture.y + cos(time) * strength);        
        TexCoords = pos;
    }
    else if (confuse)
    {
        TexCoords = vec2(1.0 - texture.x, 1.0 - texture.y);
    }
    else
    {
        TexCoords = texture;
    }
    if (shake)
    {
        float strength = 0.01;
        gl_Position.x += cos(time * 10) * strength;        
        gl_Position.y += cos(time * 15) * strength;        
    }
}
'''

postProcessing_fs='''
#version 330 core
in  vec2  TexCoords;
out vec4  color;
  
uniform sampler2D scene;
uniform vec2      offsets[9];
uniform int       edge_kernel[9];
uniform float     blur_kernel[9];

uniform bool chaos;
uniform bool confuse;
uniform bool shake;

void main()
{
    color = vec4(0.0f);
    vec3 sample[9];
    // sample from texture offsets if using convolution matrix
    if(chaos || shake)
        for(int i = 0; i < 9; i++)
            sample[i] = vec3(texture(scene, TexCoords.st + offsets[i]));

    // process effects
    if (chaos)
    {           
        for(int i = 0; i < 9; i++)
            color += vec4(sample[i] * edge_kernel[i], 0.0f);
        color.a = 1.0f;
    }
    else if (confuse)
    {
        color = vec4(1.0 - texture(scene, TexCoords).rgb, 1.0);
    }
    else if (shake)
    {
        for(int i = 0; i < 9; i++)
            color += vec4(sample[i] * blur_kernel[i], 0.0f);
        color.a = 1.0f;
    }
    else
    {
        color =  texture(scene, TexCoords);
    }
}
'''
#################################

# Particle generation
@dataclass
class Particle:
    Position:glm.vec2 = glm.vec2(0.0)
    Velocity:glm.vec2 = glm.vec2(0.0)
    Color:glm.vec4 = glm.vec4(1.0)
    Life:float = 0.0

# LearnOpenGL uses a resource manager c++ static class (a class that needs no instance and is globally accessible)
# here we don't need all of that since python and libraries offers good abstractions
# but we use a class for loading and storing moderngl.Texture objects loaded and converted from image files
# with modernGL would not make sense to use a library or a class with static methods since Texture objects are
# tied to a specific moderngl.Context which is instantiated within the Game class
# This class is part of the Game class and a TextureManager is instantiated at the game init
class Textures():
    '''
    Store a texture object as a dictionary with the key being an arbitrary name and value the file path of the image\n
    Retrieve a texture object by accessing it via the aforementioned key 
    '''
    def __init__(self,context:moderngl.Context):
        self._textures = {}
        self.context = context
    
    def __contains__(self,key):
        return key in self._textures
    
    def __getitem__(self,key):
        try:
            return self._textures[key]
        except KeyError:
            # Re-raise the KeyError with a potentially more helpful message
            raise KeyError(f"No data found for key '{key}'. Available keys: {list(self._textures.keys())}")
    
    def __setitem__(self,key,value):
        '''
        Assign to an arbitrary key value the Texture object in a context by\n
        providing in input the filename and path
        '''
        image = pygame.image.load(value)
        image_data = pygame.image.tobytes(image,"RGBA",False)
        self._textures[key] =  self.context.texture(image.get_size(),4,image_data)




def matrix_bytes(matrix:glm.mat4):
    '''
    Function to convert a glm matrix into a GLSL readable stream of bytes to pass as a uniform
    '''
    ptr = glm.value_ptr(matrix)
    matrix_size = matrix.length() * matrix.length()
    float_array = (ctypes.c_float * matrix_size).from_address(ctypes.addressof(ptr.contents))
    matrix_bytes_output = bytes(float_array)
    return matrix_bytes_output

class GameState (Enum):
    GAME_ACTIVE = 1
    GAME_MENU = 2
    GAME_WIN = 3

class SpriteRenderer():
    '''
    Generates OpenGL rendering requirements (VBO,VAO)\n
    and defines the square vertices and texture coordinates for sprite rendering\n
    Is meant to instantiate a render object for all sprites\n
    Textures are best being stored already in GPU memory via a Texture Manager object
    '''
    def __init__(self,shader:moderngl.Program,context:moderngl.Context):
        self.shader:moderngl.Program = shader
        self.context = context
        self.vao = None
        self.initRenderData()

    
    def initRenderData(self):
        # vertices of a square composed as usual of two triangles
        # texture coordinates corresponds to the full extent of the vertices
        # this is a common representation of 2D sprites in the 3D OpenGL plane
        vertices = [
            # pos     # tex
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 
        
            0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 0.0
        ]
        # uses Python's struct module to pack the list of floating-point numbers into a byte string
        # '32f': This is the format string. It specifies that we want to pack 32 floating-point numbers (f for float)
        # The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
        vertices_binaryformat = struct.pack(f"{len(vertices)}f",*vertices)
        # Define VBO (Vertex Buffer Object) containing vertex data
        vbo = self.context.buffer(vertices_binaryformat)
        vbo_parameters = [
            (vbo,"4f","vertex")
        ]
        # define VAO (Vertex Array Object)
        # essentially acts as a container that stores the state of vertex attributes. This includes:
        #    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
        #    The format of the vertex attributes (e.g., data type, number of components).
        #    Whether a particular vertex attribute is enabled or disabled.
        # NOTE: ebo with indices is not used in this example
        self.vao = self.context.vertex_array(self.shader,vbo_parameters)

    def DrawSprite(self,texture:moderngl.Context.texture,position:glm.vec2,size:glm.vec2 = glm.vec2(10.0,10.0),rotate:float = 0.0,color:glm.vec3 = glm.vec3(1.0)):
        '''
        Draw a sprite\n
        Texture is already stored in the GPU memory via a moderngl.context.texture object, routine only assingn it a location compatible with the shader\n
        position,size,rotation,color can be defined in input and are to be accepted by the fragment shader
        '''
        model = glm.mat4(1.0)
        model = glm.translate(model,glm.vec3(position,0.0))
        model = glm.translate(model,glm.vec3(0.5 * size.x, 0.5 * size.y, 0.0 ))
        model = glm.rotate(model, glm.radians(rotate), glm.vec3(0.0, 0.0, 1.0))
        model = glm.translate(model,glm.vec3(-0.5 * size.x, -0.5 * size.y, 0.0))
        model = glm.scale(model,glm.vec3(size,1.0))
        self.shader["model"].write(matrix_bytes(model))
        self.shader["spriteColor"].value = color
        self.shader["image"] = 0
        texture.use(location=0)
        self.vao.render()


class GameObject():
    def __init__(self,sprite:moderngl.Context.texture,position:glm.vec2 = glm.vec2(0.0,0.0), size:glm.vec2 = glm.vec2(1.0,1.0), velocity:glm.vec2 = glm.vec2(0.0,0.0),
                 color:glm.vec3 = glm.vec3(1.0,1.0,1.0), rotation:float = 0.0,issolid:bool = False,destroyed:bool = False
                  ):
        self.Position:glm.vec2 = position
        # size and are created as a new objects so it does not reference the initial default attribute with subsequent modifications in-game
        self.Size:glm.vec2 = glm.vec2(size)
        self.Velocity:glm.vec2 = glm.vec2(velocity)
        self.Color:glm.vec3 = color
        self.Rotation:float = rotation
        self.IsSolid:bool = issolid
        self.Destroyed:bool = destroyed
        self.Sprite:moderngl.Context.texture = sprite
    
    def Draw(self,renderer:SpriteRenderer):
        renderer.DrawSprite(self.Sprite,self.Position,self.Size,self.Rotation,self.Color)


class ParticleGenerator():

    def __init__(self,shader:moderngl.Program,context:moderngl.Context,texture:moderngl.Context.texture,amount:int):
        self.shader:moderngl.Program = shader
        self.context = context
        self.texture = texture
        self.amount = amount
        self.vao = None
        self.particles:list[Particle] = None
        self.lastUsedParticle:int = 0
        self.initRenderData()

    
    def initRenderData(self):
        # vertices of a square composed as usual of two triangles
        # texture coordinates corresponds to the full extent of the vertices
        # this is a common representation of 2D sprites in the 3D OpenGL plane
        vertices = [
            # pos     # tex
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 
        
            0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 0.0
        ]
        # uses Python's struct module to pack the list of floating-point numbers into a byte string
        # '24f': This is the format string. It specifies that we want to pack 24 floating-point numbers (f for float)
        # The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
        vertices_binaryformat = struct.pack(f"{len(vertices)}f",*vertices)
        # Define VBO (Vertex Buffer Object) containing vertex data
        vbo = self.context.buffer(vertices_binaryformat)
        vbo_parameters = [
            (vbo,"4f","vertex")
        ]
        # define VAO (Vertex Array Object)
        # essentially acts as a container that stores the state of vertex attributes. This includes:
        #    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
        #    The format of the vertex attributes (e.g., data type, number of components).
        #    Whether a particular vertex attribute is enabled or disabled.
        # NOTE: ebo with indices is not used in this example
        self.vao = self.context.vertex_array(self.shader,vbo_parameters)
        
        # generate particles list
        self.particles = [Particle() for i in range(self.amount)]

    def firstUnusedParticle(self)->int:
        # first search from last used particle, this will usually return almost instantly
        for index in range(self.lastUsedParticle,self.amount):
            if self.particles[index].Life <= 0.0:
                self.lastUsedParticle = index
                return index
                            
        # otherwise, do a linear search
        for index in range(self.lastUsedParticle,self.amount):
            if self.particles[index].Life <= 0.0:
                self.lastUsedParticle = index
                return index
        
        # all particles are taken, override the first one (note that if it repeatedly hits this case, more particles should be reserved)
        self.lastUsedParticle = 0
        return 0
    
    def respawnParticle(self,particle:Particle,object:GameObject,offset:glm.vec2)->None:        
        randomValue = ((random.randint(0,32767) % 100) - 50) / 10.0
        rColor = 0.5 + ((random.randint(0,32767) % 100) / 100.0)
        particle.Position = object.Position + randomValue + offset
        particle.Color = glm.vec4(rColor,rColor,rColor,1.0)
        particle.Life = 10.0 # much larger from learnopengl due to the use of normalized delta time
        particle.Velocity = object.Velocity * 0.1

    def Draw(self):
        # use additive blending to give it a 'glow' effect
        self.context.enable(moderngl.BLEND) # GEMINI
        self.context.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE)
        self.shader["sprite"] = 0
        self.texture.use(location=0)
        for particle in self.particles:
            if particle.Life > 0.0:
                self.shader["offset"].value = particle.Position
                self.shader["color"].value = particle.Color
                self.vao.render()
        # don't forget to reset to default blending mode
        self.context.disable(moderngl.BLEND)
        self.context.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)

    def Update(self,dt:float,object:GameObject,newParticles:int,offset:glm.vec2)->None:
        # add new particles
        for _ in range(newParticles):
            unusedParticle = self.firstUnusedParticle()
            self.respawnParticle(self.particles[unusedParticle],object,offset)
        # update all particles
        for particle in self.particles:
            particle.Life -= dt # reduce life
            if particle.Life > 0.0:
                particle.Position -= particle.Velocity * dt
                particle.Color.a -= dt * 0.05 # much smaller from learnopengl due to the use of normalized delta time

class GameLevel():
    def __init__(self,TextureManager:Textures):
        self.Bricks:list[GameObject] = []
        self.TextureManager = TextureManager

    def Load(self,file:str,levelWidth:int,levelHeight:int):
        self.Bricks.clear()
        try:
            with open(file, 'r') as f:
                # Nested list comprehension:
                # - Outer part iterates through lines in the file (after stripping)
                # - Inner part splits the line and converts each element to int
                tileData = [
                    [int(s_num) for s_num in line.strip().split()]
                    for line in f if line.strip() # Process non-empty lines
                ]
        except FileNotFoundError:
            print(f"Error: File not found at '{file}'")
            return [] # Return empty list on error
        except ValueError:
            print(f"Error: File '{file}' contains non-integer values.")
            return [] # Return empty list on error
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return [] # Return empty list on error
        if len(tileData) > 0:
            self._init(tileData,levelWidth,levelHeight)
    
    def Draw(self,renderer:SpriteRenderer):
        for brick in self.Bricks:
            if not brick.Destroyed:
                brick.Draw(renderer)

    def IsCompleted(self) -> bool:
        for brick in self.Bricks:
            if not brick.Destroyed:
                return False
        return True

    def _init(self,tileData:list[list[int]],levelWidth:int,levelHeight:int):
        height = len(tileData)
        width = len(tileData[0])
        unit_width = levelWidth / width
        unit_height = levelHeight / height
        # enumerate is used to count the current x and y offset of each brick
        for y,row in enumerate(tileData):
            for x,element in enumerate(row):
                position = glm.vec2(unit_width * x, unit_height * y)
                size = glm.vec2(unit_width,unit_height)
                match element:
                    case 1:
                        color=glm.vec3(0.8, 0.8, 0.7)
                        issolid = True
                        texture = self.TextureManager["block_solid"]
                    case 2:
                        color=glm.vec3(0.2, 0.6, 1.0)
                        issolid = False
                        texture = self.TextureManager["block"]
                    case 3:
                        color=glm.vec3(0.0, 0.7, 0.0)
                        issolid = False
                        texture = self.TextureManager["block"]
                    case 4:
                        color=glm.vec3(0.8, 0.8, 0.4)
                        issolid = False
                        texture = self.TextureManager["block"]
                    case 5:
                        color=glm.vec3(1.0, 0.5, 0.0)
                        issolid = False
                        texture = self.TextureManager["block"]
                if element in [1,2,3,4,5]:
                    object = GameObject(texture,position=position,size=size,color=color,issolid=issolid)
                    self.Bricks.append(object)


class BallObject(GameObject):
    def __init__(self,sprite:moderngl.Context.texture,radius:float = 12.5,position:glm.vec2 = glm.vec2(0.0,0.0), velocity:glm.vec2 = glm.vec2(0.0,0.0),stuck:bool = True):
        super().__init__(sprite=sprite,position=position,velocity=velocity,size=glm.vec2(radius * 2.0, radius * 2.0))
        self.Radius:float = radius
        self.Stuck:bool = stuck
        self.Sticky:bool = False
        self.PassThrough = False

    def Move(self,dt:float,window_width:int):
        if not self.Stuck:
            self.Position += self.Velocity * dt
            if self.Position.x < 0.0 :
                self.Velocity.x = -self.Velocity.x
                self.Position.x = 0.0
            elif self.Position.x + self.Size.x >= window_width :
                self.Velocity.x = -self.Velocity.x
                self.Position.x = window_width - self.Size.x
            if self.Position.y <= 0.0:
                self.Velocity.y = -self.Velocity.y
                self.Position.y = 0.0
        return self.Position
    
    def Reset(self,position:glm.vec2,velocity:glm.vec2):
        self.Position = position
        self.Velocity = velocity
        self.Stuck = True
        self.Sticky = False
        self.PassThrough = False

SIZE = glm.vec2(60.0, 20.0)
VELOCITY = glm.vec2(0.0,1.5)

class PowerUP(GameObject):
    def __init__(self,type:str,color:glm.vec3,duration:float,position:glm.vec2,texture:moderngl.Context.texture):
        super().__init__(texture,position,SIZE,VELOCITY,color,)
        self.Type:str = type
        self.Duration:float = duration
        self.Activated = False


# Collision routines (overloading with singledispatch)
@singledispatch
def CheckCollision(one:GameObject,two:GameObject) -> bool:
    collisionX:bool = one.Position.x + one.Size.x >= two.Position.x and two.Position.x + two.Size.x >= one.Position.x
    collisionY:bool = one.Position.y + one.Size.y >= two.Position.y and two.Position.y + two.Size.y >= one.Position.y
    return (collisionX and collisionY)

@CheckCollision.register(BallObject)
def CheckCollision_ball(one:BallObject,two:GameObject) -> tuple:
    # get center point circle first
    center = glm.vec2(one.Position + one.Radius)
    # calculate AABB info (center, half-extents)
    aabb_half_extents = glm.vec2(two.Size.x / 2.0, two.Size.y /2)
    aabb_center = glm.vec2(two.Position.x + aabb_half_extents.x, two.Position.y + aabb_half_extents.y)
    # get difference vector between both centers
    difference:glm.vec2 = center - aabb_center 
    clamped:glm.vec2 = glm.clamp(difference, -aabb_half_extents, aabb_half_extents)
    # add clamped value to AABB_center and we get the value of box closest to circle
    closest = aabb_center + clamped
    # retrieve vector between center circle and closest point AABB and check if length <= radius
    difference = closest - center
    result:bool = ( glm.length(difference) < one.Radius )
    # collision reaction
    direction = VectorDirection(difference)
    if result and direction is not Direction.ERROR:
        return (True,VectorDirection(difference),difference)
    else:
        return (False,Direction.UP,glm.vec2(0.0,0.0))

# Routines to implement collision reaction

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    # added because when paddle hits the ball while falling it ends up as an unmanaged case in standard learnopengl code
    # here we define an exception in the collision routine to just ignore the contact
    ERROR = -1 

def VectorDirection(target:glm.vec2) -> Direction:
    compass:list[glm.vec2] = [
        glm.vec2(0.0,1.0),  # up
        glm.vec2(1.0,0.0),  # right
        glm.vec2(0.0,-1.0), # down
        glm.vec2(-1.0,0.0)  # left
    ]
    max = 0.0
    best_match = -1
    for i,vector in enumerate(compass):
        dot_product = glm.dot(glm.normalize(target),vector)
        if dot_product > max:
            max = dot_product
            best_match = i
    return Direction(best_match)

class PostProcessor:
    def __init__(self,context:moderngl.Context,shader:moderngl.Program,width:int,height:int):
        self.vao = None
        self.context = context
        self.PostProcessingShader = shader
        self.width:int = width
        self.height:int = height
        self.Confuse:bool = False
        self.Chaos:bool = False
        self.Shake:bool = False
        num_fbo_msaa_samples = 4
        # Create texture buffer for a MULTISAMPLED FBO (depth buffer is not used in 2D)
        self.msaa_color_texture = context.texture((self.width,self.height), 4, samples=num_fbo_msaa_samples)
        # Create the MULTISAMPLED FBO (Anti Aliasing will be rendered here)
        self.msaa_framebuffer_object = context.framebuffer(color_attachments=[self.msaa_color_texture])
        # Create texture buffer for a SIMPLE SAMPLE FRAMEBUFFER (depth buffer is not used in 2D and will not be used in the resolved framebuffer to screen)
        # stencil support will be added soon to modernGL
        self.offscreen_color_texture = context.texture((self.width,self.height), 4)
        self.offscreen_color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Create Single Sample FRAMEBUFFER OBJECT that will be used to resolve (copy) the MSAA framebuffer to use with postprocessing
        self.framebuffer_object = context.framebuffer(color_attachments=[self.offscreen_color_texture])
        # re-enable default screen framebuffer (otherwise last configured FBO would still be active on rendering)
        self.context.screen.use()
        self.initRenderData()

        offset = 1.0 / 300.0
        self.offsets = [
            [-offset, offset], # top-left
            [ 0.0, offset   ], # top-center
            [ offset, offset], # top-right
            [-offset, 0.0   ], # center-left
            [ 0.0, 0.0      ], # center-center
            [ offset, 0.0   ], # center-right
            [-offset,-offset], # bottom-left
            [ 0.0, -offset  ], # bottom-center
            [ offset,-offset]  # bottom-right
        ]
        self.PostProcessingShader["offsets"] = self.offsets
        self.edge_kernel = [
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1
        ]
        self.PostProcessingShader["edge_kernel"] = self.edge_kernel
        self.blur_kernel = [
            1.0  / 16.0 , 2.0  / 16.0 , 1.0  / 16.0 ,
            2.0  / 16.0 , 4.0  / 16.0 , 2.0  / 16.0 ,
            1.0  / 16.0 , 2.0  / 16.0 , 1.0  / 16.0 
        ]
        self.PostProcessingShader["blur_kernel"] = self.blur_kernel


    def initRenderData(self):
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
        fbo_vbo = self.context.buffer(fbo_vertices_binaryformat)

        # VBO parameters to be passed to the VAO
        # This is what in modernGL is defined as "multiple buffers for all input variables"
        # meaning that each VBO buffer is described as a tuple in a list
        # elements of the tuple describes
        # 1) Vertex Buffer Object in input
        # 2) type of input parameters (4f in this case corresponds to a vec4 input) defined in shaders
        # 3) name of the input parameter in the related shader (vertex in this case)
        # NOTE: These parameters are the same also for the light source VAO
        fbo_vbo_parameters = [
            (fbo_vbo,"4f","vertex")
            ]
        
        # define VAO (Vertex Array Object)
        # essentially acts as a container that stores the state of vertex attributes. This includes:
        #    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
        #    The format of the vertex attributes (e.g., data type, number of components).
        #    Whether a particular vertex attribute is enabled or disabled.
        # NOTE: ebo with indices is not used in this example
        self.vao = self.context.vertex_array(self.PostProcessingShader,fbo_vbo_parameters)

    def BeginRender(self):
        self.msaa_framebuffer_object.use()
        self.msaa_framebuffer_object.clear(color=(0, 0, 0, 1.0))

    def EndRender(self):
        # resolve the MSAA framebuffer to a single sampled framebuffer for screen rendering
        # This transfers the antialiased image from msaa_framebuffer_object to resolve_fbo (and thus resolve_texture).
        self.context.copy_framebuffer(dst=self.framebuffer_object, src=self.msaa_framebuffer_object)
        self.context.screen.use()
        self.context.viewport = (0, 0, self.width, self.height)

    def Render(self,time:float):
        # Set uniforms
        self.PostProcessingShader["time"] = time
        self.PostProcessingShader["confuse"] = self.Confuse
        self.PostProcessingShader["chaos"] = self.Chaos
        self.PostProcessingShader["shake"] = self.Shake
        # program the shader to bind the scene sampler2D texture uniform to texture unit 0 
        # at rendering time will need to bind the resolved framebuffer to unit 0 with .use(location=0)
        self.offscreen_color_texture.use(location=0)
        self.PostProcessingShader["scene"].value = 0
        self.vao.render()

class Game:
    def __init__(self,width:int,height:int,state:GameState = GameState.GAME_ACTIVE):
        self.State:GameState = state
        self.Width:int = width
        self.Height:int = height
        self.Clock = pygame.time.Clock()
        self.screen_display = None
        self.context = None
        self.program = None
        self.renderer = None
        self.particles_renderer = None
        self.TextureManager = None
        self.levels:list[GameLevel] = []
        self.level:int = 0
        self.Player = None
        self.Ball = None
        # effects variables
        self.ShakeTime = 0.0
        # Powerups list
        self.PowerUps:list[PowerUP] = []
        # Audio dictionary
        self.audio:dict = {}

    
    def Init(self):
        pygame.init()
        screen_flags = pygame.OPENGL | pygame.RESIZABLE | pygame.DOUBLEBUF
        self.screen_display = pygame.display.set_mode((self.Width,self.Height),flags=screen_flags,vsync=VSYNC)
        # OPENGL init parameters for display
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE)
        # ModernGL create a context : a state machine or a container for OpenGL
        self.context = moderngl.create_context()
        self.program = self.context.program(vertex_shader=program_vs,fragment_shader=program_fs)
        # create shader program for particles
        self.program_particles = self.context.program(vertex_shader=particles_vs,fragment_shader=particles_fs)
        # create shader program for postprocessing effects
        self.program_effects = self.context.program(vertex_shader=postProcessing_vs,fragment_shader=postProcessing_fs)

        projection = glm.ortho(0.0,self.Width,self.Height,0.0,-1.0,1.0)
        self.program["projection"].write(matrix_bytes(projection))
        # set orthographic projection matrix for particle shaders
        self.program_particles["projection"].write(matrix_bytes(projection))

        self.TextureManager = Textures(self.context)
        # Texture loading
        self.TextureManager["face"] = "./assets/awesomeface.png"
        self.TextureManager["background"] = "./assets/background.jpg"
        self.TextureManager["block"] = "./assets/block.png"
        self.TextureManager["block_solid"] = "./assets/block_solid.png"
        self.TextureManager["paddle"] = "./assets/paddle.png"
        # load particle texture
        self.TextureManager["particle"] = "./assets/particle.png"
        # load powerups textures
        self.TextureManager["powerup_speed"] = "./assets/powerup_speed.png"
        self.TextureManager["powerup_sticky"] = "./assets/powerup_sticky.png"
        self.TextureManager["powerup_passthrough"] = "./assets/powerup_passthrough.png"
        self.TextureManager["powerup_increase"] = "./assets/powerup_increase.png"
        self.TextureManager["powerup_confuse"] = "./assets/powerup_confuse.png"
        self.TextureManager["powerup_chaos"] = "./assets/powerup_chaos.png"

        # Create Renderers
        self.renderer = SpriteRenderer(self.program,self.context)
        self.particles_renderer = ParticleGenerator(self.program_particles,self.context,self.TextureManager["particle"],500)
        self.effects = PostProcessor(self.context,self.program_effects,self.Width,self.Height)

        # Levels loading
        levelOne = GameLevel(self.TextureManager)
        levelTwo = GameLevel(self.TextureManager)
        levelThree = GameLevel(self.TextureManager)
        levelFour = GameLevel(self.TextureManager)
        levelOne.Load("./levels/one.lvl",self.Width,self.Height/2)
        levelTwo.Load("./levels/two.lvl",self.Width,self.Height/2)
        levelThree.Load("./levels/three.lvl",self.Width,self.Height/2)
        levelFour.Load("./levels/four.lvl",self.Width,self.Height/2)
        self.levels.append(levelOne)
        self.levels.append(levelTwo)
        self.levels.append(levelThree)
        self.levels.append(levelFour)
        self.level = 0
        # Load Player
        playerPos = glm.vec2(self.Width / 2.0 - PLAYER_SIZE.x / 2.0, self.Height - PLAYER_SIZE.y)
        self.Player = GameObject(self.TextureManager["paddle"],playerPos,PLAYER_SIZE)
        # Create the ball
        ballPos = playerPos + glm.vec2(PLAYER_SIZE.x / 2 - BALL_RADIUS, -BALL_RADIUS * 2.0)
        self.Ball = BallObject(self.TextureManager["face"],BALL_RADIUS,ballPos,INITIAL_BALL_VELOCITY)
        # Init audio
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.music.load("./assets/breakout.mp3")
        pygame.mixer.music.set_volume(0.1)
        pygame.mixer.music.play(loops=-1)
        self.audio["bleep"] = pygame.mixer.Sound("./assets/bleep.mp3")
        self.audio["solid"] = pygame.mixer.Sound("./assets/solid.wav")
        self.audio["powerup"] = pygame.mixer.Sound("./assets/powerup.wav")
        self.audio["bounce"] = pygame.mixer.Sound("./assets/bounce.wav")
        self.audio["bleep"].set_volume(0.2)
        self.audio["solid"].set_volume(0.2)
        self.audio["powerup"].set_volume(0.2)
        self.audio["bounce"].set_volume(0.2)
        

    def ProcessInput(self,dt:float,keys):
        if self.State is GameState.GAME_ACTIVE:
            velocity = PLAYER_VELOCITY * dt
            if keys[pygame.K_LEFT]:
                if self.Player.Position.x > 0.0:
                    self.Player.Position.x -= velocity
                    if self.Ball.Stuck:
                        self.Ball.Position.x -= velocity
            if keys[pygame.K_RIGHT]:
                if self.Player.Position.x <= self.Width - self.Player.Size.x:
                    self.Player.Position.x += velocity
                    if self.Ball.Stuck:
                        self.Ball.Position.x += velocity
            if keys[pygame.K_SPACE]:
                self.Ball.Stuck = False
    
    def Update(self,dt:float):
        # move the ball
        self.Ball.Move(dt,self.Width)
        # check collisions
        self.DoCollisions()
        # update particles
        self.particles_renderer.Update(dt,self.Ball,2,glm.vec2(self.Ball.Radius / 2.0))
        # update powerups
        self.UpdatePowerUps(dt)
        # reduce shake time
        if self.ShakeTime > 0.0:
            self.ShakeTime -= dt
            if self.ShakeTime <= 0.0:
                self.effects.Shake = False
        # reset game if ball hit the bottom edge
        if self.Ball.Position.y >= self.Height:
            self.ResetLevel()
            self.ResetPlayer()
            self.PowerUps.clear()

    def ResetLevel(self):
        match self.level:
            case 0:
                self.levels[0].Load("./levels/one.lvl",self.Width,self.Height /2)
            case 1:
                self.levels[1].Load("./levels/two.lvl",self.Width,self.Height /2)
            case 2:
                self.levels[2].Load("./levels/three.lvl",self.Width,self.Height /2)
            case 3:
                self.levels[3].Load("./levels/four.lvl",self.Width,self.Height /2)

    def ResetPlayer(self):
        self.Player.Size = glm.vec2(PLAYER_SIZE) # avoid referencing the original player size
        self.Player.Position = glm.vec2(self.Width / 2.0 - PLAYER_SIZE.x / 2.0, self.Height - PLAYER_SIZE.y)
        self.Ball.Reset(self.Player.Position + glm.vec2(PLAYER_SIZE.x / 2.0 - BALL_RADIUS, -(BALL_RADIUS * 2.0)),INITIAL_BALL_VELOCITY)
        # disable active powerups
        self.effects.Chaos = False
        self.effects.Confuse = False
        self.Ball.PassThrough = False
        self.Ball.Sticky = False
        self.Player.Color = glm.vec3(1.0)
        self.Ball.Color = glm.vec3(1.0)

    
    def Render(self):
        if self.State is GameState.GAME_ACTIVE:
            # ---> begin rendering effects
            self.effects.BeginRender()
            # Draw background
            self.renderer.DrawSprite(self.TextureManager["background"],glm.vec2(0.0,0.0),glm.vec2(self.Width,self.Height),0)
            # Draw level
            self.levels[self.level].Draw(self.renderer)
            # Draw Player
            self.Player.Draw(self.renderer)
            # Draw powerups
            for powerup in self.PowerUps:
                if not powerup.Destroyed:
                    powerup.Draw(self.renderer)
            # Draw particles
            self.particles_renderer.Draw()
            # Draw Ball
            self.Ball.Draw(self.renderer)
            # ---> end rendering effects
            self.effects.EndRender()
            # render postoprocessing screen
            self.effects.Render(pygame.time.get_ticks() * 0.0007)

    def Run(self):
        while True:
            NormalizedDeltaTime = self.Clock.tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE # set the FPS of the game to not exceed the framerate
            # Avoid excessive cumulative increase of Delta Time due to moving the windows around or delay in game processing
            if NormalizedDeltaTime > 15:
                NormalizedDeltaTime = 15.0
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if  event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.ProcessInput(NormalizedDeltaTime,keys)
            self.Update(NormalizedDeltaTime)
            self.Render() # <-- Rendering logic
            pygame.display.flip()
            self.context.clear() # clears the framebuffer (Necessary and also best practice) 

    def DoCollisions(self):
        # check collisions of the ball with bricks
        for brick in self.levels[self.level].Bricks:
            if not brick.Destroyed:
                collision:tuple = CheckCollision(self.Ball,brick)
                if collision[0]:
                    if not brick.IsSolid:
                        brick.Destroyed = True
                        self.SpawnPowerUps(brick)
                        self.audio["bleep"].play()
                    else:
                        # if block is solid enable shake effect
                        self.ShakeTime = 20
                        self.effects.Shake = True
                        self.audio["solid"].play()
                    # collision resolution
                    direction:Direction = collision[1]
                    difference_vector_v = collision[2]
                    if not (self.Ball.PassThrough and not brick.IsSolid):
                        # horizontal collisions
                        if (direction is Direction.LEFT or direction is Direction.RIGHT):
                            self.Ball.Velocity.x = -self.Ball.Velocity.x # reverse horizontal velocity
                            # relocate the ball
                            penetration_r:float = self.Ball.Radius - abs(difference_vector_v.x)
                            if direction is Direction.LEFT:
                                self.Ball.Position.x += penetration_r # move ball right (opposite to direction)
                            else:
                                self.Ball.Position.x -= penetration_r # move the ball left (opposite to direction)
                        else: # vertical collisions
                            self.Ball.Velocity.y = -self.Ball.Velocity.y # reverse on vertical
                            penetration_r:float = self.Ball.Radius - abs(difference_vector_v.y)
                            if direction is Direction.UP:
                                self.Ball.Position.y -= penetration_r # move ball up
                            else:
                                self.Ball.Position.y += penetration_r # move ball down

        # check also powerup collisions
        for powerup in self.PowerUps:
            if not powerup.Destroyed:
                if powerup.Position.y >= self.Height:
                    powerup.Destroyed = True
                if CheckCollision(self.Player,powerup):
                    self.ActivatePowerUp(powerup)
                    powerup.Destroyed = True
                    powerup.Activated = True
                    self.audio["powerup"].play()

        # check collisions of ball and player
        collision_player:tuple = CheckCollision(self.Ball,self.Player)
        if ( not self.Ball.Stuck and collision_player[0] ):
            # check where it hit the board, and change velocity based on where it hit the board
            centerBoard:float = self.Player.Position.x + (self.Player.Size.x / 2.0)
            distance:float = (self.Ball.Position.x + self.Ball.Radius) - centerBoard
            percentage:float = distance / (self.Player.Size.x / 2.0)
            # move based on impact
            strenght:float = 2.0
            oldVelocity:glm.vec2 = self.Ball.Velocity
            self.Ball.Velocity.x = INITIAL_BALL_VELOCITY.x * percentage * strenght
            #self.Ball.Velocity.y = -self.Ball.Velocity.y
            self.Ball.Velocity = glm.normalize(self.Ball.Velocity) * glm.length(oldVelocity)
            self.Ball.Velocity.y = -1.0 * abs(self.Ball.Velocity.y)
            # if Sticky powerup is activated, also stick ball to paddle once new velocity vectors were calculated
            self.Ball.Stuck = self.Ball.Sticky
            
        

    
    def ActivatePowerUp(self,powerup:PowerUP):
        match powerup.Type:
            case "speed":
                self.Ball.Velocity *= 1.2
            case "sticky":
                self.Ball.Sticky = True
                self.Player.Color = glm.vec3(1.0,0.5,1.0)
            case "pass-through":
                self.Ball.PassThrough = True
                self.Ball.Color = glm.vec3(1.0,0.5,0.5)
            case "pad-size-increase":
                self.Player.Size.x += 50
            case "confuse":
                if not self.effects.Chaos:
                    self.effects.Confuse = True
            case "chaos":
                if not self.effects.Confuse:
                    self.effects.Chaos = True

    def SpawnPowerUps(self, block:GameObject):
        
        def ShouldSpawn(chance:int) -> bool:
            random_value = random.randrange(chance)
            return random_value == 0
        
        stickyDuration = 20 * FRAMERATE_REFERENCE
        passthroughDuration = 10 * FRAMERATE_REFERENCE
        confuseDuration = 15 * FRAMERATE_REFERENCE
        chaosDuration = 15 * FRAMERATE_REFERENCE
        
        if ShouldSpawn(75):
            self.PowerUps.append(PowerUP("speed",glm.vec3(0.5,0.5,1.0),0.0,block.Position,self.TextureManager["powerup_speed"]))
        if ShouldSpawn(75):
            self.PowerUps.append(PowerUP("sticky",glm.vec3(1.0,0.5,1.0),stickyDuration,block.Position,self.TextureManager["powerup_sticky"]))
        if ShouldSpawn(75):
            self.PowerUps.append(PowerUP("pass-through",glm.vec3(0.5,1.0,0.5),passthroughDuration,block.Position,self.TextureManager["powerup_passthrough"]))
        if ShouldSpawn(75):
            self.PowerUps.append(PowerUP("pad-size-increase",glm.vec3(1.0,0.6,0.4),0.0,block.Position,self.TextureManager["powerup_increase"]))
        if ShouldSpawn(15):
            self.PowerUps.append(PowerUP("confuse",glm.vec3(1.0,0.3,0.3),confuseDuration,block.Position,self.TextureManager["powerup_confuse"]))
        if ShouldSpawn(15):
            self.PowerUps.append(PowerUP("chaos",glm.vec3(0.9,0.25,0.25),chaosDuration,block.Position,self.TextureManager["powerup_chaos"]))

    def UpdatePowerUps(self,dt:float):
        def isOtherPowerUpActive(powerups:list[PowerUP],type:str):
            '''
            check if the same powerup is already active
            '''
            for powerup in powerups:
                if powerup.Activated:
                    if powerup.Type == type:
                        return True
            return False

        for powerup in self.PowerUps:
            powerup.Position += powerup.Velocity * dt
            if powerup.Activated:
                powerup.Duration -= dt
                if powerup.Duration <= 0:
                    # remove powerup from list
                    powerup.Activated = False
                    match powerup.Type:
                        case "sticky":
                            if not isOtherPowerUpActive(self.PowerUps,"sticky"):
                                self.Ball.Sticky = False
                                self.Player.Color = glm.vec3(1.0)
                        case "pass-through":
                            if not isOtherPowerUpActive(self.PowerUps,"pass-through"):
                                self.Ball.PassThrough = False
                                self.Player.Color = glm.vec3(1.0)
                        case "confuse":
                            if not isOtherPowerUpActive(self.PowerUps,"confuse"):
                                self.effects.Confuse = False
                        case "chaos":
                            if not isOtherPowerUpActive(self.PowerUps,"chaos"):
                                self.effects.Chaos = False
        
        self.PowerUps = [powerup for powerup in self.PowerUps if powerup.Activated or (not powerup.Destroyed)]

if __name__ == "__main__":
    random.seed(int(time.time()))
    Breakout = Game(WIDTH,HEIGHT)
    Breakout.Init()
    Breakout.Run()