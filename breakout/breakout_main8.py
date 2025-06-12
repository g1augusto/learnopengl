from enum import Enum # to define movement enum class
import sys
import struct
import ctypes
from functools import singledispatch # for function overloading
from dataclasses import dataclass
import time
import random
import os
import math
import pygame
import moderngl
import glm

# GLOBAL CONSTANTS
FRAMERATE = 60                          # Framerate to use in clock tick method
FRAMERATE_REFERENCE = 60                # Reference from Framerate Target to be achieved
WIDTH = 800
HEIGHT = 600
VSYNC = False
INITIAL_PLAYER_SIZE = glm.vec2(100.0,20)
PLAYER_SIZE = glm.vec2(100.0,20)
INITIAL_PLAYER_VELOCITY = 10.0
PLAYER_VELOCITY = 10.0
INITIAL_BALL_VELOCITY = glm.vec2(5.0,-5.0)
BALL_VELOCITY = glm.vec2(5.0,-5.0)
INITIAL_BALL_RADIUS = 12.5
BALL_RADIUS = 12.5
INITIAL_POWERUP_SIZE = glm.vec2(60.0, 20.0)
POWERUP_SIZE = glm.vec2(60.0, 20.0)
INITIAL_POWERUP_VELOCITY = glm.vec2(0.0,1.5)
POWERUP_VELOCITY = glm.vec2(0.0,1.5)
INITIAL_PARTICLE_SCALE = 10.0
PARTICLE_SCALE = 10.0


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
uniform float scale;

void main()
{
    //float scale = 10.0f;
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


text_vs = '''
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
'''

text_fs = '''
#version 330 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_texture;

void main() {
    out_color = texture(u_texture, v_uv);
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

class Movement (Enum):
    NONE = 0
    LEFT = -1
    RIGHT = 1

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

    def __init__(self,shader:moderngl.Program,context:moderngl.Context,texture:moderngl.Context.texture,amount:int,scale:float = PARTICLE_SCALE):
        self.shader:moderngl.Program = shader
        self.context = context
        self.texture = texture
        self.amount = amount
        self.vao = None
        self.particles:list[Particle] = None
        self.lastUsedParticle:int = 0
        self.scale = scale
        self.initRenderData()

    def Resize(self,ballRadius):
        ratio = ballRadius / INITIAL_BALL_RADIUS
        self.scale = INITIAL_PARTICLE_SCALE * ratio

    
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
    
    def respawnParticle(self,particle:Particle,obj:GameObject,offset:glm.vec2)->None:        
        randomValue = ((random.randint(0,32767) % 100) - 50) / 10.0
        rColor = 0.5 + ((random.randint(0,32767) % 100) / 100.0)
        particle.Position = obj.Position + randomValue + offset
        particle.Color = glm.vec4(rColor,rColor,rColor,1.0)
        particle.Life = 10.0 # much larger from learnopengl due to the use of normalized delta time
        particle.Velocity = obj.Velocity * 0.1

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
                self.shader["scale"].value = self.scale
                self.vao.render()
        # don't forget to reset to default blending mode
        self.context.disable(moderngl.BLEND)
        self.context.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)

    def Update(self,dt:float,obj:GameObject,newParticles:int,offset:glm.vec2)->None:
        # add new particles
        for _ in range(newParticles):
            unusedParticle = self.firstUnusedParticle()
            self.respawnParticle(self.particles[unusedParticle],obj,offset)
        # update all particles
        for particle in self.particles:
            particle.Life -= dt # reduce life
            if particle.Life > 0.0:
                particle.Position -= particle.Velocity * dt
                particle.Color.a -= dt * 0.05 # much smaller from learnopengl due to the use of normalized delta time

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
        TextRenderer._shader_program['u_text_dimensions'].value = (self.text_width, self.text_height)
        TextRenderer._shader_program['u_text_offset'].value = tuple(self.actual_position)

        self.text_texture.use(location=0) # Bind to texture unit 0
        TextRenderer._shader_program['u_texture'].value = 0 # Tell sampler to use texture unit 0

        # Render the shared VAO
        self.ctx.enable(moderngl.BLEND)
        TextRenderer._unit_quad_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.disable(moderngl.BLEND)


        

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
            self._init(tileData,levelWidth,levelHeight/2)
    
    def Draw(self,renderer:SpriteRenderer):
        for brick in self.Bricks:
            if not brick.Destroyed:
                brick.Draw(renderer)

    def IsCompleted(self) -> bool:
        for brick in self.Bricks:
            if (not brick.IsSolid) and not (brick.Destroyed):
                return False
        return True

    def _init(self,tileData:list[list[int]],levelWidth:int,levelHeight:int):
        self.height = len(tileData)
        self.width = len(tileData[0])
        self.unit_width = levelWidth / self.width
        self.unit_height = levelHeight / self.height
        # enumerate is used to count the current x and y offset of each brick
        for y,row in enumerate(tileData):
            for x,element in enumerate(row):
                position = glm.vec2(self.unit_width * x, self.unit_height * y)
                size = glm.vec2(self.unit_width,self.unit_height)
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

    def Resize(self,levelWidth:int,levelHeight:int):
        self.unit_width = levelWidth / self.width
        self.unit_height = levelHeight / self.height
        newSize = glm.vec2(self.unit_width,self.unit_height)
        Bricks:list[GameObject] = []
        brick:GameObject
        for brick in self.Bricks:
            ratio = newSize / brick.Size
            obj = GameObject(brick.Sprite,position=brick.Position*ratio,size=newSize,color=brick.Color,issolid=brick.IsSolid,destroyed=brick.Destroyed)
            Bricks.append(obj)
        del self.Bricks
        self.Bricks = Bricks

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
    
    def Resize(self,radius:float,velocity:glm.vec2):
        self.Radius = radius
        self.Size = glm.vec2(radius * 2.0, radius * 2.0)
        self.Velocity = velocity
    
    def Reset(self,position:glm.vec2,velocity:glm.vec2):
        self.Position = position
        self.Velocity = velocity
        self.Stuck = True
        self.Sticky = False
        self.PassThrough = False


class PowerUP(GameObject):
    def __init__(self,type:str,color:glm.vec3,duration:float,position:glm.vec2,texture:moderngl.Context.texture):
        super().__init__(texture,position,POWERUP_SIZE,POWERUP_VELOCITY,color,)
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
        self.num_fbo_msaa_samples = 4

        self.Resize(width,height)
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

    def Resize(self,width,height):
        self.width = width
        self.height = height
        # Create texture buffer for a MULTISAMPLED FBO (depth buffer is not used in 2D)
        self.msaa_color_texture = self.context.texture((self.width,self.height), 4, samples=self.num_fbo_msaa_samples)
        # Create the MULTISAMPLED FBO (Anti Aliasing will be rendered here)
        self.msaa_framebuffer_object = self.context.framebuffer(color_attachments=[self.msaa_color_texture])
        # Create texture buffer for a SIMPLE SAMPLE FRAMEBUFFER (depth buffer is not used in 2D and will not be used in the resolved framebuffer to screen)
        # stencil support will be added soon to modernGL
        self.offscreen_color_texture = self.context.texture((self.width,self.height), 4)
        self.offscreen_color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Create Single Sample FRAMEBUFFER OBJECT that will be used to resolve (copy) the MSAA framebuffer to use with postprocessing
        self.framebuffer_object = self.context.framebuffer(color_attachments=[self.offscreen_color_texture])
        # re-enable default screen framebuffer (otherwise last configured FBO would still be active on rendering)
        self.context.screen.use()

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
        self.program_particles = None
        self.program_effects = None
        self.program_text = None
        self.renderer = None
        self.particles_renderer = None
        self.TextureManager = None
        self.levels:list[GameLevel] = []
        self.level:int = 0
        self.Player = None
        self.PlayerMovement = Movement.NONE
        self.Ball = None
        # effects variables
        self.ShakeTime = 0.0
        # Powerups list
        self.PowerUps:list[PowerUP] = []
        # Audio dictionary
        self.audio:dict = {}
        # Text Interface list (will be all rendered at the end of Draw)
        self.textList:dict[TextRenderer] = {}
        # Game lives
        self.lives = 3

    def SetScreenSize(self,width,height):
        global PLAYER_SIZE
        global BALL_RADIUS
        global PLAYER_VELOCITY
        global BALL_VELOCITY
        global POWERUP_VELOCITY
        
        if (self.Width != width) or (self.Height != height):
            new_resolution = glm.vec2(width,height)
            standard_resolution = glm.vec2(WIDTH,HEIGHT)
            previous_resolution = glm.vec2(self.Width,self.Height)
            ratio = new_resolution / standard_resolution
            ratio_vs_previous = new_resolution / previous_resolution
            PLAYER_SIZE = INITIAL_PLAYER_SIZE * ratio
            BALL_RADIUS = INITIAL_BALL_RADIUS * (ratio.y if ratio.y < ratio.x else ratio.x)
            PLAYER_VELOCITY = INITIAL_PLAYER_VELOCITY * ratio.x
            BALL_VELOCITY *= ratio_vs_previous
            POWERUP_VELOCITY = INITIAL_POWERUP_VELOCITY * ratio
            self.Width = width
            self.Height = height
            self.Ball.Resize(BALL_RADIUS,BALL_VELOCITY)
            if self.State is GameState.GAME_ACTIVE:
                # resize and reposition player in place
                self.Player.Size = glm.vec2(PLAYER_SIZE) # avoid referencing the original player size
                self.Player.Position = glm.vec2(self.Player.Position.x * ratio_vs_previous.x , self.Height - PLAYER_SIZE.y)
                self.Ball.Position *= ratio_vs_previous
            else:
                self.ResetPlayer()
        self.screen_display = pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.RESIZABLE | pygame.DOUBLEBUF, vsync=VSYNC)
        self.context.viewport = (0, 0, width, height)
        # calculate and upload projection matrix
        projection = glm.ortho(0.0,width,height,0.0,-1.0,1.0)
        self.program["projection"].write(matrix_bytes(projection))
        # set orthographic projection matrix for particle shaders
        self.program_particles["projection"].write(matrix_bytes(projection))
        # set orthographic projection matrix for text shader
        self.program_text["projection"].write(matrix_bytes(projection))



    def ResizeScreen(self,width,height):
        self.SetScreenSize(width,height)
        self.effects.Resize(width,height)
        self.levels[self.level].Resize(width,height/2)
        self.InitText()
        self.ResizePowerUps(width,height)
        self.particles_renderer.Resize(self.Ball.Radius)
    
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
        # create shader program for text
        self.program_text = self.context.program(vertex_shader=text_vs,fragment_shader=text_fs)

        self.SetScreenSize(self.Width,self.Height)        

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
        levelOne.Load("./levels/one.lvl",self.Width,self.Height)
        levelTwo.Load("./levels/two.lvl",self.Width,self.Height)
        levelThree.Load("./levels/three.lvl",self.Width,self.Height)
        levelFour.Load("./levels/four.lvl",self.Width,self.Height)
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
        self.Ball = BallObject(self.TextureManager["face"],BALL_RADIUS,ballPos,BALL_VELOCITY)
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
        # Init Text Renderer
        TextRenderer.initialize_shared_resources(self.context, self.program_text)
        self.InitText()
        # init game icon
        self.pygame_icon = pygame.image.load("./assets/breakout_window_icon.png").convert_alpha()
        pygame.display.set_caption("Breakout OpenGL")  # Set the window title
        pygame.display.set_icon(self.pygame_icon) # Set the icon from the image previously loaded
        
    def InitText(self):
        ratio = ((self.Width * self.Height) / (WIDTH * HEIGHT))
        # here we limit the ratio to prevent excessive resizing
        if ratio < 0.5:
            ratio = 0.5
        elif ratio > 1.5:
            ratio = 1.5
        # retrieve the status of the UI text messages if visible or not
        # because this routine is called at resize AND at the init of the game
        # during init phase the objects does not exists yet and the default value of getattr is returned (considering the design)
        # to retrieve the data safely (for object and relative attribute) we use getattr and get which achieve the same safe access
        # respectively retrieving an object attribute and retrieving an object from a dictionary
        livesVisible = getattr(self.textList.get("lives"),"visible",True)
        messageVisible = getattr(self.textList.get("message"),"visible",True)
        winVisible = getattr(self.textList.get("win"),"visible",False)
        levelVisible = getattr(self.textList.get("level"),"visible",True)
        powerupVisible = getattr(self.textList.get("powerup"),"visible",True)
        self.textList["lives"] = TextRenderer(self.context, f"Lives {self.lives}", "./assets/vastorga.ttf", int(70*ratio), (255, 255, 0), glm.vec2(0, 0),bold=False,italic=False,visible=livesVisible,alignment=TextAlign.LEFT_TOP)
        self.textList["message"] = TextRenderer(self.context, f"{'Breakout':^25}\n{'Press ENTER to start':^20}", "./assets/vastorga.ttf", int(70*ratio), (255, 255, 0), glm.vec2(self.Width/2, self.Height/2),bold=True,italic=True,visible=messageVisible,alignment=TextAlign.CENTER)
        self.textList["win"] = TextRenderer(self.context, f"{'YOU WON!!!!':^27}\n{'Congratulations!':^27}\n{'Press ENTER to try again!':^25}", "./assets/vastorga.ttf", int(70*ratio), (255, 255, 0), glm.vec2(self.Width/2, self.Height/2),bold=True,italic=False,visible=winVisible)
        self.textList["level"] = TextRenderer(self.context, f"Level {self.level+1}", "./assets/vastorga.ttf", int(70*ratio), (255, 255, 0), glm.vec2(self.Width, 0),bold=False,italic=False,visible=levelVisible,alignment=TextAlign.RIGHT_TOP)
        self.textList["powerup"] = TextRenderer(self.context, f"Powerup: ", "./assets/vastorga.ttf", int(30*ratio), (255, 255, 255), glm.vec2(self.Width, self.Height),bold=False,italic=False,visible=powerupVisible,alignment=TextAlign.RIGHT_BOTTOM)
         

    def ProcessInput(self,dt:float,keys,controlKey):
        self.PlayerMovement = Movement.NONE
        if self.State is GameState.GAME_ACTIVE:
            velocity = PLAYER_VELOCITY * dt
            if keys[pygame.K_LEFT]:
                if self.Player.Position.x > 0.0:
                    self.Player.Position.x -= velocity
                    self.PlayerMovement = Movement.LEFT
                    if self.Ball.Stuck:
                        self.Ball.Position.x -= velocity
            if keys[pygame.K_RIGHT]:
                if self.Player.Position.x <= self.Width - self.Player.Size.x:
                    self.Player.Position.x += velocity
                    self.PlayerMovement = Movement.RIGHT
                    if self.Ball.Stuck:
                        self.Ball.Position.x += velocity
            if keys[pygame.K_SPACE]:
                self.Ball.Stuck = False
        elif self.State is GameState.GAME_MENU:
            if controlKey == pygame.K_RETURN:
                self.State = GameState.GAME_ACTIVE
                self.textList["message"].visible = False
        elif self.State is GameState.GAME_WIN:
            if controlKey == pygame.K_RETURN:
                self.level = 0
                self.ResetLevel()
                self.effects.Chaos = False
                self.State = GameState.GAME_MENU
                self.textList["win"].visible = False
                self.textList["message"].visible = True
            
    
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
            self.lives -= 1
            if self.lives == 0:
                self.lives = 3
                self.level = 0
                self.ResetLevel()
                self.PowerUps.clear()
                self.State = GameState.GAME_MENU
                self.textList["message"].visible = True
            self.ResetPlayer()
            self.textList["lives"].update_text(f"Lives {self.lives}")
        # Check winning condition for level
        if self.State is GameState.GAME_ACTIVE and self.levels[self.level].IsCompleted():
            self.level += 1
            if self.level < len(self.levels):
                self.ResetLevel()
                self.ResetPlayer()
            else:
                self.level -= 1
                self.ResetPlayer()
                self.effects.Chaos = True
                self.State = GameState.GAME_WIN
                self.textList["win"].visible = True


    def ResetLevel(self):
        match self.level:
            case 0:
                self.levels[0].Load("./levels/one.lvl",self.Width,self.Height)
            case 1:
                self.levels[1].Load("./levels/two.lvl",self.Width,self.Height)
            case 2:
                self.levels[2].Load("./levels/three.lvl",self.Width,self.Height)
            case 3:
                self.levels[3].Load("./levels/four.lvl",self.Width,self.Height)
        self.textList["level"].update_text(f"Level {self.level+1}")
        self.textList["level"].set_position(self.textList["level"].position)


    def ResetPlayer(self):
        self.Player.Size = glm.vec2(PLAYER_SIZE) # avoid referencing the original player size
        self.Player.Position = glm.vec2(self.Width / 2.0 - PLAYER_SIZE.x / 2.0, self.Height - PLAYER_SIZE.y)
        self.Ball.Reset(self.Player.Position + glm.vec2(PLAYER_SIZE.x / 2.0 - BALL_RADIUS, -(BALL_RADIUS * 2.0)),BALL_VELOCITY)

        # disable active powerups
        self.effects.Chaos = False
        self.effects.Confuse = False
        self.Ball.PassThrough = False
        self.Ball.Sticky = False
        self.Player.Color = glm.vec3(1.0)
        self.Ball.Color = glm.vec3(1.0)
        self.PowerUps.clear()

    
    def Render(self):
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
        # Render Text
        key:str
        text:TextRenderer
        for key,text in self.textList.items():
            if text.visible:
                if key == "message":
                    angleSpeed = 1.0
                    angle = time.time() * angleSpeed
                    radius = 50.0
                    colorFactor = time.time() % 255
                    offset = glm.vec2(radius * math.cos(angle),radius * math.sin(angle))
                    text.set_position(glm.vec2(self.Width/2,self.Height/2)+offset)
                    text.text_rgb_color = (colorFactor,255,255 - colorFactor)
                    text.update_text(text.text)
                text.render()

    def Run(self):
        while True:
            NormalizedDeltaTime = self.Clock.tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE # set the FPS of the game to not exceed the framerate
            # Avoid excessive cumulative increase of Delta Time due to moving the windows around or delay in game processing
            if NormalizedDeltaTime > 15:
                NormalizedDeltaTime = 15.0
            keys = pygame.key.get_pressed()
            controlKey = None
            for event in pygame.event.get():
                if  event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        controlKey = event.key
                elif event.type == pygame.VIDEORESIZE:
                    self.ResizeScreen(event.w,event.h)
            self.ProcessInput(NormalizedDeltaTime,keys,controlKey)
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
        collision_player: tuple = CheckCollision(self.Ball, self.Player)
        if not self.Ball.Stuck and collision_player[0]:
            # Store the ball's speed and original Y direction before calculations
            old_ball_velocity_full = glm.vec2(self.Ball.Velocity) 
            old_ball_speed_magnitude: float = glm.length(old_ball_velocity_full)

            # 1. Calculate desired new X component based on hit position on paddle
            center_board: float = self.Player.Position.x + (self.Player.Size.x / 2.0)
            distance_from_center: float = (self.Ball.Position.x + self.Ball.Radius) - center_board
            percentage: float = distance_from_center / (self.Player.Size.x / 2.0) # -1 to 1
            
            # Base horizontal effect from hit position
            # Using INITIAL_BALL_VELOCITY.x helps keep this effect's scale consistent
            new_ball_velocity_x: float = INITIAL_BALL_VELOCITY.x * percentage * 2.0 # Original strength: 2.0

            # 2. Add influence from the paddle's own movement
            paddle_movement_influence_factor: float = 0.7 # TUNABLE: Adjust for desired "push"
            if self.PlayerMovement != Movement.NONE:
                paddle_motion_bonus: float = abs(INITIAL_BALL_VELOCITY.x) * \
                                           paddle_movement_influence_factor * \
                                           self.PlayerMovement.value
                new_ball_velocity_x += paddle_motion_bonus
            
            # 3. Determine the Y component for reflection.
            # It should generally be the opposite of the incoming Y velocity.
            new_ball_velocity_y = -old_ball_velocity_full.y 

            # If the ball was moving very flatly (small old_ball_velocity_full.y), 
            # ensure the reflected Y has a reasonable minimum magnitude.
            min_reflected_y_magnitude = abs(INITIAL_BALL_VELOCITY.y * 0.3 if INITIAL_BALL_VELOCITY.y != 0 else 0.5)
            if abs(new_ball_velocity_y) < min_reflected_y_magnitude:
                new_ball_velocity_y = -min_reflected_y_magnitude # Ensure it's upward and has minimum magnitude

            # 4. Combine X and Y into a new velocity vector and maintain original speed
            temp_new_velocity = glm.vec2(new_ball_velocity_x, new_ball_velocity_y)

            if old_ball_speed_magnitude > 0.001: # If the ball was actually moving
                if glm.length(temp_new_velocity) > 0.001:
                    self.Ball.Velocity = glm.normalize(temp_new_velocity) * old_ball_speed_magnitude
                else:
                    # temp_new_velocity is zero (e.g., X and Y components cancelled out - very unlikely for paddle hit)
                    # Default to a simple upward bounce with some X influence
                    self.Ball.Velocity.x = INITIAL_BALL_VELOCITY.x * 0.1 * \
                                           (self.PlayerMovement.value if self.PlayerMovement != Movement.NONE else \
                                            (percentage if percentage !=0 else 1))
                    self.Ball.Velocity.y = -abs(INITIAL_BALL_VELOCITY.y if INITIAL_BALL_VELOCITY.y != 0 else old_ball_speed_magnitude * 0.5 if old_ball_speed_magnitude >0 else 1.0)
                    
                    # Rescale this default escape
                    if glm.length(self.Ball.Velocity) > 0.001:
                        self.Ball.Velocity = glm.normalize(self.Ball.Velocity) * old_ball_speed_magnitude
                    else: # Still zero? Force some minimal upward movement.
                        self.Ball.Velocity.x = 0
                        self.Ball.Velocity.y = -(old_ball_speed_magnitude if old_ball_speed_magnitude > 0.1 else 1.0) 
            else:
                # Ball was stationary before impact (shouldn't happen if !self.Ball.Stuck, but as a fallback)
                self.Ball.Velocity = temp_new_velocity
                default_speed_if_was_stuck = abs(INITIAL_BALL_VELOCITY.y if INITIAL_BALL_VELOCITY.y != 0 else 1.0)
                if glm.length(self.Ball.Velocity) > 0.001:
                    self.Ball.Velocity = glm.normalize(self.Ball.Velocity) * default_speed_if_was_stuck
                else: # If temp_new_velocity from calculations was zero
                    self.Ball.Velocity.x = 0.0
                    self.Ball.Velocity.y = -default_speed_if_was_stuck

            # --- CRITICAL FINAL ENFORCEMENT for Y direction and minimum speed ---
            # Step 5a: Ensure Y is negative (upwards).
            # This takes the magnitude of Y calculated from normalization and ensures its sign is negative.
            if self.Ball.Velocity.y >= 0.0: 
                self.Ball.Velocity.y = -abs(self.Ball.Velocity.y) 
                # If Y was 0, and -abs(0) is 0, it needs a definite upward component.
                if self.Ball.Velocity.y == 0.0:
                    min_up_push = abs(INITIAL_BALL_VELOCITY.y * 0.2 if INITIAL_BALL_VELOCITY.y != 0 else (old_ball_speed_magnitude * 0.1 if old_ball_speed_magnitude >0 else 0.3))
                    self.Ball.Velocity.y = -max(min_up_push, 0.1) # Ensure at least a small upward velocity like -0.1

            # Step 5b: (Optional but recommended) Enforce a minimum absolute vertical speed to prevent overly "flat" bounces.
            # This helps if the ball's trajectory becomes too horizontal after all calculations.
            min_vertical_speed_ratio = 0.2 # Ball's Y speed should be at least 20% of its typical vertical speed or overall speed.
            # Determine a reference magnitude for the minimum Y speed
            reference_for_min_y_abs = abs(INITIAL_BALL_VELOCITY.y if INITIAL_BALL_VELOCITY.y != 0 else (old_ball_speed_magnitude if old_ball_speed_magnitude > 0 else 1.0))
            min_abs_y_target_velocity = reference_for_min_y_abs * min_vertical_speed_ratio

            if abs(self.Ball.Velocity.y) < min_abs_y_target_velocity and min_abs_y_target_velocity > 0.01:
                self.Ball.Velocity.y = -min_abs_y_target_velocity # Enforce minimum, ensure it's upwards

                # If Y's magnitude was significantly changed to meet this minimum,
                # X component should be recalculated to maintain the 'old_ball_speed_magnitude'.
                # New_X_mag = sqrt(Speed_mag^2 - New_Y_mag^2)
                if old_ball_speed_magnitude**2 >= self.Ball.Velocity.y**2: # Ensure no sqrt of negative
                    new_x_magnitude_squared = old_ball_speed_magnitude**2 - self.Ball.Velocity.y**2
                    self.Ball.Velocity.x = math.sqrt(max(0, new_x_magnitude_squared)) * \
                                           glm.sign(self.Ball.Velocity.x if self.Ball.Velocity.x != 0 else \
                                                    (new_ball_velocity_x if new_ball_velocity_x !=0 else 1) ) # Keep original X sign
                # else: Ball is now predominantly vertical; overall speed might slightly reduce if it can't be maintained.
            
            self.audio["bounce"].play()
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
            case "pad":
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
        
        speedDuration = 20 * FRAMERATE_REFERENCE
        stickyDuration = 20 * FRAMERATE_REFERENCE
        passthroughDuration = 10 * FRAMERATE_REFERENCE
        confuseDuration = 5 * FRAMERATE_REFERENCE
        chaosDuration = 10 * FRAMERATE_REFERENCE
        padIncreaseDuration = 20 * FRAMERATE_REFERENCE
        
        if ShouldSpawn(35): #75
            self.PowerUps.append(PowerUP("speed",glm.vec3(0.5,0.5,1.0),speedDuration,block.Position,self.TextureManager["powerup_speed"]))
        if ShouldSpawn(65): #75
            self.PowerUps.append(PowerUP("sticky",glm.vec3(1.0,0.5,1.0),stickyDuration,block.Position,self.TextureManager["powerup_sticky"]))
        if ShouldSpawn(55): #75
            self.PowerUps.append(PowerUP("pass-through",glm.vec3(0.5,1.0,0.5),passthroughDuration,block.Position,self.TextureManager["powerup_passthrough"]))
        if ShouldSpawn(35): #75
            self.PowerUps.append(PowerUP("pad",glm.vec3(1.0,0.6,0.4),padIncreaseDuration,block.Position,self.TextureManager["powerup_increase"]))
        if ShouldSpawn(15): #15
            self.PowerUps.append(PowerUP("confuse",glm.vec3(1.0,0.3,0.3),confuseDuration,block.Position,self.TextureManager["powerup_confuse"]))
        if ShouldSpawn(15): #15
            self.PowerUps.append(PowerUP("chaos",glm.vec3(0.9,0.25,0.25),chaosDuration,block.Position,self.TextureManager["powerup_chaos"]))

    def ResizePowerUps(self,width,height):
        global POWERUP_SIZE
        global POWERUP_VELOCITY
        ratio = glm.vec2(width,height) / glm.vec2(WIDTH,HEIGHT)
        POWERUP_SIZE = INITIAL_POWERUP_SIZE * ratio
        PowerUps:list[PowerUP] = []
        powerup:PowerUP
        for powerup in self.PowerUps:
            PowerUps.append(PowerUP(powerup.Type,powerup.Color,powerup.Duration,powerup.Position,powerup.Sprite))
        self.PowerUps = PowerUps
        
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
                                self.Ball.Color = glm.vec3(1.0)
                        case "confuse":
                            if not isOtherPowerUpActive(self.PowerUps,"confuse"):
                                self.effects.Confuse = False
                        case "chaos":
                            if not isOtherPowerUpActive(self.PowerUps,"chaos"):
                                self.effects.Chaos = False
                        case "speed":
                            if not isOtherPowerUpActive(self.PowerUps,"speed"):
                                self.Ball.Velocity = glm.vec2(BALL_VELOCITY)
                        case "pad":
                                if self.Player.Size.x > PLAYER_SIZE.x:
                                    self.Player.Size.x -= 50

        self.PowerUps = [powerup for powerup in self.PowerUps if powerup.Activated or (not powerup.Destroyed)]
        self.textList["powerup"].update_text("\n".join([f"{powerup.Type}:{powerup.Duration:.1f}" for powerup in self.PowerUps if powerup.Activated ]))
        self.textList["powerup"].set_position(self.textList["powerup"].position)


if __name__ == "__main__":
    random.seed(int(time.time()))
    Breakout = Game(WIDTH,HEIGHT,GameState.GAME_MENU)
    Breakout.Init()
    Breakout.Run()