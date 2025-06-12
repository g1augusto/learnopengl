import pygame
import moderngl
import glm
from enum import Enum # to define movement enum class
import sys
import struct
import ctypes

# GLOBAL CONSTANTS
FRAMERATE = 60                          # Framerate to use in clock tick method
FRAMERATE_REFERENCE = 60                # Reference from Framerate Target to be achieved
WIDTH = 800
HEIGHT = 600
VSYNC = False

# LearnOpenGL uses a resource manager c++ static class (a class that needs no instance and is globally accessible)
# here we don't need all of that since python and libraries offers good abstractions
# but we use a class for loading and storing moderngl.Texture objects loaded and converted from image files
# with modernGL would not make sense to use a library or a class with static methods since Texture objects are
# tied to a specific moderngl.Context which is instantiated within the Game class
# This class is part of the Game class and a TextureManager is instantiated at the game init
class Textures():
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
    def __init__(self,shader:moderngl.Program,context:moderngl.Context):
        self.shader:moderngl.Program = shader
        self.context = context
        self.vao = None
        self.initRenderData()

    
    def initRenderData(self):
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


class Game:
    def __init__(self,width:int,height:int):
        self.State:GameState
        self.Width:int = width
        self.Height:int = height
        self.Clock = pygame.time.Clock()
        self.screen_display = None
        self.context = None
        self.program = None
        self.renderer = None
        self.TextureManager = None

    
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
        self.program = self.context.program(
            vertex_shader='''
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
''',
            fragment_shader='''
// The fragment shader is relatively straightforward as well.
// We take a texture and a color vector that both affect the final color of the fragment.
// By having a uniform color vector, we can easily change the color of sprites from the game-code
#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D image;
uniform vec3 spriteColor;

void main()
{    
    color = vec4(spriteColor, 1.0) * texture(image, TexCoords);
}  
'''
        )

        projection = glm.ortho(0.0,self.Width,self.Height,0.0,-1.0,1.0)
        self.program["projection"].write(matrix_bytes(projection))
        self.renderer = SpriteRenderer(self.program,self.context)
        self.TextureManager = Textures(self.context)
        self.TextureManager["face"] = "./assets/awesomeface.png"

    def ProcessInput(self,dt:float):
        ...
    
    def Update(self,dt:float):
        ...
    
    def Render(self):
        self.renderer.DrawSprite(self.TextureManager["face"],glm.vec2(200.0,200.0),glm.vec2(300.0,400.0),45.0,glm.vec3(0.0,1.0,0.0))

    def Run(self):
        while True:
            NormalizedDeltaTime = self.Clock.tick(FRAMERATE) * 0.001 * FRAMERATE_REFERENCE # set the FPS of the game to not exceed the framerate
            # Avoid excessive cumulative increase of Delta Time due to moving the windows around or delay in game processing
            if NormalizedDeltaTime > 15:
                NormalizedDeltaTime = 15.0
            for event in pygame.event.get():
                if  event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.Render() # <-- Rendering logic
            pygame.display.flip()
            self.context.clear() # clears the framebuffer (Necessary and also best practice) 


if __name__ == "__main__":
    Breakout = Game(WIDTH,HEIGHT)
    Breakout.Init()
    Breakout.Run()