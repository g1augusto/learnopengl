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
PLAYER_SIZE = glm.vec2(100.0,20)
PLAYER_VELOCITY = 10.0


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
        self.Size:glm.vec2 = size
        self.Velocity:glm.vec2 = velocity
        self.Color:glm.vec3 = color
        self.Rotation:float = rotation
        self.IsSolid:bool = issolid
        self.Destroyed:bool = destroyed
        self.Sprite:moderngl.Context.texture = sprite
    
    def Draw(self,renderer:SpriteRenderer):
        renderer.DrawSprite(self.Sprite,self.Position,self.Size,self.Rotation,self.Color)


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
    def __init__(self,sprite:moderngl.Context.texture,radius:float = 12.5,stuck:bool = True,position:glm.vec2 = glm.vec2(0.0,0.0), velocity:glm.vec2 = glm.vec2(0.0,0.0)):
        super().__init__(sprite=sprite,position=position,velocity=velocity)
        self.Radius:float = radius
        self.Stuck:bool = stuck

    def Move(self,dt:float,window_width:int):
        if not self.Stuck:
            self.Position += self.Velocity * dt
            if self.Position < 0.0 :
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
        self.TextureManager = None
        self.levels:list[GameLevel] = []
        self.level:int = 0
        self.Player = None

    
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
        )

        projection = glm.ortho(0.0,self.Width,self.Height,0.0,-1.0,1.0)
        self.program["projection"].write(matrix_bytes(projection))
        self.renderer = SpriteRenderer(self.program,self.context)
        self.TextureManager = Textures(self.context)
        # Texture loading
        self.TextureManager["face"] = "./assets/awesomeface.png"
        self.TextureManager["background"] = "./assets/background.jpg"
        self.TextureManager["block"] = "./assets/block.png"
        self.TextureManager["block_solid"] = "./assets/block_solid.png"
        self.TextureManager["paddle"] = "./assets/paddle.png"
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

    def ProcessInput(self,dt:float,keys):
        if self.State is GameState.GAME_ACTIVE:
            velocity = PLAYER_VELOCITY * dt
            if keys[pygame.K_LEFT]:
                if self.Player.Position.x > 0.0:
                    self.Player.Position.x -= velocity
            if keys[pygame.K_RIGHT]:
                if self.Player.Position.x <= self.Width - self.Player.Size.x:
                    self.Player.Position.x += velocity
    
    def Update(self,dt:float):
        ...
    
    def Render(self):
        if self.State is GameState.GAME_ACTIVE:
            # Draw background
            self.renderer.DrawSprite(self.TextureManager["background"],glm.vec2(0.0,0.0),glm.vec2(self.Width,self.Height),0)
            # Draw level
            self.levels[self.level].Draw(self.renderer)
            # Draw Player
            self.Player.Draw(self.renderer)

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
            self.Render() # <-- Rendering logic
            pygame.display.flip()
            self.context.clear() # clears the framebuffer (Necessary and also best practice) 


if __name__ == "__main__":
    Breakout = Game(WIDTH,HEIGHT)
    Breakout.Init()
    Breakout.Run()