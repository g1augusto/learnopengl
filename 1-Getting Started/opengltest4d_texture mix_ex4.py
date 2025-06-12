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

Excercise:  Use a uniform variable as the mix function's third parameter
            to vary the amount the two textures are visible.
            Use the up and down arrow keys to change how much the container
            or the smiley face is visible
            NOTE: F2 to increase the mix uniform variable and F3 to decrease it
'''

import struct
import sys
import pygame
import moderngl

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

# Define Vertex Shader and Fragment Shader in ModernGL (GLSL language)
# ModernGL abstracts vertex and fragment shader as specific parameter of the context program method
prog = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoord;

out vec3 ourColor;
out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;
  
in vec3 ourColor;
in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float mix;

void main()
{
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, vec2(1.0 - TexCoord.x, TexCoord.y)), mix);
}
''')

# vertices
vertices = [
    # positions          # texture coords
     0.5,  0.5, 0.0,     1.0, 1.0,   # top right
     0.5, -0.5, 0.0,     1.0, 0.0,   # bottom right
    -0.5, -0.5, 0.0,     0.0, 0.0,   # bottom left
    -0.5,  0.5, 0.0,     0.0, 1.0    # top left
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
# NOTE: Stride is inferred by the position of the parameter declarations
vbo_parameters = [
    (vbo,"3f 2f","aPos","aTexCoord")
]

# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
vao = context.vertex_array(prog,vbo_parameters,ebo)

while True:
    # Vertex Array Object Rendering
    # in ModernGL, rendering is almost always performed using the VertexArray.render()
    #   Here's why:
    #       VAO Encapsulation:
    #           VAOs encapsulate the state related to vertex attributes (how vertex data is laid out and interpreted) and the associated vertex buffer bindings.
    #       This means that when you call vao.render(), ModernGL sets up all the necessary OpenGL state based on the VAO's configuration.
    #   Efficiency:
    #       VAOs reduce the number of OpenGL state changes required during rendering, which significantly improves performance.
    #       By storing the vertex attribute configuration in the VAO, you avoid having to repeatedly set up the same state every frame.
    #   Modern OpenGL Practice:
    #       VAOs are a core feature of modern OpenGL, and ModernGL is designed to promote modern OpenGL practices.
    vao.render()
    pygame.display.flip()
    context.clear(0.0,0.0,0.0) # clears the framebuffer (Necessary her and also best practice)
    for event in pygame.event.get():
        if  event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_F10:
                context.wireframe = not context.wireframe
            elif event.key == pygame.K_F2:
                prog["mix"].value += 0.1
            elif event.key == pygame.K_F3:
                prog["mix"].value -= 0.1