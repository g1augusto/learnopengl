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

Excercise:  More 3D
            So far we've been working with a 2D plane, even in 3D space,
            so let's take the adventurous route and extend our 2D plane to a 3D cube. 
'''

import ctypes # for pyglm bytes conversion
import struct
import sys
import pygame
import moderngl
import glm

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



while True:
    ## Model Matrices 
    # Model Matrix
    model = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    model = glm.rotate(model,pygame.time.get_ticks() / 1000.0 * glm.radians(50.0),glm.vec3(0.5,1.0,0.0)) # For fun, we'll let the cube rotate over time
    # View Matrix
    view = glm.mat4(1.0) # identity matrix (1.0 at the diagonal)
    view = glm.translate(view,glm.vec3(0.0,0.0,-3.0)) # We want to move slightly backwards in the scene so the object becomes visible
    # Projection Matrix
    projection = glm.perspective(glm.radians(45.0),800.0 / 600.0, 0.1, 100.0)

    prog["model"].write(matrix_bytes(model))
    prog["view"].write(matrix_bytes(view))
    prog["projection"].write(matrix_bytes(projection))
    vao.render()

    pygame.display.flip()
    context.clear(color=(0.0, 0.0, 0.0), depth=1.0) # clears the framebuffer (Necessary her and also best practice) AND clears the z-buffer setting it to the max
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