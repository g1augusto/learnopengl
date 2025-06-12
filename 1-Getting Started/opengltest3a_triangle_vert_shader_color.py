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

Excercise:  alter the shaders from the previous chapter to let the vertex
            shader decide the color for the fragment shader.
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
# NOTE: Here we will define with the SAME CONTEXT two different programs that differs only on the color
# of the fragment shader.
# the two programs (shaders) will be attached to the related VAO
prog1 = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;
out vec4 vertexColor;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    vertexColor = vec4(0.5, 0.0, 0.0, 1.0);
}
''',

    fragment_shader='''
#version 330 core
in vec4 vertexColor;
out vec4 FragColor;

void main()
{
    FragColor = vertexColor;
} 
''')

prog2 = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
} 
''')

# vertices

vertices = [
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
     0.0,  0.5, 0.0,
]

# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '12f': This is the format string. It specifies that we want to pack eight floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
vertices_binaryformat = struct.pack("9f",*vertices)

# Define VBO (Vertex Buffer Object) containing vertex data
vbo = context.buffer(vertices_binaryformat)

# EBO indices for "indexed drawing"
# indexed drawing means identifying vertices in an unique index and build shapes by referring to these indexes
indices1 = [0,1,2 ]


# we use struct.pack to convert the indices list into binary format
# 3I stands for 3 unsigned int (https://docs.python.org/3.7/library/struct.html#format-characters)
indices1_binaryformat = struct.pack("3I",*indices1)

# Define EBO containing idices
# indices will be used to build the two triangles pointing at the index of the vertex list to build 
# a sequence that will compose the two triangles
ebo1 = context.buffer(indices1_binaryformat)

# VBO parameters to be passed to the VAO
# This is what in modernGL is defined as "multiple buffers for all input variables"
# meaning that each VBO buffer is described as a tuple in a list
# elements of the tuple describes
# 1) Vertex Buffer Object in input
# 2) type of input parameters (3f in this case corresponds to a 3vec input) defined in shaders
# 3) name of the input parameter in the related shader (aPos in this case)
# NOTE: we will need two different VAOs but we can use a single VBO to pass all vertex data
#       even if it may be suboptimal because each VAO uses only a subset of all vertex data sent to both
vbo_parameters = [
    (vbo,"3f","aPos")
]


# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
# NOTE: we created two VAOs since each VAO can only use a single fragment shader 
#       (Triangle color is set on the fragment shader)
vao1 = context.vertex_array(prog1,vbo_parameters,ebo1)
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
    # NOTE: Here we are rendering both VAOs in sequence (vao1 orange triangle, vao2 yellow triangle)
    vao1.render()
    pygame.display.flip()
    context.clear(0.0,0.0,0.0) # clears the framebuffer (Necessary and also best practice)
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