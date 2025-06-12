'''
OpenGL in Python with ModernGL
Based on learngopengl.com
https://learnopengl.com/Getting-started/Hello-Triangle
Vertex Input
Vertex Buffer Object (VBO)
Vertex Shader
Fragment Shader
Vertex Array Object (VAO)

Excercise:
Try to draw 2 triangles next to each other using glDrawArrays by adding more vertices to your data
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
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
} 
''')

# vertices
vertices = [
    -1.0, -0.5, 0.0,
     0.0, -0.5, 0.0,
    -0.5,  0.5, 0.0,
     0.0, -0.5, 0.0,
     1.0, -0.5, 0.0,
     0.5, 0.5, 0.0,
]

# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '12f': This is the format string. It specifies that we want to pack eight floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
vertices_binaryformat = struct.pack("18f",*vertices)

# Define VBO (Vertex Buffer Object) containing vertex data
vbo = context.buffer(vertices_binaryformat)

# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
vao = context.vertex_array(prog,vbo,"aPos")

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
    for event in pygame.event.get():
        if  event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()