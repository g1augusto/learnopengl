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

Excercise:  Load and map a texture to a rectangle (made of two triangles)
'''

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

# Define Vertex Shader and Fragment Shader in ModernGL (GLSL language)
# ModernGL abstracts vertex and fragment shader as specific parameter of the context program method
# Here we defined a uniform array called offsets that contain a total of 100 offset vectors. 
# Within the vertex shader we retrieve an offset vector for each instance by indexing 
# the offsets array using gl_InstanceID. If we now were to draw 100 quads with instanced 
# drawing we'd get 100 quads located at different positions.
prog = context.program(
    vertex_shader='''
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aOffset;


out vec3 fColor;


void main()
{
    gl_Position = vec4(aPos + aOffset, 0.0, 1.0);
    fColor = aColor;
}
''',

    fragment_shader='''
#version 330 core
out vec4 FragColor;
  
in vec3 fColor;

void main()
{
    FragColor = vec4(fColor, 1.0);
}
''')

# vertices
quadVertices = [
    # positions      colors
    -0.05,  0.05,  1.0, 0.0, 0.0,
     0.05, -0.05,  0.0, 1.0, 0.0,
    -0.05, -0.05,  0.0, 0.0, 1.0,

    -0.05,  0.05,  1.0, 0.0, 0.0,
     0.05, -0.05,  0.0, 1.0, 0.0,   
     0.05,  0.05,  0.0, 1.0, 1.0		    		
]

# uses Python's struct module to pack the list of floating-point numbers into a byte string
# '32f': This is the format string. It specifies that we want to pack 32 floating-point numbers (f for float)
# The * operator unpacks the vertices list, passing each element as a separate argument to struct.pack
vertices_binaryformat = struct.pack(f"{len(quadVertices)}f",*quadVertices)

# Define VBO (Vertex Buffer Object) containing vertex data
vbo = context.buffer(vertices_binaryformat)





# Calculate positions to use for placing the quads in instanced rendering
offset = 0.1
translations:list[glm.vec2] = [glm.vec2(x / 10.0 + offset,y / 10.0 + offset) for y in range(-10,10,2) for x in range(-10,10,2)]
instances = len(translations)

# prepare offset data to be passed to a specific instanced VBO

# Flatten the list of vectors into a generator of float components
translations_flattened = (component for vec in translations for component in vec)
translation_binaryformat = struct.pack(f"{instances * 2}f", *translations_flattened)


# Define VBO (Vertex Buffer Object) containing vertex data for INSTANCE DATA
vbo_instances = context.buffer(translation_binaryformat)


# VBO parameters to be passed to the VAO
# This is what in modernGL is defined as "multiple buffers for all input variables"
# meaning that each VBO buffer is described as a tuple in a list
# elements of the tuple describes
# 1) Vertex Buffer Object in input
# 2) type of input parameters (3f in this case corresponds to a 3vec input) defined in shaders
# 3) name of the input parameter in the related shader (aPos in this case)
#
# NOTE: we are adding also INSTANCE VBO DATA (marked with /i)
vbo_parameters = [
    (vbo,"2f 3f","aPos","aColor"),
    (vbo_instances,"2f /i","aOffset") # this is where the instance data is marked
]

# define VAO (Vertex Array Object)
# essentially acts as a container that stores the state of vertex attributes. This includes:
#    Which VBOs (Vertex Buffer Objects) are associated with which vertex attributes.
#    The format of the vertex attributes (e.g., data type, number of components).
#    Whether a particular vertex attribute is enabled or disabled.
vao = context.vertex_array(prog,vbo_parameters)

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

    # For INSTANCED RENDERING is necessary to bass to the VAO render method the number of instances to render
    vao.render(instances=instances)
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