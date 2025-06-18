
# LearnOpenGL in Python

This repository is a Python port of the C++ code from the highly regarded [LearnOpenGL](https://learnopengl.com) tutorial series by Joey de Vries.

The primary purpose of this repository is to provide a Python-based alternative for developers who are following the LearnOpenGL tutorials.

At the moment is not 100% complete but I plan to do so in some time.

## Structure

Each folder in this repository corresponds to a specific section in the LearnOpenGL tutorials. This structure allows you to easily find the Python code relevant to the chapter you are studying.

## Getting Started

To run the examples in this repository, you will need to have Python and the required libraries installed.

### Prerequisites

* Python 3.12.x
* moderngl
* pygame
* PyGLM
* Pillow (alternative for texture loading)


## References

The following are a collection of useful links and resources for learning computer graphics and OpenGL.

### ModernGL

For those looking for a higher-level Python wrapper over OpenGL, **ModernGL** is an excellent library that simplifies many of the verbose commands associated with PyOpenGL. It is designed to be easier to learn while still providing high performance.

* **ModernGL Documentation:** [https://moderngl.readthedocs.io/](https://moderngl.readthedocs.io/)


### Pygame

At its core, Pygame is a Python wrapper for the Simple DirectMedia Layer (SDL) library. SDL is a low-level, cross-platform development library written in C that provides access to audio, keyboard, mouse, joystick, and graphics hardware.<br>
Pygame is used to handle screen creation, OpenGL scren binding, input and audio

* **Pygame Documentation:** [https://www.pygame.org/docs/](https://www.pygame.org/docs/)
---

## Acknowledgements

Thank you Joey for your amazing material, I hope you can deliver more in the future.

## Licenses

This project is licensed under the [MIT License](LICENSE).

This project includes third-party software, each with its own license. The full license texts can be found in the [LICENSES](./LICENSES/) directory.

* **Open Asset Import Library (assimp)** - [3-clause BSD License](./LICENSES/LICENSE-assimp.md)

This project uses third-party assets. Please see the [ATTRIBUTION.md](./LICENSES/ATTRIBUTION.md) file for details.


