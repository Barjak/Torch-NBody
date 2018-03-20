#!/usr/bin/python3
import glfw
# import  OpenGL, math
import numpy as np
# from pygame.locals import *
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
# from OpenGL.WGL import *
# from OpenGL.GLU import *
# from PIL import Image
import ctypes
import time
C_NULL = ctypes.c_void_p(0)
# import pdb;
# pdb.set_trace()

class Shader(object):
    def __init__(self):
        vertex_shader = shaders.compileShader("""
            #version 330
            uniform mat4 transform;
            uniform vec3 obj_translate;
            uniform vec3 scale;
            in vec3 pos;
            out vec3 pos2;
            void main()
            {
                mat4 obj_mat = mat4(1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0,
                                    obj_translate[0],obj_translate[1],obj_translate[2],1);
                mat4 scale_mat = mat4(scale[0],0,0,0,
                                      0,scale[1],0,0,
                                      0,0,scale[2],0,
                                      0,0,0,1);

                vec4 position = transform * obj_mat * scale_mat * vec4(pos, 1.0f);
                gl_Position = position;//transform * vec4(pos, 1.0f);
                pos2 = vec3(0.0, 0.0, position[2]);
            }
            """, GL_VERTEX_SHADER)
        # print(glGetShaderiv(vertex_shader, GL_COMPILE_STATUS))
        # print(glGetShaderInfoLog(vertex_shader))

        fragment_shader = shaders.compileShader( """
            #version 330
            in vec3 pos2;
            out vec4 color;
            void main()
            {
                float l = (pos2[2] - 2)/ 10.0;
                color = vec4(l, l, l, 1.0f );
            }
            """ , GL_FRAGMENT_SHADER)
        # print(glGetShaderiv(fragment_shader, GL_COMPILE_STATUS))
        # print(glGetShaderInfoLog(fragment_shader))

        self._program = shaders.compileProgram(vertex_shader, fragment_shader);
        glUseProgram(self._program)
        self._transform = glGetUniformLocation(self._program, "transform")
        self._scale = glGetUniformLocation(self._program, "scale")
        self._obj_translate = glGetUniformLocation(self._program, "obj_translate")

    @property
    def transform(self):
        return self._transform
    @transform.setter
    def transform(self, matrix):
        glUniformMatrix4fv(self._transform, 1, GL_TRUE, matrix)
    @property
    def scale(self):
        return self._transform
    @scale.setter
    def scale(self, factor):
        glUniform3fv(self._scale, 1, np.array([factor,factor,factor], dtype=np.float32))
    @property
    def obj_translate(self):
        return self._obj_translate
    @obj_translate.setter
    def obj_translate(self, matrix):
        glUniform3fv(self._obj_translate, 1, matrix)

    def attrib(self, name):
        return glGetAttribLocation(self._program, name)

def sphere():
    vertices = []
    indices = []
    with open("untitled2.obj", "r") as f:
        for line in iter(f.readline, ""):
            words = line.split()
            first_word = words[0]
            words = words[1:]
            if first_word == "v":
                vertices += [float(x) for x in words]
            elif first_word == "f":
                face = [int(x.split("/")[0])-1 for x in words]
                ft = [(face[0], t[0], t[1]) for t in zip(face[1:], face[2:])]
                indices += list(sum(ft, ()))
    return (np.array(vertices, dtype=np.float32),
            np.array(indices, dtype=np.uint32))

def view_frustum(fov, aspect, zNear, zFar):
    f = np.tan((np.pi / 2.) - (np.radians(fov) / 2.))
    return np.array([[f/aspect, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, (zFar + zNear) / (zNear - zFar), (2. * zFar * zNear) / (zNear - zFar)],
                    [0, 0, -1, 0]], dtype=np.float32)
    # return np.array([[np.arctan(np.radians(60) / 2.), 0, 0, 0],
    #                 [0, np.arctan(np.radians(80) / 2.), 0, 0],
    #                 [0, 0, (zFar + zNear) / (zNear - zFar), (2. * zFar * zNear) / (zNear - zFar)],
    #                 [0, 0, -1, 0]], dtype=np.float32)
def translation(xyz):
    return np.array([[1,0,0,xyz[0]],
                     [0,1,0,xyz[1]],
                     [0,0,1,xyz[2]],
                     [0,0,0,1]], dtype=np.float32);
def scale(xyz):
    return np.array([[xyz[0],0,0,0],
                     [0,xyz[1],0,0],
                     [0,0,xyz[2],0],
                     [0,0,0,     1]], dtype=np.float32);

class NBody(object):
    def __init__(self, n):
        self.position = torch.randn(20,3)
        self.velocity = torch.randn(20,3)




class Camera(object):
    def __init__(self, fov, x, y, z):
        self._fov = fov
        self._xyz = np.array([x,y,z])
        self._frustum = view_frustum(fov, 1, 0.001, 100.)
        self._translate = translation(self.xyz)

    def set_xyz(self, x, y, z):
        self._xyz = np.array([x,y,z])
        self._translate = translation(self.xyz)
        if hasattr(self, '_matrix'):
            del self._matrix
    def set_fov(self, fov):
        self._fov = fov
        self._frustum = view_frustum(fov, 1, 0.001, 100.)
        if hasattr(self, '_matrix'):
            del self._matrix
    @property
    def matrix(self):
        if hasattr(self, '_matrix'):
            return self._matrix
        else:
            self._matrix = np.matmul(self._frustum, self._translate)
            return self._matrix

    @property
    def xyz(self):
        return self._xyz

    @property
    def x(self):
        return self.xyz[0]
    @property
    def y(self):
        return self.xyz[0]
    @property
    def z(self):
        return self.xyz[0]

def main():
    # print(view_frustum(90., 3./4., 0.1, 100.))
        # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(1600, 1600, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    # pygame.init()
    # size = (800,600)
    # pygame.display.set_mode(size, pygame.OPENGL|pygame.DOUBLEBUF)

    shade = Shader()

    (vertices, indices) = sphere()
    # print(vertices)
    # print(indices)

    n = len(vertices) // 3
    m = len(indices) // 3

    cam = Camera(90, 0., 0., 0.)

    (v, v_i) = glGenBuffers(2)
    glBindBuffer(GL_ARRAY_BUFFER, v)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, v_i)
    glBufferData(GL_ARRAY_BUFFER,         4 * 3 * n, vertices, GL_STATIC_DRAW)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * 3 * m, indices, GL_STATIC_DRAW)

    transform = glGenBuffers(1)
    glBindBuffer(GL_UNIFORM_BUFFER, transform)
    glBufferData(GL_UNIFORM_BUFFER, 4 * len(cam.matrix), cam.matrix, GL_DYNAMIC_DRAW)

    glVertexAttribPointer(shade.attrib("pos"), 3, GL_FLOAT, GL_FALSE, 0, C_NULL)
    glEnableVertexAttribArray(shade.attrib("pos"))

    glClearColor(1.0,1.0,1.0,1.0)

    # dar = np.reshape(vertices, (n, 3))
    # dar = np.hstack((dar, np.ones(n).reshape((n,1))))
    xyz = 2 * np.random.normal(scale=1., size=(2000,3))
    xyz = np.array(xyz, dtype=np.float32)
    for ijk in xyz:
        print(ijk)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        time.sleep(1. / 60.)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for ijk in xyz:
            shade.transform = cam.matrix
            shade.scale = 0.05
            shade.obj_translate = np.array([ijk[0],ijk[1],-6+ijk[2] + 3 * np.sin(time.time())], dtype=np.float32)
            glDrawElements(GL_TRIANGLES, 3 * m, GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)
    glfw.terminate()

if __name__ == '__main__':
    main()
