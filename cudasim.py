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
            #version 440
            uniform mat4 view;
            layout(std430, binding = 0) buffer obj_data {
                double translate[];
            } obj;
            in float scale;
            in vec3 pos;
            out float distance;
            void main()
            {
                mat4 obj_mat = mat4(1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0,
                                    obj.translate[(3*gl_InstanceID)],obj.translate[(3*gl_InstanceID)+1],obj.translate[(3*gl_InstanceID)+2],1);
                mat4 scale_mat = mat4(scale,0,0,0,
                                      0,scale,0,0,
                                      0,0,scale,0,
                                      0,0,0,1);

                vec4 position = view * obj_mat * scale_mat * vec4(pos, 1.0f);
                gl_Position = position;
                distance = position[2];
            }
            """, GL_VERTEX_SHADER)
        print(glGetShaderiv(vertex_shader, GL_COMPILE_STATUS))
        print(glGetShaderInfoLog(vertex_shader))

        fragment_shader = shaders.compileShader( """
            #version 440
            in float distance;
            out vec4 color;
            void main()
            {
                float l = 1.0 - log2(distance)/6.;
                color = vec4(l, l, l, 1.0f );
            }
            """ , GL_FRAGMENT_SHADER)
        print(glGetShaderiv(fragment_shader, GL_COMPILE_STATUS))
        print(glGetShaderInfoLog(fragment_shader))

        self._program = shaders.compileProgram(vertex_shader, fragment_shader);
        glUseProgram(self._program)
        self._view = glGetUniformLocation(self._program, "view")
        self._scale = glGetAttribLocation(self._program, "scale")
        self._obj_translate = glGetAttribLocation(self._program, "obj_translate")

    @property
    def view(self):
        return self._view
    @view.setter
    def view(self, matrix):
        glUniformMatrix4fv(self._view, 1, GL_TRUE, matrix)
    @property
    def scale(self):
        return self._view
    @scale.setter
    def scale(self, factor):
        glVertexAttrib1f(self._scale, factor)
    @property
    def obj_translate(self):
        return self._obj_translate
    @obj_translate.setter
    def obj_translate(self, vec):
        glVertexAttrib3fv(self._obj_translate, vec)

    def attrib(self, name):
        return glGetAttribLocation(self._program, name)

class Model(object):
    def __init__(self, name, type=np.float32):
        self._vertices = []
        self._vertex_normals = []
        self._indices = []
        self._is_loaded = False
        with open(name, "r") as f:
            for line in iter(f.readline, ""):
                words = line.split()
                first_word = words[0]
                words = words[1:]
                if first_word == "v":
                    self._vertices += [[float(x) for x in words]]
                elif first_word == "f":
                    face = [int(x.split("/")[0])-1 for x in words]
                    ft = [(face[0], t[0], t[1]) for t in zip(face[1:], face[2:])]
                    self._indices += ft
        self._vertices = np.array(np.reshape(self._vertices, -1), dtype=np.float32)
        self._indices = np.array(np.reshape(self._indices, -1), dtype=np.uint32)
        self.load()

    @property
    def n_indices(self):
        return len(self._indices) // 3

    @property
    def is_loaded(self):
        return self._is_loaded

    def load(self):
        n = len(self._vertices) // 3
        m = len(self._indices) // 3
        self.vbuffer = glGenBuffers(1)
        self.ibuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbuffer)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibuffer)
        glBufferData(GL_ARRAY_BUFFER,         4 * 3 * n, self._vertices, GL_STATIC_DRAW)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * 3 * m, self._indices, GL_STATIC_DRAW)

    def use(self, attrib):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbuffer)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibuffer)
        glVertexAttribPointer(attrib, 3, GL_FLOAT, GL_FALSE, 0, C_NULL)
        glEnableVertexAttribArray(attrib)

    def unload(self):
        glDeleteBuffers(2, (self.vbuffer, self.ibuffer))



def view_frustum(fov, aspect, zNear, zFar):
    f = np.tan((np.pi / 2.) - (np.radians(fov) / 2.))
    return np.array([[f/aspect, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, (zFar + zNear) / (zNear - zFar), (2. * zFar * zNear) / (zNear - zFar)],
                    [0, 0, -1, 0]], dtype=np.float32)

def translate(xyz):
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
    def __init__(self, n, e=0.001, G=0.001):
        self.n = n
        self.e2 = e ** 2
        self.G = G
        self.x = np.array(np.random.normal(scale=4., size=(n,3)), dtype=np.double)
        self.v = np.array(np.random.normal(scale=0.01, size=(n,3)), dtype=np.double)#np.zeros((n,3))
        self.m = np.array(np.ones(n), dtype=np.double)
    def step(self, dt):
        # print("A")
        dx = self.x[:,np.newaxis,:] - self.x[np.newaxis,:,:]
        # dv = self.v[:,np.newaxis,:] - self.v[np.newaxis,:,:]
        # print(dx.shape)
        # print("B")

        norm2 = np.sum(dx ** 2,axis=2)
        # print(norm2.shape)
        # print("C")

        dampened_norm = np.power(norm2 + self.e2, 3/2)
        # print(denominator.shape)
        # print("D")
        a = -self.G * self.m[:,np.newaxis] * np.sum(dx / dampened_norm[:,:,np.newaxis], axis=1)
        # da = dampened_norm
        # print(a.shape)
        # print("F")
        self.x = self.x + dt*self.v + (dt*dt)*a*0.5
        # print(self.x.shape)
        # print("G")
        self.v = self.v + dt*a
        # print(self.v.shape)


class Camera(object):
    def __init__(self, fov, x, y, z):
        self._fov = fov
        self._xyz = np.array([x,y,z])
        self._frustum = view_frustum(fov, 1, 0.001, 1000.)
        self._translate = translate(self.xyz)

    def set_xyz(self, x, y, z):
        self._xyz = np.array([x,y,z])
        self._translate = translate(self.xyz)
        if hasattr(self, '_matrix'):
            del self._matrix
    def set_fov(self, fov):
        self._fov = fov
        self._frustum = view_frustum(fov, 1, 0.01, 1000.)
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
    n_particles = 3
    # for i in range(100000):
        # print(i)
        # nbody.step(0.001)
    # quit(0)
    if not glfw.init():
        return

    window = glfw.create_window(2000, 2000, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,1.0)

    # Generate random
    # xyz = 2 * np.random.normal(scale=1.2, size=(200000,3))

    cam = Camera(90, 0., 0., -20)
    shader = Shader()
    shader.view = cam.matrix

    model = Model("sphere.obj")
    model.use(shader.attrib("pos"))
    shader.scale = 0.2
    m = model.n_indices

    nbody = NBody(n_particles, 0.5, 0.5)
    SSB = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSB)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, SSB)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        time.sleep(1. / 60.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        nbody.step(0.1)
        # t = time.time()
        # time_offset = np.array([0.01*np.cos(t*20), 0.01*np.sin(t*20), -5 + 3 * np.sin(t * 0.2)])
        time_offset = np.array([0,0,0])
        temp = np.array(np.reshape(nbody.x, -1), dtype=np.double)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 8 * len(temp), temp, GL_DYNAMIC_DRAW)

        glDrawElementsInstanced(GL_TRIANGLES, 3 * m, GL_UNSIGNED_INT, None, n_particles)

        glfw.swap_buffers(window)
    glfw.terminate()

if __name__ == '__main__':
    main()
