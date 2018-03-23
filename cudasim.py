#!/usr/bin/python3
import sys
import torch
import glfw
import gc
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
def rotate(d):
    return np.array([[np.cos(d),0,-np.sin(d),0],
                     [0,1,0,0],
                     [np.sin(d),0,-np.cos(d),0],
                     [0,0,0,1]], dtype=np.float32);

class Shader(object):
    def __init__(self):
        vertex_shader = shaders.compileShader("""
            #version 440
            uniform mat4 view;
            layout(std430, binding = 0) buffer obj_data1 {
                double obj_translate[];
            };
            layout(std430, binding = 1) buffer obj_data2 {
                double obj_scale[];
            };
            in vec3 position;
            out float distance;

            void main()
            {

                vec3 _t = vec3(obj_translate[(3*gl_InstanceID)],
                               obj_translate[(3*gl_InstanceID)+1],
                               obj_translate[(3*gl_InstanceID)+2]);
                mat4 obj_mat = mat4(1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0,
                                    _t[0],_t[1],_t[2],1);
                mat4 scale_mat = mat4(obj_scale[gl_InstanceID],0,0,0,
                                      0,obj_scale[gl_InstanceID],0,0,
                                      0,0,obj_scale[gl_InstanceID],0,
                                      0,0,0,1);

                vec4 p = view * obj_mat * scale_mat * vec4(position, 1.0f);
                gl_Position = p;
                distance = p[2];
            }
            """, GL_VERTEX_SHADER)
        # print(glGetShaderiv(vertex_shader, GL_COMPILE_STATUS))
        # print(glGetShaderInfoLog(vertex_shader))

        fragment_shader = shaders.compileShader( """
            #version 440
            in float distance;
            out vec4 color;

            void main()
            {
                float l = 1. - log2(distance) / 5.5;
                color = vec4(l, l, l, 1.0f );
            }
            """ , GL_FRAGMENT_SHADER)
        # print(glGetShaderiv(fragment_shader, GL_COMPILE_STATUS))
        # print(glGetShaderInfoLog(fragment_shader))

        self._program = shaders.compileProgram(vertex_shader, fragment_shader);
        glUseProgram(self._program)
        self._view = glGetUniformLocation(self._program, "view")
        self._position = glGetAttribLocation(self._program, "position")
        # assert(self._position != 0)

    @property
    def view(self):
        return self._view
    @view.setter
    def view(self, matrix):
        glUniformMatrix4fv(self._view, 1, GL_TRUE, matrix)
    @property
    def position(self):
        return self._position
    @property
    def obj_translate(self):
        return 0
    @property
    def obj_scale(self):
        return 1

class Model(object):
    def __init__(self, name, center=True, type=np.float32):
        self.name = name
        self.vbuffer = -1
        self.ibuffer = -1
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
        if center:
            self._vertices -= np.average(self._vertices, axis=0)
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
        if self.vbuffer != -1 or self.ibuffer == -1:
            raise RuntimeError("Data is already loaded.")
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
        self.vbuffer = -1
        self.ibuffer = -1


class NBody(object):
    def __init__(self, n, damping=0.001, G=0.001, spread=4., mass_pareto=2.0, velocity_spread=0.0):
        self.n = n
        self.e2 = damping ** 2
        self.G = G
        self.x = np.array(np.random.normal(scale=spread, size=(n,3)), dtype=np.double)
        self.m = np.array(2.0*np.random.pareto(mass_pareto, size=(n,)), dtype=np.double)
        self.v = np.array(np.random.normal(scale=velocity_spread, size=(n,3)), dtype=np.double)#np.zeros((n,3))
        self.da = np.zeros(shape=(n,3), dtype=np.double)
    def step(self, dt):
        '''Explicitly calculate a and da/dt; guess at da/dtt.'''
        dx = self.x[:,np.newaxis,:] - self.x[np.newaxis,:,:] #(n,n,3)
        dv = self.v[:,np.newaxis,:] - self.v[np.newaxis,:,:] #(n,n,3)
        norm2 = np.sum(dx ** 2,axis=2) #(n,n)
        dampened_norm = norm2 + self.e2 #(n,n)
        dampened_norm_32 = np.power(dampened_norm, 3/2)[:,:,np.newaxis] #(n,n,1)
        dampened_norm_52 = np.power(dampened_norm, 5/2) #(n,n)
        a = -self.G * np.sum(self.m[np.newaxis,:,np.newaxis] * (dx / dampened_norm_32), axis=1)
        da = -self.G * np.sum(self.m[np.newaxis,:,np.newaxis] * (dv / dampened_norm_32
                                                     - (3. * (np.einsum("ijk,ijk->ij", dx,dv) / dampened_norm_52)[:,:,np.newaxis] * dx)), axis = 1)
        dda = da - self.da
        self.da = da
        self.x += dt*self.v + (dt**2)*a/2. + (dt**3)*da/6. + (dt**4)*dda/24.
        self.v += dt*a + (dt**2)*da/2. + (dt**3)*dda/6.

class torch_NBody(object):
    def __init__(self, n, damping=0.001, G=0.001, spread=4., mass_pareto=2.0, velocity_spread=0.0):
        if n == 0:
            return
        self.n = n
        self.e2 = damping ** 2
        self.G = G
        self.x = torch.DoubleTensor(n,3).normal_(std=spread).cuda()
        self.m = torch.DoubleTensor(np.random.pareto(mass_pareto, size=(n,))).unsqueeze(0).unsqueeze_(2).cuda()
        if velocity_spread <= 0.0:
            self.v = torch.DoubleTensor(n,3).zero_().cuda()
        else:
            self.v = torch.DoubleTensor(n,3).normal_(std=velocity_spread).cuda()
        self.da = torch.DoubleTensor(n,3).zero_().cuda()

    def copy(self, other):
        self.n = other.n
        self.e2 = other.e2
        self.G = other.G
        self.x = torch.DoubleTensor(other.x).cuda()
        self.m = torch.DoubleTensor(other.m).cuda().unsqueeze(0).unsqueeze_(2).cuda()
        self.v = torch.DoubleTensor(other.v).cuda()
        self.da = torch.DoubleTensor(other.da).cuda()
        self.a = torch.DoubleTensor(other.da).zero_().cuda()

    # @profile
    def step(self, dt):
        '''Explicitly calculate a and da/dt; guess at da/dtt.'''
        dx = self.x.unsqueeze(1) - self.x.unsqueeze(0) #(n,n,3)
        dv = self.v.unsqueeze(1) - self.v.unsqueeze(0) #(n,n,3)

        dampened_norm = dx.pow(2).sum(dim=2).add_(self.e2) #(n,n)
        dampened_norm_32 = dampened_norm.pow(3/2).unsqueeze(2) #(n,n,1)
        dampened_norm_52 = dampened_norm.pow(5/2) #(n,n)
        a = -self.G * (self.m * (dx / dampened_norm_32)).sum(dim=1)

        # np.einsum("ijk,ijk->ij", dx,dv)
        ijk_ijk__ij = (dx.view(-1, 1, 3) @ dv.view(-1, 3, 1)).view_as(dampened_norm_52)
        assert(ijk_ijk__ij.is_cuda)
        da = -self.G * (self.m * (dv / dampened_norm_32 - (3. * ijk_ijk__ij / dampened_norm_52).unsqueeze(2) * dx)).sum(dim=1)
        dda = da - self.da
        self.da = da
        self.a = a

        self.x += dt*self.v + (dt**2)*a/2. + (dt**3)*da/6. + (dt**4)*dda/24.
        self.v += dt*a + (dt**2)*da/2. + (dt**3)*dda/6.

class Camera(object):
    def __init__(self, fov, aspect_ratio=(800/600), xyz=(0,0,0)):
        self._fov = fov
        self._xyz = np.array(xyz)
        self._aspect_ratio = aspect_ratio
        self._frustum = view_frustum(fov, aspect_ratio, 0.001, 1000.)
        self._translate = translate(self.xyz)

    def set_xyz(self, x, y, z):
        self._xyz = np.array([x,y,z])
        self._translate = translate(self.xyz)
        if hasattr(self, '_matrix'):
            del self._matrix
    def set_fov(self, fov):
        self._fov = fov
        self._frustum = view_frustum(fov, self._aspect_ratio, 0.01, 1000.)
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
    @xyz.setter
    def xyz(self, arg):
        self._xyz = np.array(arg)
#
# def test():
#     n_particles = 2500
#     nbody = NBody(n_particles, 0.1, 0.03, spread=4.5, mass_pareto=2., velocity_spread=0.5)
#     torch_nbody = torch_NBody(0)
#     torch_nbody.copy(nbody)
#
#     n_steps = 2500
#     step_size = 0.001
#     t0 = time.time()
#     for i in range(0,n_steps,1):
#         torch_nbody.step_cheap(step_size)
#         sys.stdout.write("\r%d   " % (i))
#     for i in range(n_steps,0,-1):
#         torch_nbody.step_cheap(-step_size)
#         sys.stdout.write("\r%d   " % (i))
#     t1 = time.time()
#     sys.stdout.write("\r")
#     print(np.sum(np.abs(torch_nbody.x.cpu().numpy() - nbody.x)) / n_particles)
#     print(t1-t0)
#     torch_nbody.copy(nbody)
#     t0 = time.time()
#     factor = 5
#     for i in range(0,int(n_steps/factor) ,1):
#         torch_nbody.step(factor*step_size)
#         sys.stdout.write("\r%.4f   " % (i))
#     for i in range(int(n_steps/factor),0,-1):
#         torch_nbody.step(factor*-step_size)
#         sys.stdout.write("\r%.4f   " % (i))
#     sys.stdout.write("\r")
#     t1 = time.time()
#     print(np.sum(np.abs(torch_nbody.x.cpu().numpy() - nbody.x)) / n_particles)
#     print(t1-t0)
#
# def test2():
#     n_particles = 2500
#     nbody = NBody(n_particles, 0.1, 0.03, spread=4.5, mass_pareto=2., velocity_spread=0.5)
#     torch_nbody = torch_NBody(0)
#     torch_nbody.copy(nbody)
#
#     n_seconds = 10
#     step_size1 = 0.005
#     step_size2 = 0.01
#
#     t0 = time.time()
#     for i in np.arange(0,n_seconds,step_size1):
#         torch_nbody.step(step_size1)
#         sys.stdout.write("\r%.4f   " % (i))
#     for i in np.arange(n_seconds,0,-step_size1):
#         torch_nbody.step(-step_size1)
#         sys.stdout.write("\r%.4f   " % (i))
#     t1 = time.time()
#
#     sys.stdout.write("\r")
#     print(np.sum(np.abs(torch_nbody.x.cpu().numpy() - nbody.x)) / n_particles)
#     print(t1-t0)
#
#     torch_nbody.copy(nbody)
#
#     t0 = time.time()
#     factor = 5
#     for i in np.arange(0,n_seconds,step_size2):
#         torch_nbody.step(step_size2)
#         sys.stdout.write("\r%.4f   " % (i))
#     for i in np.arange(n_seconds,0,-step_size2):
#         torch_nbody.step(-step_size2)
#         sys.stdout.write("\r%.4f   " % (i))
#     t1 = time.time()
#
#     sys.stdout.write("\r")
#     print(np.sum(np.abs(torch_nbody.x.cpu().numpy() - nbody.x)) / n_particles)
#     print(t1-t0)

def main():
    # test2()
    # quit()
    # print(view_frustum(90., 3./4., 0.1, 100.))
        # initialize glfw
    n_particles = 2000
    nbody = NBody(n_particles, damping=0.5, G=0.01, spread=5.5, mass_pareto=2., velocity_spread=0.4)
    cam = Camera(80, aspect_ratio=3500/2000, xyz=(0,0,-15))
    torch_nbody = torch_NBody(0)
    torch_nbody.copy(nbody)

    if not glfw.init():
        return
    window = glfw.create_window(3500, 2000, "", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,1.0)

    # Generate random
    shader = Shader()
    shader.view = cam.matrix

    model = Model("sphere.obj", center=True)
    model.use(shader.position)
    m = model.n_indices

    translate  = glGenBuffers(1)
    scale = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, scale)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, shader.obj_scale, scale)
    tmpscale = np.array(0.05 * (torch_nbody.m.cpu().view(-1) * .75 / np.pi).pow(1/3).numpy(), dtype=np.double)
    glBufferData(GL_SHADER_STORAGE_BUFFER, 8 * n_particles, tmpscale, GL_DYNAMIC_DRAW)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, translate)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, shader.obj_translate, translate)

    # shader.view = translate([0,0,-25]) @ view_frustum(120, 3500/2000, 0.01, 1000.)
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # time.sleep(1. / 60.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        torch_nbody.step(0.01)

        temp = torch_nbody.x.cpu().view(-1).numpy()
        glBufferData(GL_SHADER_STORAGE_BUFFER, 8 * len(temp), temp, GL_DYNAMIC_DRAW)

        glDrawElementsInstanced(GL_TRIANGLES, 3 * m, GL_UNSIGNED_INT, None, n_particles)

        glfw.swap_buffers(window)
    glfw.terminate()



if __name__ == '__main__':
    main()
