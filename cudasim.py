#!/usr/bin/python3
import torch
import glfw
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

import time

from Shader import Shader as Shader
from Camera import Camera as Camera
from Wavefront_OBJ import Model
from NBody import NBody, torch_NBody

def main():
    n_particles = 1800
    # nbody = NBody(n_particles, damping=0.5, G=0.01, spread=5.5, mass_pareto=2., velocity_spread=0.5)
    cam = Camera(80, aspect_ratio=3500/2000, xyz=(0,0,-15))
    torch_nbody = torch_NBody(n_particles, damping=0.5, G=0.01, spread=5.5, mass_pareto=2., velocity_spread=0.5)
    # torch_nbody.copy(nbody)

    if not glfw.init():
        return
    window = glfw.create_window(3500, 2000, "", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,1.0)

    shader = Shader()
    shader.use()
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
