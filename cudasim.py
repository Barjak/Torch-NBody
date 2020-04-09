#!/usr/bin/env python3
import torch
import glfw
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

from Shader import Shader as Shader
from Camera import Camera as Camera
from Wavefront_OBJ import Model
from NBody import torch_NBody

@torch.no_grad()
def main():
    res = (2600,1800)
    n_particles = 2000
    cam = Camera(fov=80,
                 aspect_ratio=res[0]/res[1],
                 xyz=(0,0,-10))
    torch_nbody = torch_NBody(n_particles,
                              damping=0.08,
                              G=.05,
                              spread=3.,
                              mass_pareto=1.7,
                              velocity_spread=0.5,
                              dtype=torch.cuda.FloatTensor)

    if not glfw.init():
        return
    window = glfw.create_window(res[0], res[1], "", None, None)
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

    buffer_translate  = glGenBuffers(1)
    buffer_scale = glGenBuffers(1)
    buffer_brightness = glGenBuffers(1)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_scale)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, shader.obj_scale, buffer_scale)
    scale = np.array(0.03 * (torch_nbody.mass.cpu().view(-1) * .75 / np.pi).pow(1/3).to(torch.float32).numpy())
    glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * n_particles, scale, GL_DYNAMIC_DRAW)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_translate)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, shader.obj_translate, buffer_translate)

    exposure = 200.0
    while not glfw.window_should_close(window):
        
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        torch_nbody.step(0.01)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, shader.obj_translate, buffer_translate)
        translate = torch_nbody.position.cpu().view(-1).to(torch.float32).numpy()
        glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * len(translate), translate, GL_DYNAMIC_DRAW)

        # Example visualization
        brightness = torch_nbody.a.norm(dim=1)
        exposure = exposure * 0.98 + brightness.max() * 0.02
        brightness /= exposure
        brightness = 0.1 + 0.9 * brightness
        brightness = brightness.cpu().view(-1).to(torch.float32).numpy()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, shader.obj_brightness, buffer_brightness)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * len(brightness), brightness, GL_DYNAMIC_DRAW)

        glDrawElementsInstanced(GL_TRIANGLES, 3 * m, GL_UNSIGNED_INT, None, n_particles)

        glfw.swap_buffers(window)
    glfw.terminate()

if __name__ == '__main__':
    main()
