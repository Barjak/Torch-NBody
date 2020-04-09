from OpenGL.GL import *
import numpy as np

class Model(object):
    '''Not fully implemented'''
    def __init__(self, name, center=True, type=np.float32):
        self.name = name
        self.vbuffer = -1
        self.ibuffer = -1
        self._vertices = []
        self._vertex_normals = []
        self._indices = []
        with open(name, "r") as f:
            for line in f:
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

    def load(self):
        if self.vbuffer != -1 or self.ibuffer != -1:
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
        glVertexAttribPointer(attrib, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(attrib)

    def unload(self):
        glDeleteBuffers(2, (self.vbuffer, self.ibuffer))
        self.vbuffer = -1
        self.ibuffer = -1
