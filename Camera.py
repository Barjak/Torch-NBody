import numpy as np

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
