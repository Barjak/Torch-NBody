import numpy as np
import torch

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
