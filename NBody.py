import numpy as np
import torch

class torch_NBody(object):
    @torch.no_grad()
    def __init__(self, n, damping=0.001, G=0.001, spread=4., mass_pareto=2.0, velocity_spread=0.0, dtype=torch.cuda.FloatTensor):
        self.Type = dtype
        if n == 0:
            return
        self.n = n
        self.e2 = damping ** 2
        self.G = G
        self.position = self.Type(n,3).normal_(std=spread)
        self.mass = self.Type(np.random.pareto(mass_pareto, size=(n,))).unsqueeze(0).unsqueeze_(2)
        if velocity_spread <= 0.0:
            self.v = self.Type(n,3).zero_()
        else:
            self.v = self.Type(n,3).normal_(std=velocity_spread)
        self.a = self.Type(n,3).zero_()
        self.da = self.Type(n,3).zero_()

    @torch.no_grad()
    def copy(self, other):
        self.n = other.n
        self.e2 = other.e2
        self.G = other.G
        self.position = self.Type(other.x)
        self.mass = self.Type(other.m).unsqueeze_(0).unsqueeze_(2)
        self.v = self.Type(other.v)
        self.da = self.Type(other.da)
        self.a = self.Type(other.da).zero_()

    @torch.no_grad()
    def step(self, dt):
        '''Explicitly calculate a and da/dt; guess at da/dtt.'''
# TODO: Calculate only the lower triangle of each matrix and copy it to the upper?
#       Must use entire matrix because of mass calculation

        dx = self.position.unsqueeze(1) - self.position.unsqueeze(0) #(n,n,3)
        dv = self.v.unsqueeze(1) - self.v.unsqueeze(0) #(n,n,3)

        dampened_norm = dx.pow(2).sum(dim=2).add_(self.e2) #(n,n)
        dampened_norm_32 = dampened_norm.pow(3/2).unsqueeze(2) #(n,n,1)
        dampened_norm_52 = dampened_norm.pow(5/2) #(n,n)



        # Take the dot product of every point on the nxn grid.
        # dx_[i,j] . dv_[i,j]
# TODO: Profile and see which version is faster
#        ijk_ijk__ij = torch.einsum("ijk,ijk->ij", dx,dv) # (n,n)
        ijk_ijk__ij = (dx.view(-1, 1, 3) @ dv.view(-1, 3, 1)).view_as(dampened_norm_52)

        a = -self.G * (self.mass * (dx / dampened_norm_32)).sum(dim=1)
        da = -self.G * (self.mass * ((dv      / dampened_norm_32) - (3. * ijk_ijk__ij / dampened_norm_52).unsqueeze(2) * dx)).sum(dim=1)
        #       []     ([1,n,1]     (([n,n,3]     [n,n,1]       )   ([]     [n,n]           [n,n]       )+[,,1]     [n,n,3])).sum
        a0_a1 = self.a - a
        dda = (-6. * (a0_a1) - dt * (4. * self.da + 2 * da)) / (dt**2)
        ddda = (-12. * (a0_a1) - 6. * dt * (self.da + da)) / (dt**3)
        self.da = da
        self.a = a

        self.position = self.position + dt*self.v + (dt**2)*a/2. + (dt**3)*da/6. + (dt**4)*dda/24. + (dt**5)*ddda/120
        self.v += dt*a + (dt**2)*da/2. + (dt**3)*dda/6. + (dt**4)*ddda/24
