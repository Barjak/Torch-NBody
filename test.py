#!/usr/bin/python3
from NBody import *
import time
import sys
def main():
    n_particles = 2500
    nbody = NBody(n_particles, 0.1, 0.03, spread=4.5, mass_pareto=2., velocity_spread=0.5)
    torch_nbody = torch_NBody(0)
    torch_nbody.copy(nbody)

    n_seconds = 10
    step_size1 = 0.005
    step_size2 = 0.01

    t0 = time.time()
    for i in np.arange(0,n_seconds,step_size1):
        torch_nbody.step(step_size1)
        sys.stdout.write("\r%.4f   " % (i))
    for i in np.arange(n_seconds,0,-step_size1):
        torch_nbody.step(-step_size1)
        sys.stdout.write("\r%.4f   " % (i))
    t1 = time.time()

    sys.stdout.write("\r")
    print(np.sum(np.abs(torch_nbody.x.cpu().numpy() - nbody.x)) / n_particles)
    print(t1-t0)

    torch_nbody.copy(nbody)

    t0 = time.time()
    factor = 5
    for i in np.arange(0,n_seconds,step_size2):
        torch_nbody.step(step_size2)
        sys.stdout.write("\r%.4f   " % (i))
    for i in np.arange(n_seconds,0,-step_size2):
        torch_nbody.step(-step_size2)
        sys.stdout.write("\r%.4f   " % (i))
    t1 = time.time()

    sys.stdout.write("\r")
    print(np.sum(np.abs(torch_nbody.x.cpu().numpy() - nbody.x)) / n_particles)
    print(t1-t0)


if __name__ == "__main__":
    main()
