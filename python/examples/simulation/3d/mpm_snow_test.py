import math

from taichi.dynamics.mpm import MPM3
from taichi.core import tc_core
from taichi.misc.util import Vector, Vectori
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture
import taichi as tc

step_number = 1000
grid_downsample = 8
output_downsample = 1
render_epoch = 20

if __name__ == '__main__':
    downsample = grid_downsample
    resolution = (256 / downsample, 256 / downsample, 256 / downsample)

    mpm = MPM3(resolution=resolution, gravity=(0, 0, 0), base_delta_t=0.001, num_threads=1, use_mpi=False)

    '''
    levelset = mpm.create_levelset()
    levelset.add_plane(0, 1, 0, -0.4)
    levelset.set_friction(-1)
    mpm.set_levelset(levelset, False)
    '''

    tex_ball = Texture('sphere', center=(0.5, 0.7, 0.5), radius=0.1) * 10
    mpm.add_particles(density_tex=tex_ball.id, initial_velocity=(0, -5, 0), compression=1)

    tex_ball = Texture('sphere', center=(0.5, 0.3, 0.5), radius=0.1) * 10
    mpm.add_particles(density_tex=tex_ball.id, initial_velocity=(0, 5, 0), compression=1)

    t = 0
    for i in range(step_number):
        print 'process(%d/%d)' % (i, step_number)
        mpm.step(0.03)
        tc.core.print_profile_info()
        t += 0.03
