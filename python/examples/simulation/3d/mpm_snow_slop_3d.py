import math

from taichi.misc.util import Vector, Vectori
from taichi.visual.texture import Texture
import taichi as tc

num_steps = 100
grid_downsample = 4

if __name__ == '__main__':
    downsample = grid_downsample

    res = (256 / downsample, 256 / downsample, 256 / downsample)

    mpm = tc.dynamics.MPM(res=res, delta_x=1.0 / res[0], gravity=(0, -20000, 0), base_delta_t=0.001, num_threads=1)

    tex_ball = Texture('sphere', center=(0.5, 0.5, 0.5), radius=0.10) * 10
    mpm.add_particles(density_tex=tex_ball.id, initial_velocity=(0, -100, 0), compression=0.95)

    frame_dt = 0.03

    t = 0
    for i in range(num_steps):
        mpm.step(frame_dt)
        if mpm.get_mpi_world_rank() == 0:
            tc.core.print_profile_info()
            t += frame_dt
