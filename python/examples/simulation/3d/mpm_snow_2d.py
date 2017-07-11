import taichi as tc
import time

num_steps = 200
grid_downsample = 2

if __name__ == '__main__':
    downsample = grid_downsample

    res = (256 / downsample, 256 / downsample)

    mpm = tc.dynamics.MPM(res=res, delta_x=1.0 / res[0], gravity=(0, 0, 0), base_delta_t=0.001, num_threads=1,
                          implicit_ratio=0.0, implicit_solve_tolerance=1e-6, implicit_solve_iteration_limit=30)

    levelset = mpm.create_levelset()

    #levelset.add_plane(tc.Vector(0, 1), -0.4)
    levelset.add_sphere(tc.Vector(0.5, 0.5), 0.45, True)
    mpm.set_levelset(levelset, False)

    tex_ball = tc.Texture('sphere', center=(0.3, 0.7, 0.5), radius=0.12) * 4
    mpm.add_particles(density_tex=tex_ball.id, initial_velocity=(0, -0.6, 0), compression=1.0)

    tex_ball = tc.Texture('sphere', center=(0.3, 0.3, 0.5), radius=0.12) * 4
    mpm.add_particles(density_tex=tex_ball.id, initial_velocity=(0, 0.6, 0), compression=0.9)

    frame_dt = 0.06

    t = 0
    for i in range(num_steps):
        mpm.step(frame_dt)
        tc.core.print_profile_info()
        t += frame_dt
        # time.sleep(0.1)
