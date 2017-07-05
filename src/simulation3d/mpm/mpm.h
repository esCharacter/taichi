/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <memory.h>
#include <string>
#include <functional>

#include <taichi/visualization/image_buffer.h>
#include <taichi/common/meta.h>
#include <taichi/dynamics/simulation3d.h>
#include <taichi/math/array_3d.h>
#include <taichi/math/qr_svd.h>
#include <taichi/math/levelset.h>
#include <taichi/math/dynamic_levelset_3d.h>
#include <taichi/system/threading.h>

#include "mpm_scheduler.h"
#include "mpm_particle.h"
#include "mpm_utils.h"

TC_NAMESPACE_BEGIN

// Supports FLIP?
// #define TC_MPM_WITH_FLIP

template <int DIM>
class MPM : public Simulation3D {
public:
    /*
    using Vector = Vector<DIM>;
    using VectorP = Vector<DIM + 1>;
    using VectorI = Vector<DIM, int>;
    using Matrix = Matrix<DIM>;
    */
    using Vector = Vector3;
    using VectorP = Vector4;
    using VectorI = Vector3i;
    using Matrix = Matrix3;
    static const int D = DIM;
    static const int kernel_size;
    typedef MPMParticle<DIM> Particle;
    std::vector<MPMParticle<DIM> *> particles; // for (copy) efficiency, we do not use smart pointers here
    Array3D<Vector> grid_velocity;
    Array3D<real> grid_mass;
    Array3D<Vector4s> grid_velocity_and_mass;
#ifdef TC_MPM_WITH_FLIP
    Array3D<Vector> grid_velocity_backup;
#endif
    Array3D<Spinlock> grid_locks;
    VectorI res;
    Vector gravity;
    bool apic;
    bool async;
    real affine_damping;
    real base_delta_t;
    real maximum_delta_t;
    real cfl;
    real strength_dt_mul;
    real request_t = 0.0f;
    bool use_mpi;
    int mpi_world_size;
    int mpi_world_rank;
    int64 current_t_int = 0;
    int64 original_t_int_increment;
    int64 t_int_increment;
    int64 old_t_int;
    MPMScheduler<DIM> scheduler;
    static const int grid_block_size = 8;
    bool mpi_initialized;

    bool test() const override;

    void estimate_volume() {}

    void resample();

    void grid_backup_velocity() {
#ifdef TC_MPM_WITH_FLIP
        grid_velocity_backup = grid_velocity;
#endif
    }

    void calculate_force_and_rasterize(float delta_t);

    void grid_apply_boundary_conditions(const DynamicLevelSet3D &levelset, real t);

    void grid_apply_external_force(Vector acc, real delta_t) {
        for (auto &ind : grid_mass.get_region()) {
            if (grid_mass[ind] > 0) // Do not use EPS here!!
                grid_velocity[ind] += delta_t * acc;
        }
    }

    void particle_collision_resolution(real t);

    void substep();

    template <typename T>
    void parallel_for_each_particle(const T &target) {
        ThreadedTaskManager::run((int)particles.size(), num_threads, [&](int i) {
            target(*particles[i]);
        });
    }

    template <typename T>
    void parallel_for_each_active_particle(const T &target) {
        ThreadedTaskManager::run((int)scheduler.get_active_particles().size(), num_threads, [&](int i) {
            target(*scheduler.get_active_particles()[i]);
        });
    }

public:

    MPM() {}

    virtual void initialize(const Config &config) override;

    virtual void add_particles(const Config &config) override;

    virtual void step(real dt) override {
        if (dt < 0) {
            substep();
            request_t = current_t;
        } else {
            request_t += dt;
            while (current_t + base_delta_t < request_t) {
                substep();
            }
            P(t_int_increment * base_delta_t);
        }
    }

    void synchronize_particles();

    void finalize();

    void clear_particles_outside();

    std::vector<RenderParticle> get_render_particles() const override;

    int get_mpi_world_rank() const override {
        return mpi_world_rank;
    }

    void clear_boundary_particles();

    ~MPM();

    int get_stencil_start(real x) const;
};

TC_NAMESPACE_END

