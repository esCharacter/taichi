/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/profiler.h>
#include "mpm.h"
#include "mpm_kernel.h"

#ifndef MPM_TRANSFER_OPT
// Non-optimized Transfer

TC_NAMESPACE_BEGIN

// #define TC_MPM_USE_LOCKS
#ifdef TC_MPM_USE_LOCKS
#define LOCK_GRID grid_locks[ind].lock();
#define UNLOCK_GRID grid_locks[ind].unlock();
#else
#define LOCK_GRID
#define UNLOCK_GRID
#endif

template <int DIM>
void MPM<DIM>::rasterize(real delta_t) {
    real E = 0.0f;
    parallel_for_each_active_particle([&](Particle &p) {
        E += p.get_kenetic_energy();
        const Vector pos = p.pos, v = p.v;
        const real mass = p.mass;
        const Matrix apic_b_inv_d_mass = p.apic_b * (Kernel::inv_D() * mass);
        const Vector mass_v = mass * v;
        const MatrixP delta_t_tmp_force(delta_t * p.tmp_force);
        RegionND<D> region(VectorI(0), VectorI(Kernel::kernel_size));
        Vectori grid_base_pos([&](int i) -> int { return Kernel::get_stencil_start(pos[i]); });
        Kernel kernel(pos);

        for (auto &ind: region) {
            auto i = ind.get_ipos() + grid_base_pos;
            Vector dpos = i.template cast<real>() - pos;
            VectorP dw_w = kernel.get_dw_w(ind.get_ipos());
            VectorP delta =
                    dw_w[D] * VectorP(mass_v + apic_b_inv_d_mass * dpos, mass) +
                    VectorP(delta_t_tmp_force * Vector(dw_w));
            grid_velocity_and_mass[i] += delta;
        }
    });
    P(E);
#ifdef TC_MPM_WITH_FLIP
    error("grid_back_velocity is not in the correct position");
    grid_backup_velocity();
#endif
}

template <int DIM>
void MPM<DIM>::resample() {
    real alpha_delta_t = 1;
    if (apic)
        alpha_delta_t = 0;
    parallel_for_each_active_particle([&](Particle &p) {
        real delta_t = base_delta_t * (current_t_int - p.last_update);
        Vector v(0.0f), bv(0.0f);
        Matrix b(0.0f);
        Matrix cdg(0.0f);
        Vector pos = p.pos;

        RegionND<D> region(VectorI(0), VectorI(Kernel::kernel_size));
        Vectori grid_base_pos([&](int i) -> int { return Kernel::get_stencil_start(pos[i]); });
        Kernel kernel(pos);

        for (auto &ind: region) {
            auto i = ind.get_ipos() + grid_base_pos;
            auto grid_vel = grid_velocity[i];
            Vector dpos = i.template cast<real>() - pos;
            VectorP dw_w = kernel.get_dw_w(ind.get_ipos());

            v += dw_w[D] * grid_vel;
            b += Matrix::outer_product(dpos, dw_w[D] * grid_vel);
            cdg += Matrix::outer_product(Vector(dw_w), grid_vel);
        }

        if (!apic) {
            b = Matrix(0.0f);
        }
        // We should use an std::exp here, but that is too slow...
        real damping = std::max(0.0f, 1.0f - delta_t * affine_damping);
        p.apic_b = Matrix(b * damping);
        cdg = Matrix(1.0f) + delta_t * cdg;
#ifdef TC_MPM_WITH_FLIP
        // APIC part + FLIP part
        p.v = (1 - alpha_delta_t) * v + alpha_delta_t * (v - bv + p.v);
#else
        p.v = v;
#endif
        Matrix dg = cdg * p.dg_e * p.dg_p;
        p.dg_e = cdg * p.dg_e;
        p.dg_cache = dg;
    });
}

template
void MPM<2>::rasterize(real delta_t);

template
void MPM<2>::resample();

template
void MPM<3>::rasterize(real delta_t);

template
void MPM<3>::resample();


TC_NAMESPACE_END

#endif
