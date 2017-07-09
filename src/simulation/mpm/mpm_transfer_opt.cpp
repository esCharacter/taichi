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

#ifdef MPM_TRANSFER_OPT

TC_NAMESPACE_BEGIN

// #define TC_MPM_USE_LOCKS
#ifdef TC_MPM_USE_LOCKS
#define LOCK_GRID grid_locks[ind].lock();
#define UNLOCK_GRID grid_locks[ind].unlock();
#else
#define LOCK_GRID
#define UNLOCK_GRID
#endif

template <>
void MPM<2>::rasterize(real delta_t) {
    {
        Profiler _("rasterize velocity, mass, and force");
        real E = 0.0f;
        parallel_for_each_active_particle([&](Particle &p) {
            E += p.get_kenetic_energy();
            TC_MPM3D_PREPROCESS_KERNELS
            const Vector pos = p.pos, v = p.v;
            const real mass = p.mass;
            const MatrixP apic_b_inv_d_mass = MatrixP(p.apic_b) * ((6.0f - Kernel::order) * mass);
            const Vector mass_v = mass * v;
            MatrixP apic_b_inv_d_mass_with_mass_v = apic_b_inv_d_mass;
            apic_b_inv_d_mass_with_mass_v[D] = mass_v;
            apic_b_inv_d_mass_with_mass_v[D][D] = mass;

            // apic_b_mass_with_mass_v
            // ----------------------------
            //               |
            //    APIC       |   mass * v
            // --------------|
            //       0             mass
            // ----------------------------

            const MatrixP delta_t_tmp_force(delta_t * p.tmp_force);
            Vectori c_min([&](int i) -> int { return this->get_stencil_start(pos[i]); });
            for (int i = 0; i < kernel_size; i++) {
                // v_contribution = v + 3 * apic_b * d_pos;
                // Vector4 rast_v = mass_v + (apic_b_inv_d_mass * d_pos);

                const VectorP dw_2 = w_stages[0][i];
                VectorP *ptr = &grid_velocity_and_mass[c_min[0] + i][c_min[1]];

                const VectorP d_pos = c_min.template cast<real>() + VectorP(i, 0.0f, 1.0f) - VectorP(pos);

                VectorP base_vel_and_mass = apic_b_inv_d_mass_with_mass_v * d_pos;
                VectorP dw_w = dw_2 * w_stages[D - 1][0];
                *ptr += dw_w[D] * base_vel_and_mass +
                        Matrix(delta_t_tmp_force) * Vector(dw_w);
                ptr += 1;

                if (kernel_size >= 2) {
                    base_vel_and_mass += apic_b_inv_d_mass_with_mass_v[D - 1];
                    dw_w = dw_2 * w_stages[D - 1][1];
                    *ptr += dw_w[D] * base_vel_and_mass +
                            Matrix(delta_t_tmp_force) * Vector(dw_w);
                    ptr += 1;
                }

                if (kernel_size >= 3) {
                    base_vel_and_mass += apic_b_inv_d_mass_with_mass_v[D - 1];
                    dw_w = dw_2 * w_stages[D - 1][2];
                    *ptr += dw_w[D] * base_vel_and_mass +
                            Matrix(delta_t_tmp_force) * Vector(dw_w);
                    ptr += 1;
                }

                if (kernel_size >= 4) {
                    base_vel_and_mass += apic_b_inv_d_mass_with_mass_v[D - 1];
                    dw_w = dw_2 * w_stages[D - 1][3];
                    *ptr += dw_w[D] * base_vel_and_mass +
                            Matrix(delta_t_tmp_force) * Vector(dw_w);
                }
            }
        });
        P(E);
    }
#ifdef TC_MPM_WITH_FLIP
    error("grid_back_velocity is not in the correct position");
    grid_backup_velocity();
#endif
}

template <>
void MPM<3>::rasterize(real delta_t) {
    TC_PROFILE("reset velocity_and_mass", grid_velocity_and_mass.reset(Vector4(0.0f)));
    TC_PROFILE("reset velocity", grid_velocity.reset(Vector(0.0f)));
    TC_PROFILE("reset mass", grid_mass.reset(0.0f));
    {
        Profiler _("rasterize velocity, mass, and force");
        real E = 0.0f;
        parallel_for_each_active_particle([&](Particle &p) {
            TC_MPM3D_PREPROCESS_KERNELS
            E += p.get_kenetic_energy();
            const Vector pos = p.pos, v = p.v;
            const real mass = p.mass;
            const MatrixP apic_b_inv_d_mass = MatrixP(p.apic_b) * ((6.0f - kernel_size) * mass);
            const Vector mass_v = mass * v;
            MatrixP apic_b_inv_d_mass_with_mass_v = apic_b_inv_d_mass;
            apic_b_inv_d_mass_with_mass_v[D] = mass_v;
            apic_b_inv_d_mass_with_mass_v[D][D] = mass;

            // apic_b_mass_with_mass_v
            // ----------------------------
            //               |
            //    APIC       |   mass * v
            // --------------|
            //       0             mass
            // ----------------------------

            const MatrixP delta_t_tmp_force(delta_t * p.tmp_force);
            Vectori c_min([&](int i) -> int { return this->get_stencil_start(pos[i]); });
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    // v_contribution = v + 3 * apic_b * d_pos;
                    // Vector4 rast_v = mass_v + (apic_b_inv_d_mass * d_pos);

                    const VectorP dw_2 = w_stages[0][i] * w_stages[1][j];
                    VectorP *ptr = &grid_velocity_and_mass[c_min[0] + i][c_min[1] + j][c_min[2]];

                    const VectorP d_pos = c_min.template cast<real>() + VectorP(i, j, 0.0f, 1.0f) - VectorP(pos);

                    VectorP base_vel_and_mass = apic_b_inv_d_mass_with_mass_v * d_pos;
                    VectorP dw_w = dw_2 * w_stages[D - 1][0];
                    *ptr += dw_w[D] * base_vel_and_mass +
                            delta_t_tmp_force.multiply_vec3(dw_w);
                    ptr += 1;

                    if (kernel_size >= 2) {
                        base_vel_and_mass += apic_b_inv_d_mass_with_mass_v[D - 1];
                        dw_w = dw_2 * w_stages[D - 1][1];
                        *ptr += dw_w[D] * base_vel_and_mass +
                                delta_t_tmp_force.multiply_vec3(dw_w);
                        ptr += 1;
                    }

                    if (kernel_size >= 3) {
                        base_vel_and_mass += apic_b_inv_d_mass_with_mass_v[D - 1];
                        dw_w = dw_2 * w_stages[D - 1][2];
                        *ptr += dw_w[D] * base_vel_and_mass +
                                delta_t_tmp_force.multiply_vec3(dw_w);
                        ptr += 1;
                    }

                    if (kernel_size >= 4) {
                        base_vel_and_mass += apic_b_inv_d_mass_with_mass_v[D - 1];
                        dw_w = dw_2 * w_stages[D - 1][3];
                        *ptr += dw_w[D] * base_vel_and_mass +
                                delta_t_tmp_force.multiply_vec3(dw_w);
                    }
                }
            }
        });
        P(E);
    }
#ifdef TC_MPM_WITH_FLIP
    error("grid_back_velocity is not in the correct position");
    grid_backup_velocity();
#endif
}


template <>
void MPM<2>::resample() {
    real alpha_delta_t = 1;
    if (apic)
        alpha_delta_t = 0;
    parallel_for_each_active_particle([&](Particle &p) {
        real delta_t = base_delta_t * (current_t_int - p.last_update);
        Vector v(0.0f), bv(0.0f);
        Matrix b(0.0f);
        Matrix cdg(0.0f);
        Vector pos = p.pos;
        TC_MPM2D_PREPROCESS_KERNELS
        int x_min = get_stencil_start(pos.x);
        int y_min = get_stencil_start(pos.y);
        // TODO: FLIP velocity sample is temporarily disabled
        for (int i = 0; i < kernel_size; i++) {
            // Note: forth coordinate of the second parameter to Matrix3::outer_product
            // is ignored.
            //
            int j;

            j = 0;
            Vector *grid_velocity_ptr = &grid_velocity[x_min + i][y_min];
            Vector dpos = Vector(x_min + i, y_min) - pos;

            VectorP dw_2 = w_stages[0][i];
            VectorP dw_w = dw_2 * w_stages[2][j];
            Vector grid_vel = grid_velocity_ptr[j];
            v += dw_w[D] * grid_vel;
            b += Matrix::outer_product(dpos, dw_w[D] * grid_vel);
            cdg += Matrix::outer_product(Vector(dw_w), grid_vel);

            if (kernel_size >= 2) {
                j = 1;
                dpos += Vector(0.0f, 1.0f);
                dw_w = dw_2 * w_stages[1][j];
                grid_vel = grid_velocity_ptr[j];
                v += dw_w[D] * grid_vel;
                b += Matrix3::outer_product(dpos, dw_w[D] * grid_vel);
                cdg += Matrix3::outer_product(Vector(dw_w), grid_vel);
            }

            if (kernel_size >= 3) {
                j = 2;
                dpos += Vector(0.0f, 1.0f);
                dw_w = dw_2 * w_stages[1][j];
                grid_vel = grid_velocity_ptr[j];
                v += dw_w[D] * grid_vel;
                b += Matrix3::outer_product(dpos, dw_w[D] * grid_vel);
                cdg += Matrix3::outer_product(Vector(dw_w), grid_vel);
            }

            if (kernel_size >= 4) {
                j = 3;
                dpos += Vector(0.0f, 1.0f);
                dw_w = dw_2 * w_stages[1][j];
                grid_vel = grid_velocity_ptr[j];
                v += dw_w[D] * grid_vel;
                b += Matrix::outer_product(dpos, dw_w[D] * grid_vel);
                cdg += Matrix::outer_product(Vector(dw_w), grid_vel);
            }
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
        p.v = Vector3(v);
#endif
        Matrix3 dg = cdg * p.dg_e * Matrix3(p.dg_p);
        p.dg_e = cdg * p.dg_e;
        p.dg_cache = dg;
    });
}

template <>
void MPM<3>::resample() {
    real alpha_delta_t = 1;
    if (apic)
        alpha_delta_t = 0;
    parallel_for_each_active_particle([&](Particle &p) {
        real delta_t = base_delta_t * (current_t_int - p.last_update);
        Vector v(0.0f), bv(0.0f);
        Matrix b(0.0f);
        Matrix cdg(0.0f);
        Vector pos = p.pos;
        TC_MPM3D_PREPROCESS_KERNELS
        int x_min = get_stencil_start(pos.x);
        int y_min = get_stencil_start(pos.y);
        int z_min = get_stencil_start(pos.z);
        // TODO: FLIP velocity sample is temporarily disabled
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                // Note: forth coordinate of the second parameter to Matrix3::outer_product
                // is ignored.
                //
                int k;

                k = 0;
                Vector *grid_velocity_ptr = &grid_velocity[x_min + i][y_min + j][z_min];
                Vector dpos = Vector(x_min + i, y_min + j, z_min) - pos;

                VectorP dw_2 = w_stages[0][i] * w_stages[1][j];
                VectorP dw_w = dw_2 * w_stages[2][k];
                Vector grid_vel = grid_velocity_ptr[k];
                v += dw_w[3] * grid_vel;
                b += Matrix::outer_product(dpos, dw_w[3] * grid_vel);
                cdg += Matrix::outer_product(Vector(dw_w), grid_vel);

                if (kernel_size >= 2) {
                    k = 1;
                    dpos += Vector(0.0f, 0.0f, 1.0f);
                    dw_w = dw_2 * w_stages[2][k];
                    grid_vel = grid_velocity_ptr[k];
                    v += dw_w[3] * grid_vel;
                    b += Matrix3::outer_product(dpos, dw_w[3] * grid_vel);
                    cdg += Matrix3::outer_product(Vector(dw_w), grid_vel);
                }

                if (kernel_size >= 3) {
                    k = 2;
                    dpos += Vector(0.0f, 0.0f, 1.0f);
                    dw_w = dw_2 * w_stages[2][k];
                    grid_vel = grid_velocity_ptr[k];
                    v += dw_w[3] * grid_vel;
                    b += Matrix3::outer_product(dpos, dw_w[3] * grid_vel);
                    cdg += Matrix3::outer_product(Vector(dw_w), grid_vel);
                }

                if (kernel_size >= 4) {
                    k = 3;
                    dpos += Vector(0.0f, 0.0f, 1.0f);
                    dw_w = dw_2 * w_stages[2][k];
                    grid_vel = grid_velocity_ptr[k];
                    v += dw_w[3] * grid_vel;
                    b += Matrix::outer_product(dpos, dw_w[3] * grid_vel);
                    cdg += Matrix::outer_product(Vector(dw_w), grid_vel);
                }
            }
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
        p.v = Vector3(v);
#endif
        Matrix3 dg = cdg * p.dg_e * Matrix3(p.dg_p);
#ifdef CV_ON
        if (abnormal(dg) || abnormal(cdg) || abnormal(p.dg_e) || abnormal(p.dg_cache)) {
            P(dg);
            P(cdg);
            P(p.dg_e);
            P(p.dg_p);
            P(p.dg_cache);
            error("");
        }
#endif
        p.dg_e = cdg * p.dg_e;
        p.dg_cache = dg;
#ifdef CV_ON
        if (abnormal(dg) || abnormal(cdg) || abnormal(p.dg_e) || abnormal(p.dg_cache)) {
            P(dg);
            P(cdg);
            P(p.dg_e);
            P(p.dg_p);
            P(p.dg_cache);
            error("");
        }
#endif
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
