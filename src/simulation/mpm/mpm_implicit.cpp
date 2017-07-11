/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm.h"

TC_NAMESPACE_BEGIN

template <>
void MPM<3>::implicit_velocity_update(real delta_t) {
    NOT_IMPLEMENTED
}

template <>
void MPM<2>::implicit_velocity_update(real delta_t) {
    using Array = ArrayND<D, Vector>;

    auto apply = [&](const Array &x, Array &y) {
        // x: input
        // y: output

        // Calculate Ap : Grid -> Particle
        for (auto &p : scheduler.get_active_particles()) {
            const Vector pos = p->pos * inv_delta_x;
            Kernel kernel(pos, inv_delta_x);
            RegionND<D> region(VectorI(0), VectorI(Kernel::kernel_size));

            auto grid_base_pos = get_grid_base_pos(pos);

            Matrix dF(0.0f);
            for (auto &ind: region) {
                auto i = ind.get_ipos() + grid_base_pos;
                Vector dw = kernel.get_dw(ind.get_ipos());
                // row, column
                dF += Matrix::outer_product(dw, x[i]);
            }
            dF = dF * p->dg_e;

            p->Ap = p->get_first_piola_kirchoff(dF);
        }

        y = x.same_shape(Vector(0.0f));

        // Particle -> Grid
        for (auto &p : scheduler.get_active_particles()) {
            const Vector pos = p->pos * inv_delta_x;
            Kernel kernel(pos, inv_delta_x);
            RegionND<D> region(VectorI(0), VectorI(Kernel::kernel_size));
            auto grid_base_pos = get_grid_base_pos(pos);
            Matrix VApFt = p->vol * p->Ap * transposed(p->dg_e);
            for (auto &ind: region) {
                auto i = ind.get_ipos() + grid_base_pos;
                Vector dw = kernel.get_dw(ind.get_ipos());
                y[i] += VApFt * dw;
            }
        }
        for (auto &ind : grid_mass.get_region()) {
            y[ind] = (implicit_ratio * pow<2>(delta_t)) * y[ind] + x[ind] * grid_mass[ind];
        }
    };

    Array x = grid_velocity, rhs = grid_velocity.same_shape();
    for (auto &ind: grid_velocity.get_region()) {
        rhs[ind] = grid_velocity[ind] * grid_mass[ind];
    }

    auto dot = [](const Array &a, const Array &b) -> auto {
        assert_info(a.get_res() == b.get_res(), "Shape mismatch.");
        float64 sum = 0.0f;
        for (auto &ind : a.get_region()) {
            sum += a[ind].dot(b[ind]);
        }
        return float32(sum);
    };

    Array r, Ax, p, Ar, Ap;

    apply(x, Ax);
    r = rhs - Ax;
    p = r;
    apply(r, Ar);
    Ap = Ar;
    real rtAr = dot(r, Ar);
    Array tmp;
    bool early_break = false;
    for (int k = 0; k < implicit_solve_iteration_limit; k++) {
        real Ap_sqr = dot(Ap, Ap) + 1e-38f;
        real alpha = rtAr / Ap_sqr;
        x = x.add(alpha, p);
        r = r.add(-alpha, Ap);
        real d_error = sqrt(dot(r, r));
        printf("CR error: %.10f  log=%.2f\n", d_error, log(d_error) / log(10.0f));
        if (d_error < implicit_solve_tolerance) {
            printf("CR converged at iteration %d\n", k + 1);
            early_break = true;
            break;
        }
        apply(r, Ar);
        real new_rtAr = dot(r, Ar);
        real beta = new_rtAr / rtAr;
        rtAr = new_rtAr;
        p = r.add(beta, p);
        Ap = Ar.add(beta, Ap);
    }
    if (!early_break) {
        printf("Warning: CR iteration exceeds upper limit\n");
    }
    grid_velocity = x;
}

TC_NAMESPACE_END
