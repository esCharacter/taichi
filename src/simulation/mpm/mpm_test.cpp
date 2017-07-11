/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm_kernel.h"
#include "mpm_particle.h"
#include <taichi/util.h>
#include <taichi/math/qr_svd.h>

TC_NAMESPACE_BEGIN

class TestMPMKernel : public Task {

    template <int DIM, int ORDER>
    void test() {
        using Vector = VectorND<DIM, real>;

        std::cout << "Testing kernel <DIM=" << DIM << ", order=" << ORDER << ">..." << std::endl;

        for (int l = 0; l < 3; l++) {
            auto pos = Vector::rand() * 10.0f;
            MPMKernel<DIM, ORDER> kernel(pos, 1.0f);
            for (int j = 0; j < DIM; j++) {
                P(kernel.w_cache[j]);
                P(kernel.w_cache[j].sum());
                assert(abs(kernel.w_cache[j].sum() - 1) < 1e-6);
            }
            for (int j = 0; j < DIM; j++) {
                assert(kernel.dw_cache[j].sum() < 1e-6);
            }
        }
    }

    virtual void run() {
        test<3, 3>();
        test<3, 2>();
        test<2, 3>();
        test<2, 2>();

        std::cout << "All passed." << std::endl;
    }
};

TC_IMPLEMENTATION(Task, TestMPMKernel, "test_mpm_kernel")


template <int DIM>
MatrixND<DIM, real> levi_civita();

template <>
MatrixND<2, real> levi_civita<2>() {
    return MatrixND<2, real>(
            VectorND<2, real>(0, -1),
            VectorND<2, real>(1, 0)
    );
}

// Is this correct?
template <>
MatrixND<3, real> levi_civita<3>() {
    return MatrixND<3, real>(
            VectorND<3, real>(0, -1, 1),
            VectorND<3, real>(1, 0, -1),
            VectorND<3, real>(-1, 1, 0)
    );
}


class TestMPMDifferential : public Task {
    template <int DIM>
    void test_dR() {
        using Vector = VectorND<DIM, real>;
        using Matrix = MatrixND<DIM, real>;

        for (int i = 0; i < 10; i++) {
            Matrix F = Matrix::rand();

            Matrix R, S;
            polar_decomp(F, R, S);

            /*
            auto dR = [&](Matrix &dF) -> Matrix {
                // (tr(S)I - S)^-1
                Matrix term1 = inversed(Matrix(S.trace()) - S);
                Matrix term2 = elementwise_product(levi_civita<DIM>().transposed(), transposed(R) * dF);

                return R * elementwise_product(levi_civita<DIM>(), term1 * term2);
            };
            */

            auto dR = [&](Matrix &dF) -> Matrix {
                // set W = R^T dR = [  0    x  ]
                //                  [  -x   0  ]
                //
                // R^T dF - dF^T R = WS + SW
                //
                // WS + SW = [ x(s21 - s12)   x(s11 + s22) ]
                //           [ -x[s11 + s22]  x(s21 - s12) ]
                // ----------------------------------------------------
                Matrix lhs = transposed(R) * dF - transposed(dF) * R;
                real x = lhs[1][0] / (S[0][0] + S[1][1]);
                Matrix W = Matrix(Vector(0, -x), Vector(x, 0));
                return R * W;
            };

            auto dR_bf = [&](Matrix &dF) -> Matrix {
                Matrix R_, S_;
                polar_decomp(F + dF, R_, S_);
                return R_ - R;
            };

            Matrix dF = Matrix::rand() * 0.005f;
            P(dR(dF));
            P(dR_bf(dF));
            printf("%f\n", (dR(dF) - dR_bf(dF)).frobenius_norm());
        }
    }

    template <int DIM>
    void test_first_piola_kirchoff() {
        using Vector = VectorND<DIM, real>;
        using Matrix = MatrixND<DIM, real>;
        auto get_particle = [](const Matrix F) {
            EPParticle<DIM> p;
            p.dg_e = F;
            return p;
        };

        for (int i = 0; i < 10; i++) {
            Matrix F = Matrix::rand();

            auto p = get_particle(F);

            auto dP = [&](const Matrix &dF) -> Matrix {
                return p.get_first_piola_kirchoff(dF);
            };

            auto dP_bf = [&](const Matrix &dF) -> Matrix {
                return get_particle(F + dF).get_energy_gradient() - get_particle(F).get_energy_gradient();
            };

            Matrix dF = Matrix::rand() * 0.005f;
            P(dP(dF));
            P(dP_bf(dF));
            printf("%f\n", (dP(dF) - dP_bf(dF)).frobenius_norm());
        }
    }

    virtual void run() {
        test_dR<2>();
        test_first_piola_kirchoff<2>();
        // test<3>();

        std::cout << "All passed." << std::endl;
    }
};

TC_IMPLEMENTATION(Task, TestMPMDifferential, "test_mpm_differential")

TC_NAMESPACE_END
