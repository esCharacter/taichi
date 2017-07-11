/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm_particle.h"

TC_NAMESPACE_BEGIN


template<>
MatrixND<3, real> EPParticle<3>::get_first_piola_kirchoff(const MatrixND<3, real> &dF) {
    NOT_IMPLEMENTED;
    return Matrix(0.0f);
};

template<>
MatrixND<2, real> EPParticle<2>::get_first_piola_kirchoff(const MatrixND<2, real> &dF) {
    const Matrix &f = this->dg_e;
    const real j_e = determinant(this->dg_e);
    const real j_p = determinant(this->dg_p);
    const real e = expf(hardening * (1.0f - j_p));
    const real mu = mu_0 * e;
    const real lambda = lambda_0 * e;
    const auto F = this->dg_e;
    Matrix r, s;
    polar_decomp(this->dg_e, r, s);
    Matrix dR = dR_from_dF(this->dg_e, r, s, dF);
    Matrix JFmT = Matrix(Vector(F[1][1], -F[1][0]), Vector(-F[0][1], F[0][0]));
    Matrix dJFmT = Matrix(Vector(dF[1][1], -dF[1][0]), Vector(-dF[0][1], dF[0][0]));
    return 2.0 * mu * (dF - dR) + lambda * JFmT * (JFmT.elementwise_product(dF)).sum() + lambda * (j_e - 1) * dJFmT;
};

TC_NAMESPACE_END
