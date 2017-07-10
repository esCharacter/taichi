/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/util.h>
#include <taichi/math/qr_svd.h>
#include <taichi/math/array.h>
#include <taichi/math/levelset.h>

TC_NAMESPACE_BEGIN

template <int DIM>
class MPMParticle {
public:
    using Vector = VectorND<DIM, real>;
    using Matrix = MatrixND<DIM, real>;
    using Region = RegionND<DIM>;
    static const int D = DIM;
    Vector3 color = Vector3(1, 0, 0);
    real vol;
    Vector v;
    real mass;
    Vector pos;
    Matrix dg_e, dg_p, tmp_force;
    Matrix apic_b;
    Matrix dg_cache;
    enum State {
        INACTIVE = 0,
        BUFFER = 1,
        UPDATING = 2,
    };
    int state = INACTIVE;
    int64 last_update;

    MPMParticle() {
        last_update = 0;
        dg_e = Matrix(1.0f);
        dg_p = Matrix(1.0f);
        apic_b = Matrix(0);
        v = Vector(0.0f);
        vol = 1.0f;
    }

    virtual real get_allowed_dt() const = 0;

    virtual void initialize(const Config &config) {
    }

    virtual void set_compression(float compression) {
        dg_p = Matrix(compression); // 1.0f = no compression
    }

    virtual Matrix get_energy_gradient() = 0;

    virtual void calculate_force() = 0;

    virtual void plasticity() {};

    virtual void resolve_collision(const DynamicLevelSet <DIM> &levelset, real t) {
        real phi = levelset.sample(pos, t);
        if (phi < 0) {
            Vector gradient = levelset.get_spatial_gradient(pos, t);
            pos -= gradient * phi;
            v -= dot(gradient, v) * gradient;
        }
    }

    virtual void print() {
        P(pos);
        P(v);
        P(dg_e);
        P(dg_p);
    }

    virtual ~MPMParticle() {}

    real get_kenetic_energy() const {
        return dot(v, v) * mass * 0.5f;
    }

    Vector get_momentum() const {
        return mass * v;
    }

    uint64 key() const {
        // 3D Morton Coding
        const uint64 mask_x = 0x9249249249249249ULL;
        return _pdep_u64(uint64(pos.x), mask_x) | _pdep_u64(uint64(pos.y), mask_x << 1) |
               _pdep_u64(uint64(pos.z), mask_x << 2);
    }
};

template <int DIM>
class EPParticle : public MPMParticle<DIM> {
public:
    using Vector = typename MPMParticle<DIM>::Vector;
    using Matrix = typename MPMParticle<DIM>::Matrix;
    real hardening = 10.0f;
    real mu_0 = 58333.3, lambda_0 = 38888.9;
    real theta_c = 2.5e-2f, theta_s = 7.5e-3f;

    EPParticle() : MPMParticle<DIM>() {
    }

    EPParticle(const EPParticle<DIM> &other) {
        this->hardening = other.hardening;
        this->mu_0 = other.mu_0;
        this->lambda_0 = other.lambda_0;
        this->theta_c = other.theta_c;
        this->theta_s = other.theta_s;
        this->color = other.color;
        this->pos = other.pos;
        this->v = other.v;
        this->dg_e = other.dg_e;
        this->dg_p = other.dg_p;
        this->tmp_force = other.tmp_force;
        this->mass = other.mass;
        this->vol = other.vol;
        this->apic_b = other.apic_b;
        this->dg_cache = other.dg_cache;
        this->state = other.state;
        this->last_update = other.last_update;
    }

    void initialize(const Config &config) override {
        hardening = config.get("hardening", hardening);
        lambda_0 = config.get("lambda_0", lambda_0);
        mu_0 = config.get("mu_0", mu_0);
        theta_c = config.get("theta_c", theta_c);
        theta_s = config.get("theta_s", theta_s);
        real compression = config.get("compression", 1.0f);
        this->dg_p = Matrix(compression);
    }

    virtual Matrix get_energy_gradient() override {
        real j_e = determinant(this->dg_e);
        real j_p = determinant(this->dg_p);
        auto lame = get_lame_parameters();
        real mu = lame.first, lambda = lame.second;
        Matrix r, s;
        polar_decomp(this->dg_e, r, s);
        Matrix grad = 2 * mu * (this->dg_e - r) +
                      lambda * (j_e - 1) * j_e * inverse(transpose(this->dg_e));
        return grad;
    }

    virtual void calculate_force() override {
        this->tmp_force = -this->vol * get_energy_gradient() * transpose(this->dg_e);
    };

    virtual void plasticity() override {
        Matrix svd_u, sig, svd_v;
        svd(this->dg_e, svd_u, sig, svd_v);
        for (int i = 0; i < DIM; i++) {
            sig[i][i] = clamp(sig[i][i], 1.0f - theta_c, 1.0f + theta_s);
        }
        this->dg_e = svd_u * sig * transposed(svd_v);
        this->dg_p = inversed(this->dg_e) * this->dg_cache;
        // clamp dg_p to ensure that it does not explode
        svd(this->dg_p, svd_u, sig, svd_v);
        for (int i = 0; i < DIM; i++) {
            sig[i][i] = clamp(sig[i][i], 0.1f, 10.0f);
        }
        this->dg_p = svd_u * sig * transposed(svd_v);
    };

    std::pair<real, real> get_lame_parameters() const {
        real j_e = determinant(this->dg_e);
        real j_p = determinant(this->dg_p);
        // real e = std::max(1e-7f, std::exp(std::min(hardening * (1.0f - j_p), 5.0f)));
        // no clamping
        real e = std::exp(hardening * (1.0f - j_p));
        real mu = mu_0 * e;
        real lambda = lambda_0 * e;
        return {mu, lambda};
    }

    virtual real get_allowed_dt() const override {
        auto lame = get_lame_parameters();
        real strength_limit = 0.5f / std::sqrt(lame.first + 2 * lame.second + 1e-7f);
        return strength_limit;
    }
};

template <int DIM>
class DPParticle : public MPMParticle<DIM> {
public:
    using Vector = typename MPMParticle<DIM>::Vector;
    using Matrix = typename MPMParticle<DIM>::Matrix;
    real h_0 = 35.0f, h_1 = 9.0f, h_2 = 0.2f, h_3 = 10.0f;
    real lambda_0 = 204057.0f, mu_0 = 136038.0f;
    real alpha = 1.0f;
    real q = 0.0f;

    DPParticle() : MPMParticle<DIM>() {
    }

    void initialize(const Config &config) override {
        h_0 = config.get("h_0", h_0);
        h_1 = config.get("h_1", h_1);
        h_2 = config.get("h_2", h_2);
        h_3 = config.get("h_3", h_3);
        lambda_0 = config.get("lambda_0", lambda_0);
        mu_0 = config.get("mu_0", mu_0);
        alpha = config.get("alpha", alpha);
        real compression = config.get("compression", 1.0f);
        this->dg_p = Matrix(compression);
    }

    Matrix get_energy_gradient() override {
        return Matrix(1.f);
    }

    void project(Matrix sigma, real alpha, Matrix &sigma_out, real &out) {
        const real d = this->D;
        Matrix epsilon(sigma.diag().map(logf));
        real tr = epsilon.diag().sum();
        Matrix epsilon_hat = epsilon - (tr) / d * Matrix(1.0f);
        real epsilon_for = epsilon.diag().length();
        real epsilon_hat_for = epsilon_hat.diag().length();
        if (epsilon_hat_for <= 0 || tr > 0.0f) {
            sigma_out = Matrix(1.0f);
            out = epsilon_for;
        } else {
            real delta_gamma = epsilon_hat_for + (d * lambda_0 + 2 * mu_0) / (2 * mu_0) * tr * alpha;
            if (delta_gamma <= 0) {
                sigma_out = sigma;
                out = 0;
            } else {
                Matrix h = epsilon - delta_gamma / epsilon_hat_for * epsilon_hat;
                sigma_out = Matrix(h.diag().map(expf));
                out = delta_gamma;
            }
        }
    }

    void calculate_force() override {
        Matrix u, v, sig, dg = this->dg_e;
        svd(this->dg_e, u, sig, v);

#ifdef CV_ON
        assert_info(sig[0][0] > 0, "negative singular value");
        assert_info(sig[1][1] > 0, "negative singular value");
        assert_info(sig[2][2] > 0, "negative singular value");
#endif

        Matrix log_sig(sig.diag().template map(logf));
        Matrix inv_sig(Vector(1.f) / sig.diag());
        Matrix center =
                2.0f * mu_0 * inv_sig * log_sig + lambda_0 * (log_sig.diag().sum()) * inv_sig;

        this->tmp_force = -this->vol * (u * center * transpose(v)) * transpose(dg);
    }

    void plasticity() override {
        Matrix u, v, sig;
        svd(this->dg_e, u, sig, v);
        Matrix t(1.0f);
        real delta_q = 0;
        project(sig, alpha, t, delta_q);
        Matrix rec = u * sig * transpose(v);
        Matrix diff = rec - this->dg_e;
#ifdef CV_ON
        if (!(frobenius_norm(diff) < 1e-4f)) {
            // debug code
            P(dg_e);
            P(rec);
            P(u);
            P(sig);
            P(v);
            error("SVD error\n");
        }
#endif
        this->dg_e = u * t * transposed(v);
        this->dg_p = v * inversed(t) * sig * transposed(v) * this->dg_p;
        q += delta_q;
        real phi = h_0 + (h_1 * q - h_3) * expf(-h_2 * q);
        alpha = std::sqrt(2.0f / 3.0f) * (2.0f * std::sin(phi * pi / 180.0f)) / (3.0f - std::sin(phi * pi / 180.0f));
    }

    real get_allowed_dt() const override {
        return 0.0f;
    }

};

TC_NAMESPACE_END
