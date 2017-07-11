/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/


#ifdef TC_USE_MPI

#include <mpi.h>

#endif

#include <taichi/system/threading.h>
#include <taichi/visual/texture.h>
#include <taichi/math/qr_svd.h>
#include <taichi/math.h>
#include <taichi/common/asset_manager.h>
#include <taichi/system/profiler.h>

#include "mpm.h"

TC_NAMESPACE_BEGIN

template <int DIM>
void MPM<DIM>::initialize(const Config &config) {
    Simulation<DIM>::initialize(config);
    res = config.get_vec3i("res");
    delta_x = config.get("delta_x", delta_x);
    implicit_ratio = config.get("implicit_ratio", 0.0f);
    inv_delta_x = 1.0f / delta_x;
    gravity = config.get("gravity", Vector(0.0f));
    use_mpi = config.get("use_mpi", false);
    apic = config.get("apic", true);
    async = config.get("async", false);
    mpi_initialized = false;
    t_int_increment = 0;
    if (use_mpi) {
#ifndef TC_USE_MPI
        error("Not compiled with MPI. Please recompile with cmake -DTC_USE_MPI=True")
#else
        assert_info(!async, "AsyncMPM support is not finished.");
        MPI_Init(nullptr, nullptr);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
#endif
    } else {
        mpi_world_size = 1;
        mpi_world_rank = 0;
    }
    base_delta_t = config.get("base_delta_t", 1e-6f);
    cfl = config.get("cfl", 1.0f);
    strength_dt_mul = config.get("strength_dt_mul", 1.0f);
    TC_LOAD_CONFIG(affine_damping, 0.0f);
    if (async) {
        maximum_delta_t = config.get("maximum_delta_t", 1e-1f);
    } else {
        maximum_delta_t = base_delta_t;
    }
    implicit_solve_tolerance = config.get("implicit_solve_tolerance", 1e-5f);
    implicit_solve_iteration_limit = config.get("implicit_solve_iteration_limit", 100);

    grid_velocity.initialize(res + VectorI(1), Vector(0.0f), Vector(0.0f));
    grid_mass.initialize(res + VectorI(1), 0, Vector(0.0f));
    grid_velocity_and_mass.initialize(res + VectorI(1), VectorP(0.0f), Vector(0.0f));
    grid_locks.initialize(res + VectorI(1), 0, Vector(0.0f));
    scheduler.initialize(res, base_delta_t, cfl, strength_dt_mul, &this->levelset, mpi_world_rank, grid_block_size,
                         delta_x);

    imex_mask.initialize(res + VectorI(1), 0.0f, Vector(0.0f));

    for (auto &ind: imex_mask.get_region()) {
        if (ind.i < res[0] / 2) {
            imex_mask[ind] = 1.0f;
        }
    }

    pakua = create_instance<Pakua>("webgl");
    Config config_;
    config_.set("port", 9563);
    pakua->initialize(config_);
}

template <int DIM>
void MPM<DIM>::add_particles(const Config &config) {
    std::shared_ptr<Texture> density_texture = AssetManager::get_asset<Texture>(config.get_int("density_tex"));
    auto region = RegionND<DIM>(Vectori(0), res);
    for (auto &ind: region) {
        Vector coord = (ind.get_ipos().template cast<real>() + Vector::rand()) * this->delta_x;
        real num = density_texture->sample(coord).x;
        int t = (int)num + (rand() < fract(num));
        for (int l = 0; l < t; l++) {
            Particle *p = nullptr;
            if (config.get("type", std::string("ep")) == std::string("ep")) {
                p = new EPParticle<DIM>();
            } else {
                p = new DPParticle<DIM>();
            }
            p->initialize(config);
            p->pos = (ind.get_ipos().template cast<real>() + Vector::rand()) * delta_x;
            p->vol = pow<D>(delta_x / 2.0f);
            p->mass = p->vol * 1000.0f;
            p->v = config.get("initial_velocity", p->v);
            p->last_update = current_t_int;
            particles.push_back(p);
            scheduler.insert_particle(p, true);
        }
    }
    P(particles.size());
}

template <int DIM>
std::vector<RenderParticle> MPM<DIM>::get_render_particles() const {
    std::vector<RenderParticle> render_particles;
    render_particles.reserve(particles.size());
    Vector center(res.template cast<real>() * 0.5f);
    for (auto p_p : particles) {
        Particle &p = *p_p;
        // at least synchronize the position
        Vector pos = (p.pos + (current_t_int - p.last_update) * base_delta_t * p.v) * inv_delta_x - center;
        if (p.state == Particle::UPDATING) {
            render_particles.push_back(RenderParticle(pos, Vector4(0.8f, 0.1f, 0.2f, 0.5f)));
        } else if (p.state == Particle::BUFFER) {
            render_particles.push_back(RenderParticle(pos, Vector4(0.8f, 0.8f, 0.2f, 0.5f)));
        } else {
            render_particles.push_back(RenderParticle(pos, Vector4(0.8f, 0.9f, 1.0f, 0.5f)));
        }
    }
    return render_particles;
}

template <int DIM>
void MPM<DIM>::normalize_grid() {
    real E = 0;
    for (auto ind : grid_mass.get_region()) {
        auto &velocity_and_mass = grid_velocity_and_mass[ind];
        const real mass = velocity_and_mass[D];
        if (mass > 0) {
            grid_mass[ind] = mass;
            grid_velocity[ind] = (1.0f / mass) * Vector(velocity_and_mass);
            E += 0.5f * mass * grid_velocity[ind].length2();
        }
    }
    //P(E);
}

template <int DIM>
void MPM<DIM>::calculate_force() {
    parallel_for_each_active_particle([&](Particle &p) {
        p.calculate_force();
    });
}

template <int DIM>
void MPM<DIM>::grid_apply_boundary_conditions(const DynamicLevelSet<D> &levelset, real t) {
    ArrayND<DIM, int> cache(scheduler.res, 0);
    for (auto ind: cache.get_region()) {
        Vector pos = Vector(ind.get_pos()) * real(grid_block_size);
        if (levelset.sample(pos, t) < grid_block_size) {
            cache[ind] = 1;
        } else {
            cache[ind] = 0;
        }
    }
    for (auto &ind : scheduler.get_active_grid_points()) {
        if (cache[ind / Vector3i(grid_block_size)] == 0) {
            continue;
        }
        Vector pos = Vector(ind) + Vector(0.5f);
        real phi = levelset.sample(pos, t);
        if (1 < phi || phi < -3) continue;
        Vector n = levelset.get_spatial_gradient(pos, t);
        Vector boundary_velocity = levelset.get_temporal_derivative(pos, t) * n;
        Vector v = grid_velocity[ind] - boundary_velocity;
        if (phi > 0) { // 0~1
            real pressure = std::max(-dot(v, n), 0.0f);
            real mu = levelset.levelset0->friction;
            if (mu < 0) { // sticky
                v = Vector(0.0f);
            } else {
                Vector t = v - n * dot(v, n);
                if (length(t) > 1e-6f) {
                    t = normalize(t);
                }
                real friction = -clamp(dot(t, v), -mu * pressure, mu * pressure);
                v = v + n * pressure + t * friction;
            }
        } else if (phi < 0.0f) {
            v = n * std::max(0.0f, dot(v, n));
        }
        v += boundary_velocity;
        grid_velocity[ind] = v;
    }
}

template <int DIM>
void MPM<DIM>::particle_collision_resolution(real t) {
    parallel_for_each_active_particle([&](Particle &p) {
        if (p.state == Particle::UPDATING) {
            Vector pos = p.pos * inv_delta_x;
            real phi = this->levelset.sample(pos, t);
            if (phi < 0) {
                Vector gradient = this->levelset.get_spatial_gradient(pos, t);
                p.pos -= gradient * phi * delta_x;
                p.v -= dot(gradient, p.v) * gradient;
            }
        }
    });
}

template <int DIM>
void MPM<DIM>::step(real dt) {
    if (dt < 0) {
        substep();
        request_t = this->current_t;
    } else {
        request_t += dt;
        while (this->current_t + base_delta_t < request_t) {
            substep();
        }
        P(t_int_increment * base_delta_t);
    }

    pakua->start();
    {
        if (DIM == 2) {
            std::vector<Vector3> pos_v{
                    Vector3(0.f, 0.f, 0.5f), Vector3(0.f, 1.f, 0.5f),
                    Vector3(0.f, 1.f, 0.5f), Vector3(1.f, 1.f, 0.5f),
                    Vector3(1.f, 1.f, 0.5f), Vector3(1.f, 0.f, 0.5f),
                    Vector3(1.f, 0.f, 0.5f), Vector3(0.f, 0.f, 0.5f)
            };
            std::vector<Vector3> color_v(8, Vector3());
            for (int i = 0; i < res[0]; i++)
                for (int j = 0; j < res[1]; j++) {
                    real d00 = this->levelset.sample(Vector2((real)i, (real)j), this->current_t);
                    real d01 = this->levelset.sample(Vector2((real)i, (real)(j + 1)), this->current_t);
                    real d10 = this->levelset.sample(Vector2((real)(i + 1), (real)j), this->current_t);
                    real d11 = this->levelset.sample(Vector2((real)(i + 1), (real)(j + 1)), this->current_t);
                    int count = 0;
                    if (d00 * d01 < 0) {
                        count++;
                        real d = abs(d00 / (d00 - d01));
                        pos_v.push_back(Vector3(delta_x * i, delta_x * (d + j), 0.5f));
                        color_v.push_back(Vector3());
                    }
                    if (d00 * d10 < 0) {
                        count++;
                        real d = abs(d00 / (d00 - d10));
                        pos_v.push_back(Vector3(delta_x * (d + i), delta_x * j, 0.5f));
                        color_v.push_back(Vector3());
                    }
                    if (d01 * d11 < 0) {
                        count++;
                        real d = abs(d01 / (d01 - d11));
                        pos_v.push_back(Vector3(delta_x * (d + i), delta_x * (1 + j), 0.5f));
                        color_v.push_back(Vector3());
                    }
                    if (d10 * d11 < 0) {
                        count++;
                        real d = abs(d10 / (d10 - d11));
                        pos_v.push_back(Vector3(delta_x * (1 + i), delta_x * (d + j), 0.5f));
                        color_v.push_back(Vector3());
                    }
                    if (count % 2 == 1) {
                        pos_v.pop_back();
                        color_v.pop_back();
                    }
                }
            pakua->add_line(pos_v, color_v);
        }
        for (auto p: particles) {
            pakua->add_point(Vector3(p->pos, 0.5f), p->color);
        }
    }
    TC_PROFILE("pakua_finish", pakua->finish());
}

template <int DIM>
void MPM<DIM>::substep() {
    Profiler _p("mpm_substep");
    synchronize_particles();
    if (particles.empty()) {
        return;
    }
    if (async) {
        scheduler.update_particle_groups();
        scheduler.reset_particle_states();
        old_t_int = current_t_int;
        scheduler.reset();
        scheduler.update_dt_limits(this->current_t);

        original_t_int_increment = std::min(get_largest_pot(int64(maximum_delta_t / base_delta_t)),
                                            scheduler.update_max_dt_int(current_t_int));

        t_int_increment = original_t_int_increment - current_t_int % original_t_int_increment;

        current_t_int += t_int_increment;
        this->current_t = current_t_int * base_delta_t;

        scheduler.set_time(current_t_int);

        scheduler.expand(false, true);
    } else {
        // sync
        t_int_increment = 1;
        scheduler.states = 2;
        scheduler.update_particle_groups();
        scheduler.reset_particle_states();
        old_t_int = current_t_int;
        for (auto &p : particles) {
            p->state = Particle::UPDATING;
        }
        current_t_int += t_int_increment;
        this->current_t = current_t_int * base_delta_t;
    }
    if (!use_mpi) {
        TC_PROFILE("update", scheduler.update());
    }
    TC_PROFILE("calculate force", this->calculate_force());
    {
        Profiler _("reset grids");
        grid_velocity_and_mass.reset(Vector4(0.0f));
        grid_velocity.reset(Vector(0.0f));
        grid_mass.reset(0.0f);
    }
    TC_PROFILE("rasterize", rasterize(t_int_increment * base_delta_t));
    TC_PROFILE("normalize_grid", normalize_grid());
    TC_PROFILE("external_force", grid_apply_external_force(gravity, t_int_increment * base_delta_t));
    if (implicit_ratio > 0) {
        TC_PROFILE("implicit velocity update", implicit_velocity_update(t_int_increment * base_delta_t));
    }
    TC_PROFILE("boundary_condition", grid_apply_boundary_conditions(this->levelset, this->current_t));
    TC_PROFILE("resample", resample());
    if (!async) {
        for (auto &p: particles) {
            assert_info(p->state == Particle::UPDATING, "should be updating");
        }
    }
    {
        Profiler _("plasticity");
        // TODO: should this be active particle?
        parallel_for_each_particle([&](Particle &p) {
            if (p.state == Particle::UPDATING) {
                p.pos += (current_t_int - p.last_update) * base_delta_t * p.v;
                p.last_update = current_t_int;
                p.pos = (p.pos * inv_delta_x).clamp(Vector(0.0f), res - Vector(eps)) * delta_x;
                p.plasticity();
            }
        });
    }
    TC_PROFILE("particle_collision", particle_collision_resolution(this->current_t));
    if (async)
        scheduler.enforce_smoothness(original_t_int_increment);
    TC_PROFILE("clean boundary", clear_boundary_particles());
}


template <int DIM>
void MPM<DIM>::clear_particles_outside() {
    std::vector<Particle *> new_active_particles;
    for (auto p: scheduler.get_active_particles()) {
        if (scheduler.belongs_to(p) == mpi_world_rank) {
            new_active_particles.push_back(p);
        } else {
            // delete p;
        }
    }
    scheduler.get_active_particles() = new_active_particles;
}


template <int DIM>
bool MPM<DIM>::test() const {
    /*
    for (int i = 0; i < 100000; i++) {
        Matrix3 m(1.000000238418579101562500000000, -0.000000000000000000000000000000,
                  -0.000000000000000000000220735070, 0.000000000000000000000000000000, 1.000000238418579101562500000000,
                  -0.000000000000000000216840434497, 0.000000000000000000000211758237,
                  -0.000000000000000001084202172486, 1.000000000000000000000000000000);
        Matrix3 u, sig, v;
        svd(m, u, sig, v);
        if (!is_normal(sig)) {
            P(m);
            P(u);
            P(sig);
            P(v);
        }
    }
    */
    return false;
}

template <int DIM>
void MPM<DIM>::clear_boundary_particles() {
    std::vector<MPMParticle<DIM> *> particles;
    real bound = 3.0f;
    int deleted = 0;
    for (auto p : scheduler.get_active_particles()) {
        auto pos = p->pos * inv_delta_x;
        if (pos.min() < bound || (pos - res).max() > -bound) {
            deleted += 1;
            continue;
        }
        particles.push_back(p);
    }
    if (deleted != 0) {
        printf("Warning: %d boundary particles deleted\n", deleted);
        P(particles.size());
    }
    scheduler.active_particles = particles;
}

template <int DIM>
MPM<DIM>::~MPM() {
    for (auto &p : particles) {
        delete p;
    }
}

typedef MPM<2> MPM2D;
typedef MPM<3> MPM3D;

template
class MPM<2>;

template
class MPM<3>;

TC_IMPLEMENTATION(Simulation2D, MPM2D, "mpm");

TC_IMPLEMENTATION(Simulation3D, MPM3D, "mpm");

TC_NAMESPACE_END
