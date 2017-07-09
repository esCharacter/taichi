/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <taichi/math/array.h>
#include <taichi/math/math.h>
#include <taichi/math/levelset.h>

#include "mpm_particle.h"

TC_NAMESPACE_BEGIN

template <int DIM>
class MPMScheduler {
public:

    using Particle = MPMParticle<DIM>;
    using Vector = VectorND<DIM, real>;
    using Vectori = VectorND<DIM, int>;

    template <typename T>
    using Array = ArrayND<DIM, T>;

    Array<int64> max_dt_int_strength;
    Array<int64> max_dt_int_cfl;
    Array<int64> max_dt_int;
    Array<int> belonging;
    Array<int> states;
    Array<int> updated;
    Array<Vector> max_vel, min_vel;
    Array<Vector> max_vel_expanded, min_vel_expanded;
    std::vector<std::vector<MPMParticle<DIM> *>> particle_groups;
    Vectori res;
    Vectori sim_res;
    std::vector<MPMParticle<DIM> *> active_particles;
    std::vector<Vectori> active_grid_points;
    DynamicLevelSet<DIM> *levelset;
    real base_delta_t;
    real cfl, strength_dt_mul;
    int grid_block_size;
    int node_id;

    void initialize(const Vectori &sim_res, real base_delta_t, real cfl, real strength_dt_mul,
                    DynamicLevelSet<DIM> *levelset, int node_id, int grid_block_size) {

        this->grid_block_size = grid_block_size;

        this->sim_res = sim_res;
        res = (sim_res - Vectori(1)) / Vectori(grid_block_size) + Vectori(1);

        this->base_delta_t = base_delta_t;
        this->levelset = levelset;
        this->cfl = cfl;
        this->strength_dt_mul = strength_dt_mul;
        this->node_id = node_id;

        states.initialize(res, 0);
        updated.initialize(res, 1);
        particle_groups.resize(res[0] * res[1] * res[2]);
        for (int i = 0; i < res[0] * res[1] * res[2]; i++) {
            particle_groups[i] = std::vector<MPMParticle<DIM> *>();
        }

        min_vel.initialize(res, Vector(0));
        min_vel = Vector(1e30f);
        max_vel.initialize(res, Vector(0));
        max_vel = Vector(-1e30f);
        min_vel_expanded.initialize(res, Vector(0));
        max_vel_expanded.initialize(res, Vector(0));

        max_dt_int_strength.initialize(res, 0);
        max_dt_int_cfl.initialize(res, 0);
        max_dt_int.initialize(res, 1);
        belonging.initialize(res, 0);
    }

    void reset() {
        states = 0;
    }

    bool has_particle(const IndexND<DIM> &ind) {
        return has_particle(ind.get_ipos());
    }

    template <int DIM_ = DIM>
    std::enable_if_t<DIM_ == 2, int> linearize(const Vectori &ind) const {
        return ind.x * res[1] + ind.y;
    }

    template <int DIM_ = DIM>
    std::enable_if_t<DIM_ == 3, int> linearize(const Vectori &ind) const {
        return ind.x * res[1] * res[2] + ind.y * res[2] + ind.z;
    }

    int linearize(const IndexND<DIM> &ind) const {
        return this->linearize(ind.get_ipos());
    }


    bool has_particle(const Vectori &ind) {
        return particle_groups[linearize(ind)].size() > 0;
    }

    void expand(bool expand_vel, bool expand_state);

    void update();

    int64 update_max_dt_int(int64 t_int);

    void set_time(int64 t_int) {
        for (auto &ind : states.get_region()) {
            if (t_int % max_dt_int[ind] == 0) {
                states[ind] = 1;
            }
        }
    }

    void update_particle_groups();

    void insert_particle(MPMParticle<DIM> *p, bool is_new_particle = false);

    void update_dt_limits(real t);

    int get_num_active_grids() {
        int count = 0;
        for (auto &ind : states.get_region()) {
            count += int(states[ind] != 0);
        }
        return count;
    }

    const std::vector<MPMParticle<DIM> *> &get_active_particles() const {
        return active_particles;
    }

    std::vector<MPMParticle<DIM> *> &get_active_particles() {
        return active_particles;
    }

    const std::vector<Vectori> &get_active_grid_points() const {
        return active_grid_points;
    }

    void update_particle_states();

    void reset_particle_states();

    void enforce_smoothness(int64 t_int_increment);

    Vectori get_rough_pos(const MPMParticle<DIM> *p) const {
        return p->pos.template cast<int>() / Vectori(grid_block_size);
    }

    Vectori get_rough_pos(const IndexND<DIM> &pos) const {
        return pos.get_ipos() / Vectori(grid_block_size);
    }

    int belongs_to(const IndexND<DIM> &pos) const {
        return belonging[get_rough_pos(pos)];
    }

    int belongs_to(const MPMParticle<DIM> *p) const {
        return belonging[get_rough_pos(p)];
    }

};


TC_NAMESPACE_END

