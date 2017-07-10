/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/


#ifdef TC_USE_MPI

#include <mpi.h>

#endif

#include <taichi/system/threading.h>
#include <taichi/visual/texture.h>
#include <taichi/math.h>
#include <taichi/common/asset_manager.h>
#include <taichi/system/profiler.h>

#include "mpm.h"

TC_NAMESPACE_BEGIN

#define TC_MPM_TAG_BELONGING 1
#define TC_MPM_TAG_PARTICLE_COUNT 2
#define TC_MPM_TAG_PARTICLES 3

template <int DIM>
void MPM<DIM>::synchronize_particles() {
    if (!use_mpi) {
        // No need for this
        return;
    }
#ifdef TC_USE_MPI
    // Count number of particles with in each block
    Array3D<int> self_particle_count(scheduler.res, 0);
    P(scheduler.get_active_particles().size());
    for (auto p: scheduler.get_active_particles()) {
        self_particle_count[scheduler.get_rough_pos(p)] += 1;
    }
    auto old_belonging = scheduler.belonging;
    if (mpi_world_rank == 0) { // Master
        // Receive particle counts
        Array3D<int> particle_count_all(scheduler.res, 0);
        Array3D<int> particle_count(scheduler.res, 0);
        for (int i = 0; i < mpi_world_size; i++) {
            if (i != 0) {
                // Fetch from slaves
                MPI_Recv(static_cast<void *>(&particle_count.get_data()[0]), particle_count.get_size(), MPI_INT, i,
                         TC_MPM_TAG_PARTICLE_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int sum = 0;
                for (auto &ind: particle_count.get_region()) {
                    sum += particle_count[ind];
                    if (particle_count[ind] > 0) {
                        //P(scheduler.belonging[ind]);
                    }
                }
                //P(sum);
            } else {
                particle_count = self_particle_count;
            }
            for (auto &ind: particle_count.get_region()) {
                if (scheduler.belonging[ind] == i) {
                    particle_count_all[ind] = particle_count[ind];
                }
            }
        }
        // Re-decomposition according to x-axis slices
        std::vector<int> total_num_particles_x((size_t)scheduler.res[0], 0);
        std::vector<int> belonging_x((size_t)scheduler.res[0], 0);
        Array3D<int> belonging(scheduler.res);
        int total_num_particles = 0;
        for (auto &ind: particle_count_all.get_region()) {
            total_num_particles_x[ind.i] += particle_count_all[ind];
            total_num_particles += particle_count_all[ind];
        }
        // Determine slices
        int accumulated_num_particles = 0;
        int head = 0;
        int threshold = total_num_particles / mpi_world_size + 1;
        P(total_num_particles);
        P(threshold);
        for (int i = 0; i < scheduler.res[0]; i++) {
            accumulated_num_particles += total_num_particles_x[i];
            while (accumulated_num_particles >= threshold) {
                accumulated_num_particles -= threshold;
                head += 1;
            }
            belonging_x[i] = head;
            // P(head);
        }
        // Broadcast into y and z
        for (auto &ind: belonging.get_region()) {
            belonging[ind] = belonging_x[ind.i];
        }
        scheduler.belonging = belonging;
        for (int i = 1; i < mpi_world_size; i++) {
            // Send partition information to other nodes
            MPI_Send((void *)&scheduler.belonging.get_data()[0], scheduler.belonging.get_size(), MPI_INT, i,
                     TC_MPM_TAG_BELONGING, MPI_COMM_WORLD);
        }
    } else {
        // Slaves
        // Send particle count to master
        MPI_Send(&self_particle_count.get_data()[0], self_particle_count.get_size(), MPI_INT, 0,
                 TC_MPM_TAG_PARTICLE_COUNT,
                 MPI_COMM_WORLD);
        // Receive new domain decomposition information
        MPI_Recv((void *)&scheduler.belonging.get_data()[0], scheduler.belonging.get_size(), MPI_INT, 0,
                 TC_MPM_TAG_BELONGING,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int block_count = 0;
        for (auto ind: scheduler.belonging.get_region()) {
            block_count += scheduler.belonging[ind] == mpi_world_rank;
        }
        P(block_count);
    }
    if (!mpi_initialized) {
        // Delete particles outside
        // During first iteration, clear particles outside before synchronization
        clear_particles_outside();
    }
    std::vector<std::vector<EPParticle>> particles_to_send;
    particles_to_send.resize(mpi_world_size);
    for (int i = 0; i < mpi_world_size; i++) {
        particles_to_send[i] = std::vector<EPParticle>();
    }
    // Exchange (a small amount of) particles to other nodes
    for (auto &p: scheduler.get_active_particles()) {
        int belongs_to = scheduler.belongs_to(p);
        if (belongs_to != mpi_world_rank) {
            particles_to_send[belongs_to].push_back(*(EPParticle *)p);
        }
    }
    P(scheduler.get_active_particles().size());
    std::vector<char> to_receive(0);
    for (int i = 0; i < mpi_world_size; i++) {
        if (i == mpi_world_rank) continue;
        if (i < mpi_world_rank) {
            // Send, and then receive
            int to_send = particles_to_send[i].size();
            MPI_Send(&to_send, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD);
            if (to_send)
                MPI_Send(&particles_to_send[i][0], to_send * sizeof(EPParticle), MPI_CHAR, i,
                         TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD);
            int to_recv;
            MPI_Recv(&to_recv, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            to_receive.resize(to_recv * sizeof(EPParticle));
            if (to_recv) {
                MPI_Recv(&to_receive[0], to_recv * sizeof(EPParticle), MPI_CHAR, i, TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else if (i > mpi_world_rank) {
            // Receive, and then send
            int to_recv;
            MPI_Recv(&to_recv, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            to_receive.resize(to_recv * sizeof(EPParticle));
            if (to_recv)
                MPI_Recv(&to_receive[0], to_recv * sizeof(EPParticle), MPI_CHAR, i, TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int to_send = particles_to_send[i].size();
            MPI_Send(&to_send, 1, MPI_INT, i, TC_MPM_TAG_PARTICLES, MPI_COMM_WORLD);
            if (to_send) {
                MPI_Send(&particles_to_send[i][0], particles_to_send[i].size() * sizeof(EPParticle), MPI_CHAR, i,
                         TC_MPM_TAG_PARTICLES,
                         MPI_COMM_WORLD);
            }
        }
        for (int i = 0; i < to_receive.size() / sizeof(EPParticle); i++) {
            EPParticle *ptr = new EPParticle(*(EPParticle *)&to_receive[i * sizeof(EPParticle)]);
            scheduler.get_active_particles().push_back(ptr);
        }
    }
    for (auto &p: scheduler.get_active_particles()) {
        p->state = Particle::UPDATING;
        int b = scheduler.belongs_to(p);
        p->color = Vector3(b % 2, b / 2 % 2, b / 4 % 2);
    }
    if (mpi_initialized) {
        // Delete particles outside
        clear_particles_outside();
    }
    particles = scheduler.active_particles;
    mpi_initialized = true;
#endif
}

template <int DIM>
void MPM<DIM>::finalize() {
#ifdef TC_USE_MPI
    MPI_Finalize();
#endif
}

template
void MPM<2>::synchronize_particles();

template
void MPM<3>::synchronize_particles();

template
void MPM<2>::finalize();

template
void MPM<3>::finalize();

TC_NAMESPACE_END
