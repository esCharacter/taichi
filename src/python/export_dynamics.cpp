/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/dynamics/fluid2d/fluid.h>
#include <taichi/dynamics/simulation3d.h>
#include <taichi/common/asset_manager.h>

PYBIND11_MAKE_OPAQUE(std::vector<taichi::RenderParticle>);

TC_NAMESPACE_BEGIN

void export_dynamics(py::module &m) {
    m.def("register_levelset3d", &AssetManager::insert_asset<LevelSet3D>);

    py::class_<Fluid>(m, "Fluid")
            .def(py::init<>())
            .def("initialize", &Fluid::initialize)
            .def("step", &Fluid::step)
            .def("add_particle", &Fluid::add_particle)
            .def("get_current_time", &Fluid::get_current_time)
            .def("get_particles", &Fluid::get_particles)
            .def("set_levelset", &Fluid::set_levelset)
            .def("get_liquid_levelset", &Fluid::get_liquid_levelset)
            .def("get_density", &Fluid::get_density)
            .def("get_pressure", &Fluid::get_pressure)
            .def("add_source", &Fluid::add_source);

    py::class_<Simulation3D, std::shared_ptr<Simulation3D>>(m, "Simulation3D")
            .def(py::init<>())
            .def("initialize", &Simulation3D::initialize)
            .def("add_particles", &Simulation3D::add_particles)
            .def("update", &Simulation3D::update)
            .def("step", &Simulation3D::step)
            .def("get_current_time", &Simulation3D::get_current_time)
            .def("get_render_particles", &Simulation3D::get_render_particles)
            .def("set_levelset", &Simulation3D::set_levelset)
            .def("get_mpi_world_rank", &Simulation3D::get_mpi_world_rank)
            .def("test", &Simulation3D::test);

    typedef std::vector<Fluid::Particle> FluidParticles;
    py::class_<FluidParticles>(m, "FluidParticles");
}

TC_NAMESPACE_END
