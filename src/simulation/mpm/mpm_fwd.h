/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/util.h>

TC_NAMESPACE_BEGIN

constexpr int mpm_kernel_order = 2;

template <int DIM, int ORDER>
struct MPMKernel;

template <int DIM>
class MPM;

template <int DIM>
class MPMScheduler;

template <int DIM>
class MPMParticle;

TC_NAMESPACE_END
