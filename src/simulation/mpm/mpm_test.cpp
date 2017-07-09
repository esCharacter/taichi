/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm_kernel.h"
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

class TestMPMKernel : public Task {

    template<int DIM, int ORDER>
    void test() {
        using Vector = VectorND<DIM, real>;

        std::cout << "Testing kernel <DIM=" << DIM << ", order=" << ORDER << ">..." << std::endl;

        for (int l = 0; l < 3; l++) {
            auto pos = Vector::rand() * 10.0f;
            MPMKernel<DIM, ORDER> kernel(pos);
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

TC_NAMESPACE_END
