/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/util.h>
#include <taichi/math.h>

TC_NAMESPACE_BEGIN

class Pakua {
public:
    using Vector = VectorND<3, real>;

    virtual void initialize(const Config &config) {
        int port = config.get_int("port");
        //...
    }

    void add_particle(Vector pos, Vector color) {
        // Add a particle to buffer
    }

    // Reset and start a new canvas
    virtual void start() {

    }

    // Finish and send canvas (buffer) to frontend
    virtual void finish() {

    }
};

TC_NAMESPACE_END

