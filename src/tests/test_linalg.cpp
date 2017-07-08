/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

void test_linalg() {
    Vector3 a(1, 2, 3), b(4, 2, 5);
    assert(a + b == Vector3(5, 4, 8));
    assert(b - a == Vector3(3, 0, 2));
    assert(b * a == Vector3(4, 4, 15));
    assert(b / a == Vector3(4, 1, 5.0f / 3.0f));
    a += b;
    assert(a == Vector3(5, 4, 8));
    a -= b;
    assert(a == Vector3(1, 2, 3));
    a *= b;
    assert(a == Vector3(4, 4, 15));
    a /= b;
    assert(a == Vector3(1, 2, 3));
    a = Vector3({7.0f, 8.0f, 9.0f});
    assert(a == Vector3(7, 8, 9));

    auto t = __m128(a);

    Vector2 c(1, 2), d(2, 5);
    assert(c + d == Vector2(3, 7));

    assert(Vector4(1, 2, 3, 1).length2() == 15.0f);
    assert(Vector3(1, 2, 3, 1).length2() == 14.0f);
    assert(dot(Vector2(1, 2), Vector2(3, 2)) == 7.0f);
    assert(dot(Vector2i(1, 2), Vector2i(3, 2)) == 7);
    assert((fract(Vector2(1.3f, 2.7f)) - Vector2(0.3f, 0.7f)).length2() < 1e-10f);
    assert(Vector2(1.3f, 2.7f).sin() == Vector2(sin(1.3f), sin(2.7f)));

    assert(Matrix3(3.0f) + Matrix3(4.0f) == Matrix3(7.0f));
    assert(Matrix3(3.0f) + Matrix3(Vector3(1, 2, 3)) == Matrix3(Vector3(4, 5, 6)));

    assert(Matrix2(Vector2(1, 2)) * Vector2(2, 3) == Vector2(2, 6));
    assert(Matrix3(Vector3(1, 2, 3)) * Vector3(2, 3, 4) == Vector3(2, 6, 12));
    assert(Matrix4(Vector4(1, 2, 3, 4)) * Vector4(2, 3, 4, 5) == Vector4(2, 6, 12, 20));

    std::cout << "Passed." << std::endl;
}

TC_NAMESPACE_END
