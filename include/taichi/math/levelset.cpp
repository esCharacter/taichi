/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "levelset.h"

TC_NAMESPACE_BEGIN

template <int DIM>
void LevelSet<DIM>::add_sphere(LevelSet<DIM>::Vector center, real radius, bool inside_out) {
    for (auto &ind : get_region()) {
        Vector sample = ind.get_pos();
        real dist = (inside_out ? -1 : 1) * (length(center - sample) - radius);
        set(ind, std::min(Array::get(ind), dist));
    }
}

template <>
void LevelSet<2>::add_polygon(std::vector<Vector2> polygon, bool inside_out) {
    for (auto &ind : this->get_region()) {
        Vector2 p = ind.get_pos();
        real dist = ((inside_polygon(p, polygon) ^ inside_out) ? -1 : 1) * (nearest_distance(p, polygon));
        set(ind, std::min(Array::get(ind), dist));
    }
}

template <>
Vector2 LevelSet<2>::get_gradient(const Vector2 &pos) const {
    assert_info(inside(pos),
                "LevelSet Gradient Query out of Bound! (" + std::to_string(pos.x) + ", " + std::to_string(pos.y) + ")");
    real x = pos.x, y = pos.y;
    x = clamp(x - storage_offset.x, 0.f, width - 1.f - eps);
    y = clamp(y - storage_offset.y, 0.f, height - 1.f - eps);
    const int x_i = clamp(int(x), 0, width - 2);
    const int y_i = clamp(int(y), 0, height - 2);
    const real x_r = x - x_i;
    const real y_r = y - y_i;
    const real gx = lerp(y_r, Array::get(x_i + 1, y_i) - Array::get(x_i, y_i),
                         Array::get(x_i + 1, y_i + 1) - Array::get(x_i, y_i + 1));
    const real gy = lerp(x_r, Array::get(x_i, y_i + 1) - Array::get(x_i, y_i),
                         Array::get(x_i + 1, y_i + 1) - Array::get(x_i + 1, y_i));
    return Vector2(gx, gy);
}

template <int DIM>
auto LevelSet<DIM>::get_normalized_gradient(const LevelSet<DIM>::Vector &pos) const {
    Vector gradient = get_gradient(pos);
    if (length(gradient) < 1e-10f)
        gradient[0] = 1.0f;
    return normalize(gradient);
}

template <>
real LevelSet<2>::get(const Vector2 &pos) const {
    assert_info(inside(pos),
                "LevelSet Query out of Bound! (" + std::to_string(pos.x) + ", " + std::to_string(pos.y) + ")");
    real x = pos.x, y = pos.y;
    x = clamp(x - this->storage_offset.x, 0.f, this->res[0] - 1.f - eps);
    y = clamp(y - this->storage_offset.y, 0.f, this->res[1] - 1.f - eps);
    const int x_i = clamp(int(x), 0, this->res[0] - 2);
    const int y_i = clamp(int(y), 0, this->res[1] - 2);
    const real x_r = x - x_i;
    const real y_r = y - y_i;
    const real ly0 = lerp(x_r, Array::get(x_i, y_i), Array::get(x_i + 1, y_i));
    const real ly1 = lerp(x_r, Array::get(x_i, y_i + 1), Array::get(x_i + 1, y_i + 1));
    return lerp(y_r, ly0, ly1);
}

template <>
real LevelSet<3>::get(const Vector3 &pos) const {
    assert_info(inside(pos), "LevelSet Query out of Bound! ("
                             + std::to_string(pos.x) + ", "
                             + std::to_string(pos.y) + ", "
                             + std::to_string(pos.z) + ")");
    real x = pos.x, y = pos.y, z = pos.z;
    x = clamp(x - storage_offset.x, 0.f, width - 1.f - eps);
    y = clamp(y - storage_offset.y, 0.f, height - 1.f - eps);
    z = clamp(z - storage_offset.z, 0.f, depth - 1.f - eps);
    const int x_i = clamp(int(x), 0, width - 2);
    const int y_i = clamp(int(y), 0, height - 2);
    const int z_i = clamp(int(z), 0, depth - 2);
    const real x_r = x - x_i;
    const real y_r = y - y_i;
    const real z_r = z - z_i;
    return lerp(x_r,
                lerp(y_r,
                     lerp(z_r, Array3D<real>::get(x_i, y_i, z_i), Array3D<real>::get(x_i, y_i, z_i + 1)),
                     lerp(z_r, Array3D<real>::get(x_i, y_i + 1, z_i), Array3D<real>::get(x_i, y_i + 1, z_i + 1))),
                lerp(y_r,
                     lerp(z_r, Array3D<real>::get(x_i + 1, y_i, z_i), Array3D<real>::get(x_i + 1, y_i, z_i + 1)),
                     lerp(z_r, Array3D<real>::get(x_i + 1, y_i + 1, z_i),
                          Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1))));
}

template <int DIM>
Array2D<real> LevelSet<DIM>::rasterize(Vector2i output_res) {
    for (auto &p : (*this)) {
        if (std::isnan(p)) {
            printf("Warning: nan in levelset.");
        }
    }
    Array2D<real> out(output_res);
    Vector2 actual_size;
    if (storage_offset == Vector2(0.0f, 0.0f)) {
        actual_size = Vector2(this->width - 1, this->height - 1);
    } else {
        actual_size = Vector2(this->width, this->height);
    }

    Vector2 scale_factor = actual_size / Vector2(output_res);

    for (auto &ind : Region2D(0, width, 0, height, Vector2(0.5f, 0.5f))) {
        Vector2 p = scale_factor * ind.get_pos();
        out[ind] = sample(p);
        if (std::isnan(out[ind])) {
            out[ind] = std::numeric_limits<real>::infinity();
        }
    }
    return out;
}

template <>
Array3D<real> LevelSet<3>::rasterize(Vector3i output_res) {
    for (auto &p : (*this)) {
        if (std::isnan(p)) {
            printf("Warning: nan in levelset.");
        }
    }
    Array3D<real> out(output_res);
    Vector3 actual_size;
    if (storage_offset == Vector3(0.0f, 0.0f, 0.0f)) {
        actual_size = Vector3(this->width - 1, this->height - 1, this->depth - 1);
    } else {
        actual_size = Vector3(this->width, this->height, this->depth);
    }

    Vector3 scale_factor = actual_size / Vector3(output_res);

    for (auto &ind : Region3D(0, width, 0, height, 0, depth, Vector3(0.5f, 0.5f, 0.5f))) {
        Vector3 p = scale_factor * ind.get_pos();
        out[ind] = sample(p);
        if (std::isnan(out[ind])) {
            out[ind] = std::numeric_limits<real>::infinity();
        }
    }
    return out;
}

template <int DIM>
void LevelSet<DIM>::add_plane(const Vector3 &normal, real d) {
    real coeff = 1.0f / length(normal);
    for (auto &ind : get_region()) {
        Vector sample = ind.get_pos();
        real dist = (dot(sample, normal) + d) * coeff;
        set(ind, std::min(Array3D<real>::get(ind), dist));
    }
}

template <>
void LevelSet<3>::add_cuboid(Vector3 lower_boundry, Vector3 upper_boundry, bool inside_out) {
    for (auto &ind : get_region()) {
        Vector3 sample = ind.get_pos();
        bool in_cuboid = true;
        for (int i = 0; i < 3; ++i) {
            if (!(lower_boundry[i] <= sample[i] && sample[i] <= upper_boundry[i]))
                in_cuboid = false;
        }
        real dist = INF;
        if (in_cuboid) {
            for (int i = 0; i < 3; ++i) {
                dist = std::min(dist, std::min(upper_boundry[i] - sample[i], sample[i] - lower_boundry[i]));
            }
        } else {
            Vector3 nearest_p;
            for (int i = 0; i < 3; ++i) {
                nearest_p[i] = clamp(sample[i], lower_boundry[i], upper_boundry[i]);
            }
            dist = -length(nearest_p - sample);
        }
        set(ind, inside_out ? dist : -dist);
    }
}

template <int DIM>
void LevelSet<DIM>::global_increase(real delta) {
    for (auto &ind : get_region()) {
        set(ind, Array::get(ind) + delta);
    }
}

template <>
Vector3 LevelSet<3>::get_gradient(const Vector3 &pos) const {
    assert_info(inside(pos), "LevelSet Gradient Query out of Bound! ("
                             + std::to_string(pos.x) + ", "
                             + std::to_string(pos.y) + ", "
                             + std::to_string(pos.z) + ")");
    real x = pos.x, y = pos.y, z = pos.z;
    x = clamp(x - storage_offset.x, 0.f, width - 1.f - eps);
    y = clamp(y - storage_offset.y, 0.f, height - 1.f - eps);
    z = clamp(z - storage_offset.z, 0.f, depth - 1.f - eps);
    const int x_i = clamp(int(x), 0, width - 2);
    const int y_i = clamp(int(y), 0, height - 2);
    const int z_i = clamp(int(z), 0, depth - 2);
    const real x_r = x - x_i;
    const real y_r = y - y_i;
    const real z_r = z - z_i;
    // TODO: speed this up
    const real gx = lerp(y_r,
                         lerp(z_r, Array3D<real>::get(x_i + 1, y_i, z_i) - Array3D<real>::get(x_i, y_i, z_i),
                              Array3D<real>::get(x_i + 1, y_i, z_i + 1) - Array3D<real>::get(x_i, y_i, z_i + 1)),
                         lerp(z_r, Array3D<real>::get(x_i + 1, y_i + 1, z_i) - Array3D<real>::get(x_i, y_i + 1, z_i),
                              Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1) -
                              Array3D<real>::get(x_i, y_i + 1, z_i + 1)));
    const real gy = lerp(z_r,
                         lerp(x_r, Array3D<real>::get(x_i, y_i + 1, z_i) - Array3D<real>::get(x_i, y_i, z_i),
                              Array3D<real>::get(x_i + 1, y_i + 1, z_i) - Array3D<real>::get(x_i + 1, y_i, z_i)),
                         lerp(x_r, Array3D<real>::get(x_i, y_i + 1, z_i + 1) - Array3D<real>::get(x_i, y_i, z_i + 1),
                              Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1) -
                              Array3D<real>::get(x_i + 1, y_i, z_i + 1)));
    const real gz = lerp(x_r,
                         lerp(y_r, Array3D<real>::get(x_i, y_i, z_i + 1) - Array3D<real>::get(x_i, y_i, z_i),
                              Array3D<real>::get(x_i, y_i + 1, z_i + 1) - Array3D<real>::get(x_i, y_i + 1, z_i)),
                         lerp(y_r, Array3D<real>::get(x_i + 1, y_i, z_i + 1) - Array3D<real>::get(x_i + 1, y_i, z_i),
                              Array3D<real>::get(x_i + 1, y_i + 1, z_i + 1) -
                              Array3D<real>::get(x_i + 1, y_i + 1, z_i)));
    return Vector3(gx, gy, gz);
}

template<> class LevelSet<2>;
template<> class LevelSet<3>;

TC_NAMESPACE_END
