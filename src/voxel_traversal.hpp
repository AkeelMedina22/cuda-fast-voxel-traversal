#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

struct PointCloud
{
    std::vector<double> points; // Flattened xyz points
    size_t num_points;

    PointCloud(const double *data, size_t n_points)
        : points(data, data + n_points * 3), num_points(n_points) {}
};