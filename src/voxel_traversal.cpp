#include "voxel_traversal.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void validate_pointcloud(const py::array_t<double> &points)
{
    py::buffer_info buf = points.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Point cloud must be 2-dimensional");
    if (buf.shape[1] != 3)
        throw std::runtime_error("Points must be 3D (shape: N x 3)");
}

void validate_camera_pos(const py::array_t<double> &camera_pos)
{
    py::buffer_info buf = camera_pos.request();
    if (buf.ndim != 1 || buf.shape[0] != 3)
        throw std::runtime_error("Camera position must be a 3D point (shape: 3)");
}

namespace py = pybind11;
namespace voxel_traversal
{

    py::array_t<double> create_point_cloud( // Changed return type to array
        py::array_t<double> points,
        py::array_t<double> camera_pos,
        int image_width,
        int image_height,
        double focal_length = 1.0,
        double sensor_width = 36.0,
        double sensor_height = 24.0)
    {
        // Validate inputs
        validate_pointcloud(points);
        validate_camera_pos(camera_pos);

        // Get raw pointers to the numpy array data
        py::buffer_info points_buf = points.request();
        py::buffer_info camera_buf = camera_pos.request();

        double *points_ptr = static_cast<double *>(points_buf.ptr);
        double *camera_ptr = static_cast<double *>(camera_buf.ptr);
        size_t num_points = points_buf.shape[0];

        // Create output buffer
        std::vector<double> output_data(image_width * image_height, 0.0); // Initialize with zeros

        // TODO: Your actual computation here
        // This would modify output_data

        // Create output numpy array
        std::vector<ssize_t> shape = {image_height, image_width};
        std::vector<ssize_t> strides = {
            static_cast<ssize_t>(image_width * sizeof(double)),
            static_cast<ssize_t>(sizeof(double))};

        // Return array with copy of our data
        return py::array_t<double>(
            shape,
            strides,
            output_data.data() // Using our output buffer
        );
    }
} // namespace voxel_traversal

PYBIND11_MODULE(voxel_traversal, m)
{
    m.doc() = "Fast voxel traversal module for ray tracing";

    m.def("create_point_cloud", &voxel_traversal::create_point_cloud,
          py::arg("points"),
          py::arg("camera_pos"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("focal_length") = 1.0,
          py::arg("sensor_width") = 36.0,
          py::arg("sensor_height") = 24.0,
          "Ray trace a point cloud from given camera position");
}