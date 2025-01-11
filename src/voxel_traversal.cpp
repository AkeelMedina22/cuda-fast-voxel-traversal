#include "voxel_traversal.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <iostream>

namespace py = pybind11;

void validate_voxelgrid(const py::array_t<uint64_t> &voxels)
{
    py::buffer_info buf = voxels.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Voxel Grid must be a 1-dimensional array of Morton-Encoded ints");
}

namespace voxel_traversal
{

    py::array_t<float> trace(
        py::array_t<uint64_t> voxels,
        py::array_t<float> ray_starts,
        py::array_t<float> ray_ends,
        float resolution,
        int image_width,
        int image_height)
    {
        validate_voxelgrid(voxels);

        py::buffer_info voxels_buf = voxels.request();
        py::buffer_info camera_buf = ray_starts.request();
        py::buffer_info ray_ends_buf = ray_ends.request();

        uint64_t *voxels_ptr = static_cast<uint64_t *>(voxels_buf.ptr);
        float *ray_starts_ptr = static_cast<float *>(camera_buf.ptr);
        float *ray_ends_ptr = static_cast<float *>(ray_ends.request().ptr);

        size_t num_voxels = voxels_buf.shape[0];

        py::array_t<float> result({image_width, image_height});
        py::buffer_info buf = result.request();
        float *ptr = static_cast<float *>(buf.ptr);

        launchVisibilityCheck(
            ptr,
            voxels_ptr,
            num_voxels,
            ray_starts_ptr,
            ray_ends_ptr,
            resolution,
            image_width,
            image_height);

        return result;
    }

} // namespace voxel_traversal

PYBIND11_MODULE(voxel_traversal, m)
{
    m.doc() = "Fast voxel traversal module for ray tracing";

    m.def("trace", &voxel_traversal::trace,
          py::arg("voxels"),
          py::arg("ray_starts"),
          py::arg("ray_ends"),
          py::arg("resolution"),
          py::arg("image_width"),
          py::arg("image_height"),
          "Ray trace a voxel grid from given camera position");
}