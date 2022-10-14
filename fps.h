#include <torch/extension.h>

void fps_launcher(torch::Tensor points,
                  torch::Tensor dists,
                  torch::Tensor indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fps", &fps_launcher);
}