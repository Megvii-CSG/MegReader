#include "ctc2d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ctc2d_forward", &ctc2d_forward, "ctc2d_forward (cuda)");
  m.def("ctc2d_backward", &ctc2d_backward, "ctc2d_backward (cuda)");
}