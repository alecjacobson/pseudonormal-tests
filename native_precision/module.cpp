#include <nanobind/nanobind.h>
namespace nb = nanobind;
extern void bind_signed_distance(nb::module_ &m);
NB_MODULE(PRECISION_MODULE, m)
{
  bind_signed_distance(m);
}
