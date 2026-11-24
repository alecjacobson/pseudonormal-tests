#include <nanobind/nanobind.h>
namespace nb = nanobind;
extern void bind_signed_distance(nb::module_ &m);
#ifdef PRECISION_INCLUDE_WINDING
extern void bind_fast_winding_number(nb::module_ &m);
#endif
NB_MODULE(PRECISION_MODULE, m)
{
  bind_signed_distance(m);
#ifdef PRECISION_INCLUDE_WINDING
  bind_fast_winding_number(m);
#endif
}
