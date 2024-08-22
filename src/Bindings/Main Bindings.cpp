#include "core/MatVecCore.hpp"
#include <nanobind/stl/string.h>
#include "Bindings/Common.hpp"
namespace nb = nanobind;

NB_MODULE(MatVec, m)
{
	bindMatVecClass(m);
    // nb::class_<MatVec<double>>(m, "matvec")
    //     .def(nb::init<const nb::ndarray<double, nb::c_contig>&>())
    //     // .def("toStringVerbose", &MatVec<double>::toStringVerbose)
    //     .def("__repr__", [](const MatVec<double>& self) {
    //         return self.toStringVerbose();
    //     })
    //     .def("__str__", [](const MatVec<double>& self) {
    //         return self.toStringData();
    //     })
	// 	.def("toNumPy", &MatVec<double>::toNumPy);
}