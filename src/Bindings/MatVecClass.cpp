#include "Bindings/Common.hpp"

#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

void bindMatVecClass(nb::module_& m)
{
	nb::class_<MatVec<double>>(m, "matvec")
		.def(nb::init<const nb::ndarray<double, nb::c_contig>&>())
		// .def("toStringVerbose", &MatVec<double>::toStringVerbose)
		.def("__repr__", [](const MatVec<double>& self) {
			return self.toStringVerbose();
		})
		.def("__str__", [](const MatVec<double>& self) {
			return self.toStringData();
		})
		.def("toNumPy", &MatVec<double>::toNumPy)
		.def_prop_ro("shape", &MatVec<double>::shapeToPythonTuple)
		;
}