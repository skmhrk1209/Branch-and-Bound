#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include "cvxqp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pycvxqp, module)
{
    module.def("branch_and_bound",
               [](const cvxqp::Matrix<double> &doublyStochasticMatrix) {
                   boost::mpi::environment environment;
                   cvxqp::MixedBooleanQPSolver<double> solver(environment);
                   return solver.solve(doublyStochasticMatrix);
               },
               py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
}
