#include <Eigen/Dense>
#include "cvxqp.hpp"
#include "iostream.hpp"
#include "functional.hpp"


int main(...)
{
    cvxqp::MixedBooleanQPSolver<double> solver(Eigen::MatrixXd::Identity(4, 4));
    auto solution = solver.solve();
    std::cout << solution << std::endl;
    return 0;
}
