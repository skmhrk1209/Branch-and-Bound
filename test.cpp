#include "cvxqp.hpp"
#include "range.hpp"
#include "iostream.hpp"

int main(...)
{
    boost::mpi::environment environment;

    for (auto i : range<10>())
    {
        cvxqp::Matrix<float> doublyStochasticMatrix = cvxqp::Matrix<float>::Random(9, 9) / 0.1;
        cvxqp::Matrix<float> permutationMatrix;
        // Softmax operation
        doublyStochasticMatrix = doublyStochasticMatrix.array().exp() / doublyStochasticMatrix.array().exp().sum();
        // Sinkhorn Normalization
        doublyStochasticMatrix = doublyStochasticMatrix.array() / (cvxqp::Matrix<float>::Ones(9, 9) * doublyStochasticMatrix).array();
        doublyStochasticMatrix = doublyStochasticMatrix.array() / (doublyStochasticMatrix * cvxqp::Matrix<float>::Ones(9, 9)).array();
        // Solve!
        cvxqp::MixedBooleanQPSolver<float> solver(environment);
        std::tie(permutationMatrix, doublyStochasticMatrix) = solver.solve(doublyStochasticMatrix);

        std::cout << "doublyStochasticMatrix:\n"
                  << doublyStochasticMatrix << std::endl;
        std::cout << "permutationMatrix:\n"
                  << permutationMatrix << std::endl;
    }

    return 0;
}
