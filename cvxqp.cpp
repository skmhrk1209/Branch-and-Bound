#include <boost/serialization/vector.hpp>
#include <algorithm>
#include "cvxqp.hpp"

// #define DEBUG

namespace cvxqp
{

template <typename DataType>
MixedBooleanQPSolver<DataType>::MixedBooleanQPSolver(const Matrix<DataType> &doublyStochasticMatrix)
{
    std::cout << "MPI initialized (Rank:" << mCommunicator.rank() << ", Size:" << mCommunicator.size() << ")" << std::endl;

    mRows = doublyStochasticMatrix.rows();
    mCols = doublyStochasticMatrix.cols();
    mSize = doublyStochasticMatrix.size();

    Matrix<DataType> permutationMatrix = greedy_search(doublyStochasticMatrix);

    std::vector<DataType> doublyStochasticMatrixVector(mSize);
    Eigen::Map<Matrix<DataType>>(doublyStochasticMatrixVector.data(), mRows, mCols) = doublyStochasticMatrix;

    std::vector<DataType> permutationMatrixVector(mSize);
    Eigen::Map<Matrix<DataType>>(permutationMatrixVector.data(), mRows, mCols) = permutationMatrix;

    boost::mpi::broadcast(mCommunicator, doublyStochasticMatrixVector, 0);
    boost::mpi::broadcast(mCommunicator, permutationMatrixVector, 0);

    mDoublyStochasticMatrix = Eigen::Map<const Matrix<DataType>>(doublyStochasticMatrixVector.data(), mRows, mCols);
    mPermutationMatrix = Eigen::Map<const Matrix<DataType>>(permutationMatrixVector.data(), mRows, mCols);

    mHessianMatrix = SparseMatrix<DataType>(mSize, mSize);
    mHessianMatrix.setIdentity();

    mGradient = -Eigen::Map<const Vector<DataType>>(doublyStochasticMatrix.data(), mSize);

    mSolution = Eigen::Map<const Vector<DataType>>(permutationMatrix.data(), mSize);
    mValue = objective(mSolution);
}

template <typename DataType>
bool MixedBooleanQPSolver<DataType>::synchronize(const Vector<DataType> &solution, const DataType &value, bool endFlag)
{
    std::vector<DataType> solutionVector(solution.size());
    Eigen::Map<Vector<DataType>>(solutionVector.data(), solution.size()) = solution;

    std::vector<std::vector<DataType>> solutionVectors;
    boost::mpi::all_gather(mCommunicator, solutionVector, solutionVectors);

    std::vector<DataType> values;
    boost::mpi::all_gather(mCommunicator, value, values);

    std::vector<int> endFlags;
    boost::mpi::all_gather(mCommunicator, endFlag ? 1 : 0, endFlags);

    auto argmin = std::distance(values.begin(), std::min_element(values.begin(), values.end()));
    if (values[argmin] < mValue)
    {
        mSolution = Eigen::Map<const Vector<DataType>>(solutionVectors[argmin].data(), solution.size());
        mValue = values[argmin];
    }

    return std::all_of(endFlags.begin(), endFlags.end(), [](auto x) { return x; });
}

template <typename DataType>
Matrix<DataType> MixedBooleanQPSolver<DataType>::greedy_search(Matrix<DataType> doublyStochasticMatrix)
{
    Matrix<DataType> permutationMatrix = Matrix<DataType>::Zero(mRows, mCols);

    while ((doublyStochasticMatrix.array() > 0).any())
    {
        typename Matrix<DataType>::Index i, j;
        doublyStochasticMatrix.maxCoeff(&i, &j);

        permutationMatrix(i, j) = 1;
        doublyStochasticMatrix.row(i) = Vector<DataType>::Constant(mCols, -1);
        doublyStochasticMatrix.col(j) = Vector<DataType>::Constant(mRows, -1);
    }

    return permutationMatrix;
}

template <typename DataType>
std::tuple<Matrix<DataType>, Matrix<DataType>> MixedBooleanQPSolver<DataType>::solve()
{
    SparseMatrix<DataType> linearConstraintMatrix = SparseMatrix<DataType>(mSize + mRows + mCols, mSize);
    {
        std::vector<Triplet<DataType>> triplets;
        triplets.reserve(mSize + mRows * mCols + mCols * mRows);

        for (auto i = 0; i < mSize; ++i)
            triplets.emplace_back(i, i, 1);

        for (auto i = 0; i < mRows; ++i)
            for (auto j = 0; j < mCols; ++j)
                triplets.emplace_back(i + mSize, i * mCols + j, 1);

        for (auto i = 0; i < mCols; ++i)
            for (auto j = 0; j < mRows; ++j)
                triplets.emplace_back(i + mSize + mRows, j * mCols + i, 1);

        linearConstraintMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    Vector<DataType> lowerBound(mSize + mRows + mCols);
    lowerBound << Vector<DataType>::Zero(mSize), Vector<DataType>::Ones(mRows), Vector<DataType>::Ones(mCols);

    Vector<DataType> upperBound(mSize + mRows + mCols);
    upperBound << Vector<DataType>::Ones(mSize), Vector<DataType>::Ones(mRows), Vector<DataType>::Ones(mCols);

    Matrix<int> solutionMatrix = Matrix<int>::Constant(mRows, mCols, -1);

    branch(linearConstraintMatrix, lowerBound, upperBound, solutionMatrix);

    while (!synchronize(mSolution, mValue, true))
    {
    }

    mPermutationMatrix = Eigen::Map<const Matrix<DataType>>(mSolution.data(), mRows, mCols);

    return std::make_tuple(mPermutationMatrix, mDoublyStochasticMatrix);
}

template <typename DataType>
void MixedBooleanQPSolver<DataType>::bound(const SparseMatrix<DataType> &linearConstraintMatrix,
                                           const Vector<DataType> &lowerBound,
                                           const Vector<DataType> &upperBound,
                                           const Matrix<int> &solutionMatrix)
{
    if ((solutionMatrix.array() < 0).count() == 1)
    {
        Vector<DataType> solution = Eigen::Map<const Vector<int>>(solutionMatrix.data(), mSize).template cast<DataType>();
        typename Vector<DataType>::Index i, j;
        solution.minCoeff(&i, &j);
        solution(i, j) = 1;

        DataType value = objective(solution);
        synchronize(solution, value, false);
    }
    else
    {
        Vector<DataType> solution = optimize(linearConstraintMatrix, lowerBound, upperBound);
        DataType value = objective(solution);

        if (solution.isApprox(Vector<DataType>(solution.array().round()), 1e-3))
        {
            synchronize(solution, value, false);
        }
        else
        {
            synchronize(mSolution, mValue, false);
            if (value < mValue)
            {
                branch(linearConstraintMatrix, lowerBound, upperBound, solutionMatrix);
            }
        }
    }
}

template <typename DataType>
void MixedBooleanQPSolver<DataType>::branch(const SparseMatrix<DataType> &linearConstraintMatrix,
                                            const Vector<DataType> &loweBound,
                                            const Vector<DataType> &upperBound,
                                            const Matrix<int> &solutionMatrix)
{
    for (auto i = 0; i < mRows; ++i)
    {
        if ((solutionMatrix.row(i).array() < 0).any())
        {
            auto blockSize = mCols / mCommunicator.size();
            auto blockBegin = blockSize * mCommunicator.rank();
            auto blockEnd = mCommunicator.rank() == mCommunicator.size() - 1 ? mCols : blockBegin + blockSize;

            for (auto j = 0; (i == 0 && j >= blockBegin && j < blockEnd) || (i > 0 && j < mCols); ++j)
            {
                if (solutionMatrix(i, j) < 0)
                {
                    Matrix<int> newSolutionMatrix(solutionMatrix);
                    newSolutionMatrix.row(i) = Vector<int>::Zero(mCols);
                    newSolutionMatrix.col(j) = Vector<int>::Zero(mRows);
                    newSolutionMatrix(i, j) = 1;

                    SparseMatrix<DataType> newLinearConstraintMatrix(linearConstraintMatrix);
                    newLinearConstraintMatrix.conservativeResize(linearConstraintMatrix.rows() + mRows + mCols - 1, linearConstraintMatrix.cols());

                    for (auto k = 0; k < mRows; ++k)
                        if (k != i)
                            newLinearConstraintMatrix.insert(k + linearConstraintMatrix.rows(), k * mCols + j) = 1;

                    for (auto l = 0; l < mCols; ++l)
                        if (l != j)
                            newLinearConstraintMatrix.insert(l + linearConstraintMatrix.rows() + mRows - 1, i * mCols + l) = 1;

                    newLinearConstraintMatrix.insert(linearConstraintMatrix.rows() + mRows + mCols - 2, i * mCols + j) = 1;

                    Vector<DataType> newLowerBound(loweBound.size() + mRows + mCols - 1);
                    newLowerBound << loweBound, Vector<DataType>::Zero(mRows - 1), Vector<DataType>::Zero(mCols - 1), Vector<DataType>::Ones(1);

                    Vector<DataType> newUpperBound(upperBound.size() + mRows + mCols - 1);
                    newUpperBound << upperBound, Vector<DataType>::Zero(mRows - 1), Vector<DataType>::Zero(mCols - 1), Vector<DataType>::Ones(1);

                    bound(newLinearConstraintMatrix, newLowerBound, newUpperBound, newSolutionMatrix);
                }
            }
            break;
        }
    }
}

template <typename DataType>
Vector<DataType> MixedBooleanQPSolver<DataType>::optimize(const SparseMatrix<DataType> &linearConstraintMatrix,
                                                          const Vector<DataType> &lowerBound,
                                                          const Vector<DataType> &upperBound)
{
    Vector<double> doubleGradient = mGradient.template cast<double>();
    Vector<double> doubleLowerBound = lowerBound.template cast<double>();
    Vector<double> doubleUpperBound = upperBound.template cast<double>();

    OsqpEigen::Solver solver;

    solver.settings()->setVerbosity(false);
    solver.data()->setNumberOfVariables(linearConstraintMatrix.cols());
    solver.data()->setNumberOfConstraints(linearConstraintMatrix.rows());

    if (!solver.data()->setHessianMatrix(mHessianMatrix))
        std::cout << "Setting hessian matrix failed." << std::endl;
    if (!solver.data()->setGradient(doubleGradient))
        std::cout << "Setting gradient failed." << std::endl;
    if (!solver.data()->setLinearConstraintsMatrix(linearConstraintMatrix))
        std::cout << "Setting linesr constraint matrix failed." << std::endl;
    if (!solver.data()->setLowerBound(doubleLowerBound))
        std::cout << "Setting lowe bound failed." << std::endl;
    if (!solver.data()->setUpperBound(doubleUpperBound))
        std::cout << "Setting upper bound failed." << std::endl;

    if (!solver.initSolver())
        std::cout << "Initializing solver failed." << std::endl;
    if (!solver.solve())
        std::cout << "Solving problem failed." << std::endl;

    Vector<DataType> solution = solver.getSolution().template cast<DataType>();

    return solution;
}

template <typename DataType>
DataType MixedBooleanQPSolver<DataType>::objective(const Vector<DataType> &solution)
{
    DataType quadraticTerm = 0.5 * solution.transpose() * mHessianMatrix * solution;
    DataType linearTerm = mGradient.transpose() * solution;
    return quadraticTerm + linearTerm;
}

} // namespace cvxqp

template cvxqp::MixedBooleanQPSolver<float>::MixedBooleanQPSolver(const Matrix<float> &);
template cvxqp::MixedBooleanQPSolver<double>::MixedBooleanQPSolver(const Matrix<double> &);

template bool cvxqp::MixedBooleanQPSolver<float>::synchronize(const cvxqp::Vector<float> &solution, const float &value, bool finished);
template bool cvxqp::MixedBooleanQPSolver<double>::synchronize(const cvxqp::Vector<double> &solution, const double &value, bool finished);

template cvxqp::Matrix<float> cvxqp::MixedBooleanQPSolver<float>::greedy_search(cvxqp::Matrix<float>);
template cvxqp::Matrix<double> cvxqp::MixedBooleanQPSolver<double>::greedy_search(cvxqp::Matrix<double>);

template std::tuple<cvxqp::Matrix<float>, cvxqp::Matrix<float>> cvxqp::MixedBooleanQPSolver<float>::solve();
template std::tuple<cvxqp::Matrix<double>, cvxqp::Matrix<double>> cvxqp::MixedBooleanQPSolver<double>::solve();

template void cvxqp::MixedBooleanQPSolver<float>::bound(const cvxqp::SparseMatrix<float> &,
                                                        const cvxqp::Vector<float> &,
                                                        const cvxqp::Vector<float> &,
                                                        const cvxqp::Matrix<int> &);
template void cvxqp::MixedBooleanQPSolver<double>::bound(const cvxqp::SparseMatrix<double> &,
                                                         const cvxqp::Vector<double> &,
                                                         const cvxqp::Vector<double> &,
                                                         const cvxqp::Matrix<int> &);

template void cvxqp::MixedBooleanQPSolver<float>::branch(const cvxqp::SparseMatrix<float> &,
                                                         const cvxqp::Vector<float> &,
                                                         const cvxqp::Vector<float> &,
                                                         const cvxqp::Matrix<int> &);
template void cvxqp::MixedBooleanQPSolver<double>::branch(const cvxqp::SparseMatrix<double> &,
                                                          const cvxqp::Vector<double> &,
                                                          const cvxqp::Vector<double> &,
                                                          const cvxqp::Matrix<int> &);

template cvxqp::Vector<float> cvxqp::MixedBooleanQPSolver<float>::optimize(const cvxqp::SparseMatrix<float> &,
                                                                           const cvxqp::Vector<float> &,
                                                                           const cvxqp::Vector<float> &);
template cvxqp::Vector<double> cvxqp::MixedBooleanQPSolver<double>::optimize(const cvxqp::SparseMatrix<double> &,
                                                                             const cvxqp::Vector<double> &,
                                                                             const cvxqp::Vector<double> &);

template float cvxqp::MixedBooleanQPSolver<float>::objective(const cvxqp::Vector<float> &);
template double cvxqp::MixedBooleanQPSolver<double>::objective(const cvxqp::Vector<double> &);
