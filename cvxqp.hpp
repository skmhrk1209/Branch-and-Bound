#pragma once

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>

namespace cvxqp
{
template <typename DataType>
using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename DataType>
using Vector = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

template <typename DataType>
using SparseMatrix = Eigen::SparseMatrix<DataType>;

template <typename DataType>
using Triplet = Eigen::Triplet<DataType>;

template <typename DataType>
class MixedBooleanQPSolver
{
public:
    MixedBooleanQPSolver(boost::mpi::environment &);

    std::tuple<Matrix<DataType>, Matrix<DataType>> solve(const Matrix<DataType> &);

protected:
    void initialize(const Matrix<DataType> &);

    Matrix<DataType> greedy_search(Matrix<DataType>);

    void bound(const SparseMatrix<DataType> &,
               const Vector<DataType> &,
               const Vector<DataType> &,
               const Matrix<int> &,
               Matrix<int>::Index,
               std::vector<int> &);

    void branch(const SparseMatrix<DataType> &,
                const Vector<DataType> &,
                const Vector<DataType> &,
                const Matrix<int> &,
                Matrix<int>::Index,
                std::vector<int> &);

    Vector<DataType> optimize(const SparseMatrix<DataType> &,
                              const Vector<DataType> &,
                              const Vector<DataType> &);

    DataType objective(const Vector<DataType> &);

    bool synchronize(const Vector<DataType> &,
                     const DataType &,
                     bool);

    boost::mpi::environment &mEnvironment;
    boost::mpi::communicator mCommunicator;

    typename Matrix<DataType>::Index mRows;
    typename Matrix<DataType>::Index mCols;
    typename Matrix<DataType>::Index mSize;

    Matrix<DataType> mDoublyStochasticMatrix;
    Matrix<DataType> mPermutationMatrix;

    SparseMatrix<DataType> mHessianMatrix;
    Vector<DataType> mGradient;

    SparseMatrix<DataType> mLinearConstraintMatrix;
    Vector<DataType> mLowerBound;
    Vector<DataType> mUpperBound;

    Vector<DataType> mSolution;
    DataType mValue;
};
} // namespace cvxqp
