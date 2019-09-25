from mpi4py import MPI
import pycvxqp
import cvxpy
import numpy as np
import time


def solve(doubly_stochastic_matrix):
    permutation_matrix = cvxpy.Variable(doubly_stochastic_matrix.shape, boolean=True)
    objective = cvxpy.Minimize(cvxpy.sum_squares(permutation_matrix - doubly_stochastic_matrix))
    constraints = [
        np.ones(permutation_matrix.shape[0]) @ permutation_matrix == 1,
        permutation_matrix @ np.ones(permutation_matrix.shape[1]) == 1,
    ]
    problem = cvxpy.Problem(objective, constraints)
    problem.solve()
    return permutation_matrix.value, doubly_stochastic_matrix


np.random.seed(0)

for i in range(10):

    doubly_stochastic_matrix = np.random.randn(9, 9) / 0.1
    doubly_stochastic_matrix = np.exp(doubly_stochastic_matrix) / np.sum(np.exp(doubly_stochastic_matrix), keepdims=True)
    doubly_stochastic_matrix /= np.matmul(np.ones_like(doubly_stochastic_matrix), doubly_stochastic_matrix)
    doubly_stochastic_matrix /= np.matmul(doubly_stochastic_matrix, np.ones_like(doubly_stochastic_matrix))

    begin = time.time()
    permutation_matrix, doubly_stochastic_matrix = pycvxqp.branch_and_bound(doubly_stochastic_matrix)
    end = time.time()
    loss = np.sum((permutation_matrix - doubly_stochastic_matrix) ** 2)
    if not MPI.COMM_WORLD.Get_rank():
        print(f'pycvxqp: {loss}({end - begin}s)')

    begin = time.time()
    permutation_matrix, doubly_stochastic_matrix = solve(doubly_stochastic_matrix)
    end = time.time()
    loss = np.sum((permutation_matrix - doubly_stochastic_matrix) ** 2)
    if not MPI.COMM_WORLD.Get_rank():
        print(f'cvxpy: {loss}({end - begin}s)')
