import pycvxqp
import cvxqp
import cvxpy
import numpy as np
import time

def optimize(y):
    x = cvxpy.Variable(y.shape, boolean=True)
    objective = cvxpy.Minimize(cvxpy.sum_squares(x - y))
    constraints = [
        np.ones(x.shape[0]) @ x == 1,
        x @ np.ones(x.shape[1]) == 1,
    ]
    problem = cvxpy.Problem(objective, constraints)
    problem.solve()
    return x.value

x = np.random.randn(10, 10) / 0.1
x = np.exp(x) / np.sum(np.exp(x), keepdims=True)
x /= np.matmul(np.ones_like(x), x)
x /= np.matmul(x, np.ones_like(x))
t1 = time.time()
y, x = pycvxqp.branch_and_bound(x)
t2 = time.time()
l = np.sum((x - y) ** 2)
print('pycvxqp', t2 - t1, l)
t2 = time.time()
y = cvxqp.branch_and_bound(x)
t3 = time.time()
l = np.sum((x - y) ** 2)
print('cvxqp', t3 - t2, l)
t3 = time.time()
y = optimize(x)
t4 = time.time()
l = np.sum((x - y) ** 2)
print('cvxpy', t4 - t3, l)
