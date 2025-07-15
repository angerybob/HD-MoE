from skopt import gp_minimize
from skopt.space import Real
from scipy.optimize import linear_sum_assignment

def objective(x):
    print(x[0],x[0]**2)
    return x[0]**2

space = [Real(-2,2)]
max_iter=20
initial_params=[-1]
result = gp_minimize(
    objective,
    space,
    n_calls=max_iter,
    x0=initial_params,
    random_state=None,
    n_initial_points=min(10, max_iter),  # 初始评估点数
    verbose=True
)