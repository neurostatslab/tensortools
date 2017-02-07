
def _least_squares(A, B, warm_start=None):
    return np.linalg.solve(A, B).T

def _lasso(A, B, warm_start=None):
    