# Only the mean squared error
def eval_mse(y, yhat):
    m = len(y)
    err = 0.0
    for i in range(m):
        err += (y[i] - yhat[i]) ** 2
    err /= 2 * m

    return err
