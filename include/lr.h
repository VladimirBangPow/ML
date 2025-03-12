#ifndef LR_H
#define LR_H

#include "tensor.h"

/**
 * Trains a linear regressor of the form y_pred = X * W + b
 * using simple batch gradient descent on MSE.
 *
 * @param X        Input features, shape = [n, d]
 * @param y        Target values, shape = [n, 1]
 * @param W        Weights, shape = [d, 1] (initialized externally)
 * @param b        Bias, shape = [1] (scalar) (initialized externally)
 * @param lr       Learning rate (e.g., 0.01)
 * @param epochs   Number of training epochs (e.g., 1000)
 * @param verbose  If nonzero, prints loss every few iterations
 */
void train_linear_regression(
    const Tensor *X,
    const Tensor *y,
    Tensor *W,
    Tensor *b,
    double lr,
    int epochs,
    int verbose
);

/**
 * Computes mean squared error (MSE) = mean( (y_pred - y)^2 )
 * @param y_pred shape = [n, 1]
 * @param y      shape = [n, 1]
 * @return MSE as double
 */
double mse_loss(const Tensor *y_pred, const Tensor *y);

/**
 * Forward pass for linear regression: out = matmul(X, W) + b
 *   X: shape=[n,d], W: shape=[d,1], b: shape=[1]
 * returns a new Tensor of shape=[n,1]
 */
Tensor* linear_forward(const Tensor *X, const Tensor *W, const Tensor *b);

#endif /* LR_H */
