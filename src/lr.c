#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lr.h"
#include "tensor.h"  // <-- Ensure we include "tensor.h" so we know about tensor_*()

/**
 * Forward pass for linear regression: y_pred = X * W + b.
 * Returns a newly allocated tensor of shape [n, 1].
 */
Tensor* linear_forward(const Tensor *X, const Tensor *W, const Tensor *b) {
    // 1) out = matmul(X, W) => shape=[n,1]
    Tensor *out = tensor_matmul(X, W);
    if (!out) {
        fprintf(stderr, "[linear_forward] matmul returned NULL.\n");
        return NULL;
    }
    // 2) out = out + b (broadcast the bias) => shape=[n,1]
    Tensor *out_plus_b = tensor_add(out, b);
    tensor_free(out); // free intermediate
    return out_plus_b;
}

/**
 * MSE loss = mean( (y_pred - y)^2 ).
 *   y_pred, y => shape=[n,1].
 */
double mse_loss(const Tensor *y_pred, const Tensor *y) {
    // 1) diff = (y_pred - y)
    Tensor *diff = tensor_sub(y_pred, y);
    
    // 2) square = diff * diff (element-wise)
    Tensor *square = tensor_mul(diff, diff);
    
    // 3) mean of square
    double sum_sq = tensor_sum(square);
    size_t n = diff->shape[0]; // for shape [n,1]
    double mse = sum_sq / (double)n;
    
    // cleanup
    tensor_free(diff);
    tensor_free(square);
    
    return mse;
}

/**
 * Train a linear regressor y_pred = X*W + b using gradient descent on MSE.
 *
 * We'll do:
 *   y_pred = X*W + b
 *   loss = mean((y - y_pred)^2)
 *
 * Gradient wrt W: dW = (2/n) * X^T * (XW + b - y)
 * Gradient wrt b: db = (2/n) * sum( (XW + b - y) )
 *
 * We'll implement a naive approach using the tensor ops we have, 
 * ignoring in-place for clarity. 
 */
void train_linear_regression(
    const Tensor *X,
    const Tensor *y,
    Tensor *W,
    Tensor *b,
    double lr,
    int epochs,
    int verbose
) {
    // Basic checks omitted for brevity (see your original code).
    // ...

    size_t n = X->shape[0];
    size_t d = X->shape[1];

    for (int e = 0; e < epochs; e++) {
        // (1) Forward pass
        Tensor *y_pred = linear_forward(X, W, b); // shape=[n,1]
        
        // (2) Compute loss
        double loss_val = mse_loss(y_pred, y);
        
        // (3) Gradients:
        // diff = (y_pred - y)
        Tensor *diff = tensor_sub(y_pred, y);
        
        double scale = 2.0 / (double)n;
        double sum_diff = tensor_sum(diff);
        double grad_b_val = scale * sum_diff; // scalar grad wrt b
        
        // Build X^T => shape=[d,n]
        size_t shapeXT[2] = { d, n };
        Tensor *X_trans = tensor_create(2, shapeXT, X->dtype);
        if (!X_trans) {
            fprintf(stderr, "[train_linear_regression] Failed to alloc X_trans.\n");
            tensor_free(y_pred);
            tensor_free(diff);
            return;
        }
        // fill X_trans
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                size_t idxX[2] = { i, j };
                double val = tensor_get(X, idxX);
                
                size_t idxXT[2] = { j, i };
                tensor_set(X_trans, idxXT, val);
            }
        }
        
        // grad_w_raw = X^T * diff => shape=[d,1]
        Tensor *grad_w_raw = tensor_matmul(X_trans, diff);
        
        // scale each element in grad_w_raw => multiply by scale
        // We'll do a naive loop over its elements:
        for (size_t i = 0; i < grad_w_raw->num_elems; i++) {
            double old_val = tensor_read_at_offset(grad_w_raw, i);
            double new_val = old_val * scale;
            tensor_write_at_offset(grad_w_raw, i, new_val);
        }
        
        // (4) Update W, b
        // W := W - lr * grad_w_raw
        for (size_t i = 0; i < W->num_elems; i++) {
            double wv = tensor_read_at_offset(W, i);
            double gw = tensor_read_at_offset(grad_w_raw, i);
            wv -= (lr * gw);
            tensor_write_at_offset(W, i, wv);
        }
        // b := b - lr * grad_b_val
        double b_old = tensor_read_at_offset(b, 0); // offset=0 for shape=[1]
        double b_new = b_old - (lr * grad_b_val);
        tensor_write_at_offset(b, 0, b_new);
        
        // (5) Print progress if desired
        if (verbose && (e % 100 == 0 || e == epochs - 1)) {
            printf("Epoch %d, Loss = %.6f\n", e, loss_val);
        }
        
        // cleanup
        tensor_free(y_pred);
        tensor_free(diff);
        tensor_free(X_trans);
        tensor_free(grad_w_raw);
    }
}
