#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "test_lr.h"
#include "dataframe.h"   // your DataFrame library
#include "tensor.h"      // your Tensor module
#include "lr.h"          // linear regression functions

int test_linear_regression(void)
{
    // 1) Create & read CSV
    DataFrame df;
    DataFrame_Create(&df);

    const char* csvFile = "../data/btcusd.csv";
    bool success = df.readCsv(&df, csvFile);
    if (!success) {
        fprintf(stderr, "Failed to read CSV: %s\n", csvFile);
        DataFrame_Destroy(&df);
        return 1; 
    }

    size_t n = df.numRows(&df);
    size_t c = df.numColumns(&df);
    if (n == 0) {
        fprintf(stderr, "CSV has no data.\n");
        DataFrame_Destroy(&df);
        return 1;
    }
    printf("Loaded DataFrame with %zu rows, %zu columns.\n", n, c);

    // 2) Confirm we have enough columns for open(1) + close(2)
    if (c < 3) {
        fprintf(stderr, "Not enough columns (need >=3)!\n");
        DataFrame_Destroy(&df);
        return 1;
    }

    // 3) Create Tensors for X, y
    size_t shape[2] = { n, 1 };
    Tensor *X = tensor_create(2, shape, TENSOR_FLOAT64);
    Tensor *y = tensor_create(2, shape, TENSOR_FLOAT64);
    if (!X || !y) {
        fprintf(stderr, "Failed to create X or y.\n");
        DataFrame_Destroy(&df);
        return 1;
    }

    // 4) Fill X, y from columns open(1), close(2)
    size_t openColIndex  = 1;
    size_t closeColIndex = 2;
    for (size_t i = 0; i < n; i++) {
        void** rowBuf = NULL;
        bool gotRow = df.getRow(&df, i, &rowBuf);
        if (!gotRow || !rowBuf) {
            fprintf(stderr, "Failed to get row %zu\n", i);
            continue;
        }

        double openVal = 0.0;
        double closeVal = 0.0;

        if (rowBuf[openColIndex]) {
            openVal = *((double*)rowBuf[openColIndex]);
        }
        if (rowBuf[closeColIndex]) {
            closeVal = *((double*)rowBuf[closeColIndex]);
        }

        size_t idx[2] = { i, 0 };
        tensor_set(X, idx, openVal);
        tensor_set(y, idx, closeVal);

        // if your DataFrame wants you to free rowBuf, do so
        // but only if the library explicitly instructs it
        // free(rowBuf);
    }
    printf("Loaded %zu rows of data.\n", n);

    // 5) Create weights & bias
    size_t shapeW[2] = { 1, 1 };
    Tensor *W = tensor_create(2, shapeW, TENSOR_FLOAT64);
    Tensor *b = tensor_create(1, (size_t[]){1}, TENSOR_FLOAT64);

    // 6) Train
    double lr = 1e-5;  // small LR to avoid inf/nan
    int epochs = 2000;
    printf("Training linear regressor with %d epochs, LR=%.4f...\n", epochs, lr);
    train_linear_regression(X, y, W, b, lr, epochs, /*verbose=*/1);

    // 7) Print final W,b
    double W_val = tensor_get(W, (size_t[]){0,0});
    double b_val = tensor_get(b, (size_t[]){0});
    printf("Learned model: close = %.5f * open + %.5f\n", W_val, b_val);

    // 8) Quick test on last row
    size_t lastRow = n - 1;
    double openLast = tensor_get(X, (size_t[]){lastRow, 0});
    double predictedClose = (W_val * openLast) + b_val;
    double actualClose    = tensor_get(y, (size_t[]){lastRow, 0});
    printf("Last row => open=%.2f, predicted close=%.2f, actual close=%.2f\n",
           openLast, predictedClose, actualClose);

    // 9) Cleanup
    tensor_free(X);
    tensor_free(y);
    tensor_free(W);
    tensor_free(b);
    DataFrame_Destroy(&df);

    return 0;
}
