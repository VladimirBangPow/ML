#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "dataframe.h"  // Your DataFrame library
#include "tensor.h"     // Our tensor module
#include "lr.h"         // Linear regression functions (train_linear_regression, etc.)

int main(void) {
    // 1) Create a DataFrame object & read CSV
    DataFrame df;
    DataFrame_Create(&df);

    const char* csvFile = "../data/btcusd.csv";
    bool success = df.readCsv(&df, csvFile);
    if (!success) {
        fprintf(stderr, "Failed to read CSV: %s\n", csvFile);
        // Cleanup & exit
        DataFrame_Destroy(&df);
        return 1;
    }

    // 2) Let's check how many rows and columns we got
    size_t n = df.numRows(&df);
    size_t c = df.numColumns(&df);
    if (n == 0) {
        fprintf(stderr, "CSV has no data.\n");
        DataFrame_Destroy(&df);
        return 1;
    }
    printf("Loaded DataFrame with %zu rows, %zu columns.\n", n, c);

    // We'll assume columns are in this order:
    // time(0), open(1), close(2), high(3), low(4), volume(5)
    // If the order is different, adjust the colIndex accordingly.
    size_t openColIndex = 1;
    size_t closeColIndex = 2;


    // 3) Build Tensors for X (open) and y (close) => shape=[n,1]
    size_t shape[2] = { n, 1 };
    Tensor *X = tensor_create(2, shape, TENSOR_FLOAT64);
    Tensor *y = tensor_create(2, shape, TENSOR_FLOAT64);
    if (!X || !y) {
        fprintf(stderr, "Failed to create X or y tensor.\n");
        DataFrame_Destroy(&df);
        return 1;
    }

    // 4) For each row, retrieve open & close, store in X, y
    //    DataFrame's getRow() presumably returns an array of pointers (one per col).
    //    We'll NOT free them unless the DataFrame doc explicitly says we should.
    for (size_t i = 0; i < n; i++) {
        void** rowBufPtr = NULL;  // only two stars
        bool gotRow = df.getRow(&df, i, &rowBufPtr);
        if (!gotRow || !rowBufPtr) {
            fprintf(stderr, "Failed to get row %zu\n", i);
            continue;
        }
        // Declare your double variables here:
        double openVal = 0.0;
        double closeVal = 0.0;

        
        // Index is rowBufPtr[colIndex], which is a 'void*'
        if (rowBufPtr[openColIndex]) {
            openVal = *((double*) rowBufPtr[openColIndex]);
        }
        if (rowBufPtr[closeColIndex]) {
            closeVal = *((double*) rowBufPtr[closeColIndex]);
        }

        // Now store them in X, y
        size_t idx[2] = { i, 0 };
        tensor_set(X, idx, openVal);
        tensor_set(y, idx, closeVal);

    }

    printf("Loaded %zu rows of data.\n", n);

    // 5) Create W, b for the model => shape of W = [1,1], b=[1]
    size_t shapeW[2] = { 1, 1 };
    Tensor *W = tensor_create(2, shapeW, TENSOR_FLOAT64);
    size_t shapeB[1] = { 1 };
    Tensor *b = tensor_create(1, shapeB, TENSOR_FLOAT64);
    // They start at zero. If you prefer random init, you can set them.

    // 6) Train the linear model: close ≈ W * open + b
    double lr = 1e-5;
    int epochs = 2000;
    printf("Training linear regressor with %d epochs, LR=%.4f...\n", epochs, lr);
    train_linear_regression(X, y, W, b, lr, epochs, 1 /*verbose*/);

    // Print final W,b
    double W_val = tensor_get(W, (size_t[]){0,0});
    double b_val = tensor_get(b, (size_t[]){0});
    printf("Learned model: close = %.5f * open + %.5f\n", W_val, b_val);

    // 7) Optional test: Let's predict the close price for the last row's open
    // For example, the last row
    size_t lastRow = n - 1;
    double openLast = tensor_get(X, (size_t[]){lastRow, 0});
    double predictedClose = W_val * openLast + b_val;
    double actualClose    = tensor_get(y, (size_t[]){lastRow, 0});
    printf("Last row => open=%.2f, predicted close=%.2f, actual close=%.2f\n",
           openLast, predictedClose, actualClose);

    // 8) Cleanup
    tensor_free(X);
    tensor_free(y);
    tensor_free(W);
    tensor_free(b);
    DataFrame_Destroy(&df);

    return 0;
}
