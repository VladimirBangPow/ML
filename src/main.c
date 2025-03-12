#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "test_lr.h"

int main(void) {
    int status = test_linear_regression();
    if (status == 0) {
        printf("All tests passed.\n");
    } else {
        printf("Some tests failed.\n");
    }
    return 0;
}
