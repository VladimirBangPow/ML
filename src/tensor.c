#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ------------------------------------------------------------------------- */
/*                           HELPER FUNCTIONS                                */
/* ------------------------------------------------------------------------- */

/** Compute total number of elements = product of shape dimensions. */
static size_t compute_num_elems(size_t ndim, const size_t *shape) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        total *= shape[i];
    }
    return total;
}

/** Compute strides for a row-major contiguous layout. */
static void compute_strides(size_t ndim, const size_t *shape, size_t *strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

/** Returns the size in bytes for one element of the given data type. */
static size_t dtype_size(TensorDtype dtype) {
    switch (dtype) {
        case TENSOR_FLOAT32: return sizeof(float);
        case TENSOR_FLOAT64: return sizeof(double);
        case TENSOR_INT32:   return sizeof(int);
        default:
            fprintf(stderr, "Unsupported dtype\n");
            return 0;
    }
}

/* ------------------------------------------------------------------------- */
/*              PUBLIC: Low-Level Offset-Based Read/Write                    */
/* ------------------------------------------------------------------------- */

/** Convert a value from the tensor's dtype at offset to double. */
double tensor_read_at_offset(const Tensor *t, size_t offset) {
    if (!t) {
        fprintf(stderr, "[tensor_read_at_offset] Null tensor.\n");
        return 0.0;
    }
    switch (t->dtype) {
        case TENSOR_FLOAT32:
            return (double)((float*)t->data)[offset];
        case TENSOR_FLOAT64:
            return ((double*)t->data)[offset];
        case TENSOR_INT32:
            return (double)((int*)t->data)[offset];
        default:
            return 0.0;
    }
}

/** Convert a double value to the tensor's dtype and store at offset. */
void tensor_write_at_offset(Tensor *t, size_t offset, double value) {
    if (!t) {
        fprintf(stderr, "[tensor_write_at_offset] Null tensor.\n");
        return;
    }
    switch (t->dtype) {
        case TENSOR_FLOAT32:
            ((float*)t->data)[offset] = (float)value;
            break;
        case TENSOR_FLOAT64:
            ((double*)t->data)[offset] = (double)value;
            break;
        case TENSOR_INT32:
            ((int*)t->data)[offset] = (int)value;
            break;
        default:
            // unsupported dtype
            break;
    }
}

/* ------------------------------------------------------------------------- */
/*                        BASIC TENSOR LIFECYCLE                             */
/* ------------------------------------------------------------------------- */

Tensor* tensor_create(size_t ndim, const size_t *shape, TensorDtype dtype) {
    // Allocate the Tensor struct
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "Failed to allocate Tensor struct.\n");
        return NULL;
    }

    t->ndim = ndim;
    t->dtype = dtype;
    t->owner = 1;      // By default, this tensor owns its data
    t->ref_count = 1;  // new tensor has ref_count=1

    // Copy shape
    t->shape = (size_t*)malloc(ndim * sizeof(size_t));
    t->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!t->shape || !t->strides) {
        fprintf(stderr, "Failed to allocate shape/strides.\n");
        free(t->shape); 
        free(t->strides);
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(size_t));

    // Compute strides & total elements
    compute_strides(ndim, shape, t->strides);
    t->num_elems = compute_num_elems(ndim, shape);

    // Allocate data buffer
    size_t elem_sz = dtype_size(dtype);
    if (elem_sz == 0) {
        // dtype not supported
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }
    size_t total_bytes = t->num_elems * elem_sz;

    t->data = calloc(1, total_bytes); // zero-initialized
    if (!t->data) {
        fprintf(stderr, "Failed to allocate data buffer.\n");
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    return t;
}

/** Decrement ref_count if this is an owner; free data if it hits 0. */
static void tensor_decref(Tensor *t) {
    if (!t) return;
    if (t->owner) {
        t->ref_count--;
        if (t->ref_count == 0) {
            // Free data buffer
            free(t->data);
            t->data = NULL;
        }
    }
}

void tensor_free(Tensor *t) {
    if (!t) return;
    // Decrement ref_count if this is an owner
    tensor_decref(t);

    // Free shape/strides and the struct itself
    free(t->shape);
    free(t->strides);
    free(t);
}

Tensor* tensor_copy(const Tensor *src) {
    if (!src) return NULL;

    // Create a new tensor with the same shape & dtype
    Tensor *dst = tensor_create(src->ndim, src->shape, src->dtype);
    if (!dst) return NULL;

    // Copy the data
    size_t elem_sz = dtype_size(src->dtype);
    memcpy(dst->data, src->data, src->num_elems * elem_sz);

    return dst;
}

/* ------------------------------------------------------------------------- */
/*                           SHAPE MANIPULATION                              */
/* ------------------------------------------------------------------------- */

int tensor_reshape(Tensor *t, size_t ndim, const size_t *new_shape) {
    if (!t || !new_shape) return -1;

    // Check total elements
    size_t new_num = compute_num_elems(ndim, new_shape);
    if (new_num != t->num_elems) {
        fprintf(stderr, "[tensor_reshape] total elements mismatch.\n");
        return -1;
    }

    // Freed old shape/strides
    free(t->shape);
    free(t->strides);

    // Allocate new shape/strides
    t->shape = (size_t*)malloc(ndim * sizeof(size_t));
    t->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!t->shape || !t->strides) {
        fprintf(stderr, "[tensor_reshape] allocation failure.\n");
        return -1;
    }

    memcpy(t->shape, new_shape, ndim * sizeof(size_t));
    t->ndim = ndim;
    compute_strides(ndim, t->shape, t->strides);

    return 0;
}

Tensor* tensor_slice(Tensor *src, const size_t *start, const size_t *end) {
    if (!src || !start || !end) return NULL;

    // Create a new Tensor struct that references the same data
    Tensor *slice_t = (Tensor*)malloc(sizeof(Tensor));
    if (!slice_t) return NULL;

    slice_t->dtype = src->dtype;
    slice_t->owner = 0;  // doesn't own the data
    slice_t->ref_count = 1; // new struct
    slice_t->ndim = src->ndim;

    // Allocate shape & strides
    slice_t->shape = (size_t*)malloc(slice_t->ndim * sizeof(size_t));
    slice_t->strides = (size_t*)malloc(slice_t->ndim * sizeof(size_t));
    if (!slice_t->shape || !slice_t->strides) {
        free(slice_t->shape);
        free(slice_t->strides);
        free(slice_t);
        return NULL;
    }

    // Compute new shape
    for (size_t i = 0; i < src->ndim; i++) {
        if (end[i] <= start[i] || end[i] > src->shape[i]) {
            fprintf(stderr, "[tensor_slice] invalid slice range.\n");
            free(slice_t->shape);
            free(slice_t->strides);
            free(slice_t);
            return NULL;
        }
        slice_t->shape[i] = end[i] - start[i];
    }
    slice_t->num_elems = compute_num_elems(slice_t->ndim, slice_t->shape);

    // Copy strides from the parent
    memcpy(slice_t->strides, src->strides, src->ndim * sizeof(size_t));

    // Compute the offset (in elements)
    size_t offset = 0;
    for (size_t i = 0; i < src->ndim; i++) {
        offset += start[i] * src->strides[i];
    }

    // Convert element offset to byte offset
    size_t elem_sz = dtype_size(src->dtype);
    slice_t->data = (char*)src->data + (offset * elem_sz);

    // If src is an owner, increment ref_count
    if (src->owner) {
        src->ref_count++;
    }

    return slice_t;
}

void tensor_print(const Tensor *t, const char *name) {
    if (!t) {
        printf("Tensor '%s' is NULL\n", name);
        return;
    }
    printf("Tensor '%s':\n", name);
    printf("  ndim = %zu\n", t->ndim);
    printf("  shape = [");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("]\n  strides = [");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu", t->strides[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("]\n");
    printf("  dtype = ");
    switch (t->dtype) {
        case TENSOR_FLOAT32: printf("float32\n"); break;
        case TENSOR_FLOAT64: printf("float64\n"); break;
        case TENSOR_INT32:   printf("int32\n");   break;
        default:             printf("unknown\n"); break;
    }
    printf("  num_elems = %zu\n", t->num_elems);
    printf("  owner = %d, ref_count = %d\n", t->owner, t->ref_count);

    // Print the first few elements
    size_t max_print = (t->num_elems < 10) ? t->num_elems : 10;
    printf("  data[0..%zu]: ", max_print - 1);
    for (size_t i = 0; i < max_print; i++) {
        double val = tensor_read_at_offset(t, i);
        printf("%.3g ", val);
    }
    if (max_print < t->num_elems) {
        printf("...");
    }
    printf("\n");
}

/* ------------------------------------------------------------------------- */
/*                        INDEXING & BROADCASTING                            */
/* ------------------------------------------------------------------------- */

double tensor_get(const Tensor *t, const size_t *indices) {
    if (!t || !indices) return 0.0;
    // Compute the linear offset
    size_t offset = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    return tensor_read_at_offset(t, offset);
}

void tensor_set(Tensor *t, const size_t *indices, double value) {
    if (!t || !indices) return;
    size_t offset = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    tensor_write_at_offset(t, offset, value);
}

/* --------------------- Broadcasting Helper Routines ----------------------- */

/** 
 * Compute broadcasted shape for a and b.
 * Return number of dimensions in out_ndim,
 * and store shape in out_shape.
 * Return 0 on success, -1 on incompatible shapes.
 */
static int broadcast_shapes(const Tensor *a, const Tensor *b,
                            size_t *out_ndim, size_t *out_shape) {
    // We'll handle up to 16 dims for demonstration. Real code might do dynamic allocation.
    const size_t MAX_DIMS = 16;
    size_t rev_shape_a[MAX_DIMS] = {0};
    size_t rev_shape_b[MAX_DIMS] = {0};

    // Copy shapes in reverse order for easier handling
    for (size_t i = 0; i < a->ndim; i++) {
        rev_shape_a[i] = a->shape[a->ndim - 1 - i];
    }
    for (size_t i = 0; i < b->ndim; i++) {
        rev_shape_b[i] = b->shape[b->ndim - 1 - i];
    }

    size_t out_len = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    // Broadcast dimension by dimension
    for (size_t i = 0; i < out_len; i++) {
        size_t dim_a = (i < a->ndim) ? rev_shape_a[i] : 1;
        size_t dim_b = (i < b->ndim) ? rev_shape_b[i] : 1;
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            // Incompatible
            return -1;
        }
        out_shape[out_len - 1 - i] = (dim_a > dim_b) ? dim_a : dim_b;
    }
    *out_ndim = out_len;
    return 0;
}

static double add_op(double x, double y) { return x + y; }
static double sub_op(double x, double y) { return x - y; }
static double mul_op(double x, double y) { return x * y; }
static double div_op(double x, double y) { return x / y; }

/**
 * Recursively iterate over 'out' shape, picking corresponding elements in 'a' and 'b'.
 */
static void broadcast_recursive(const Tensor *a, const Tensor *b, Tensor *out,
                                size_t dim, size_t *idx,
                                double (*f)(double, double)) {
    if (dim == out->ndim) {
        // base case => compute offset, read a/b, write out
        // offset in out
        size_t offset_out = 0;
        for (size_t i = 0; i < out->ndim; i++) {
            offset_out += idx[i] * out->strides[i];
        }

        // offset in a
        size_t idx_a[16] = {0};
        {
            int ai = (int)a->ndim - 1;
            int oi = (int)out->ndim - 1;
            for (; oi >= 0 && ai >= 0; oi--, ai--) {
                if (a->shape[ai] == out->shape[oi]) {
                    idx_a[ai] = idx[oi];
                } else {
                    idx_a[ai] = 0;
                }
            }
        }
        size_t offset_a = 0;
        for (size_t i = 0; i < a->ndim; i++) {
            offset_a += idx_a[i] * a->strides[i];
        }

        // offset in b
        size_t idx_b[16] = {0};
        {
            int bi = (int)b->ndim - 1;
            int oi = (int)out->ndim - 1;
            for (; oi >= 0 && bi >= 0; oi--, bi--) {
                if (b->shape[bi] == out->shape[oi]) {
                    idx_b[bi] = idx[oi];
                } else {
                    idx_b[bi] = 0;
                }
            }
        }
        size_t offset_b = 0;
        for (size_t i = 0; i < b->ndim; i++) {
            offset_b += idx_b[i] * b->strides[i];
        }

        // apply
        double va = tensor_read_at_offset(a, offset_a);
        double vb = tensor_read_at_offset(b, offset_b);
        double vr = f(va, vb);
        tensor_write_at_offset(out, offset_out, vr);

        return;
    }

    for (size_t v = 0; v < out->shape[dim]; v++) {
        idx[dim] = v;
        broadcast_recursive(a, b, out, dim + 1, idx, f);
    }
}

static Tensor* tensor_broadcast_op(const Tensor *a, const Tensor *b,
                                   double (*f)(double, double),
                                   const char *op_name) {
    if (!a || !b) return NULL;

    // 1) Compute broadcasted shape
    size_t out_ndim = 0;
    size_t out_shape[16];
    if (broadcast_shapes(a, b, &out_ndim, out_shape) != 0) {
        fprintf(stderr, "[%s] shape mismatch for broadcasting.\n", op_name);
        return NULL;
    }

    // 2) Create output tensor
    Tensor *out = tensor_create(out_ndim, out_shape, a->dtype);
    if (!out) return NULL;

    // 3) Fill out with recursive broadcast
    size_t idx[16] = {0};
    broadcast_recursive(a, b, out, 0, idx, f);
    return out;
}

/* ----------------------- Actual Public Eltwise Ops ------------------------ */

Tensor* tensor_add(const Tensor *a, const Tensor *b) {
    return tensor_broadcast_op(a, b, add_op, "tensor_add");
}
Tensor* tensor_sub(const Tensor *a, const Tensor *b) {
    return tensor_broadcast_op(a, b, sub_op, "tensor_sub");
}
Tensor* tensor_mul(const Tensor *a, const Tensor *b) {
    return tensor_broadcast_op(a, b, mul_op, "tensor_mul");
}
Tensor* tensor_div(const Tensor *a, const Tensor *b) {
    return tensor_broadcast_op(a, b, div_op, "tensor_div");
}

/* ------------------------------------------------------------------------- */
/*                       REDUCTIONS & LINEAR ALGEBRA                         */
/* ------------------------------------------------------------------------- */

double tensor_sum(const Tensor *t) {
    if (!t) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < t->num_elems; i++) {
        s += tensor_read_at_offset(t, i);
    }
    return s;
}

double tensor_mean(const Tensor *t) {
    if (!t || t->num_elems == 0) return 0.0;
    double s = tensor_sum(t);
    return s / (double)t->num_elems;
}

double tensor_dot(const Tensor *v1, const Tensor *v2) {
    if (!v1 || !v2) {
        fprintf(stderr, "[tensor_dot] NULL input.\n");
        return 0.0;
    }
    if (v1->ndim != 1 || v2->ndim != 1 || v1->shape[0] != v2->shape[0]) {
        fprintf(stderr, "[tensor_dot] both tensors must be 1D of same length.\n");
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < v1->shape[0]; i++) {
        double a = tensor_read_at_offset(v1, i);
        double b = tensor_read_at_offset(v2, i);
        sum += (a * b);
    }
    return sum;
}

Tensor* tensor_matmul(const Tensor *A, const Tensor *B) {
    // A: [M, K], B: [K, N] => out: [M, N]
    if (!A || !B || A->ndim != 2 || B->ndim != 2) {
        fprintf(stderr, "[tensor_matmul] only supports 2D.\n");
        return NULL;
    }
    size_t M = A->shape[0];
    size_t K1 = A->shape[1];
    size_t K2 = B->shape[0];
    size_t N = B->shape[1];
    if (K1 != K2) {
        fprintf(stderr, "[tensor_matmul] shape mismatch.\n");
        return NULL;
    }

    // Create output
    size_t out_shape[2] = { M, N };
    Tensor *out = tensor_create(2, out_shape, A->dtype);
    if (!out) return NULL;

    // Naive triple loop
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K1; k++) {
                // offset in A => i*K1 + k
                // offset in B => k*N + j
                size_t offsetA = i * A->strides[0] + k * A->strides[1];
                size_t offsetB = k * B->strides[0] + j * B->strides[1];
                double va = tensor_read_at_offset(A, offsetA);
                double vb = tensor_read_at_offset(B, offsetB);
                sum += va * vb;
            }
            // set out
            size_t offsetOut = i * out->strides[0] + j * out->strides[1];
            tensor_write_at_offset(out, offsetOut, sum);
        }
    }
    return out;
}
