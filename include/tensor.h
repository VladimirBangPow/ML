#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/*                          DATA TYPES & STRUCTS                             */
/* ------------------------------------------------------------------------- */

/** Supported data types. */
typedef enum {
    TENSOR_FLOAT32,
    TENSOR_FLOAT64,
    TENSOR_INT32,
    // Extend as needed...
} TensorDtype;

/**
 * The main Tensor structure.
 *
 * - 'ndim':       Number of dimensions.
 * - 'shape':      Array of dimension sizes (length = ndim).
 * - 'strides':    Array of strides for each dimension (length = ndim).
 * - 'data':       Pointer to the raw data buffer (shared among references).
 * - 'dtype':      Data type (float32, float64, int32, etc.).
 * - 'ref_count':  Reference counter for shared data ownership.
 * - 'owner':      If 1, this tensor is considered the 'owner' of the data buffer.
 *                 If 0, it means the data pointer is shared from another tensor.
 * - 'num_elems':  Total number of elements (product of shape).
 */
typedef struct {
    size_t      ndim;
    size_t     *shape;
    size_t     *strides;
    void       *data;
    TensorDtype dtype;
    int         ref_count;  
    int         owner;      
    size_t      num_elems;  
} Tensor;

/* ------------------------------------------------------------------------- */
/*                        BASIC TENSOR LIFECYCLE                             */
/* ------------------------------------------------------------------------- */

/**
 * Create a new tensor with the specified shape and data type.
 * The underlying data buffer is allocated and zero-initialized.
 *
 * @param ndim   Number of dimensions
 * @param shape  Array of dimension sizes (length = ndim)
 * @param dtype  Data type (e.g., TENSOR_FLOAT32)
 * @return       Pointer to a newly allocated Tensor (or NULL on failure)
 */
Tensor* tensor_create(size_t ndim, const size_t *shape, TensorDtype dtype);

/**
 * Free a Tensor. Decrements the reference count. If it reaches zero, data is deallocated.
 */
void tensor_free(Tensor *t);

/**
 * Create a (deep) copy of an existing tensor. The new tensor owns its own data.
 *
 * @param src  The source tensor
 * @return     A new Tensor with identical shape, dtype, and data
 */
Tensor* tensor_copy(const Tensor *src);

/* ------------------------------------------------------------------------- */
/*                           SHAPE MANIPULATION                              */
/* ------------------------------------------------------------------------- */

/**
 * Reshape a tensor in-place (if contiguous and the total number of elements
 * remains the same). Returns 0 on success, -1 on failure.
 */
int tensor_reshape(Tensor *t, size_t ndim, const size_t *new_shape);

/**
 * Create a sliced 'view' of an existing tensor. The returned tensor
 * shares the underlying data (no copy). 
 *
 * @param src        The source tensor
 * @param start      Array of start indices for each dimension
 * @param end        Array of end indices (exclusive) for each dimension
 * @return           A new Tensor that references the same data
 *
 * Example: For a 2D tensor shape=[4,5], slicing rows [1..3) and columns [0..4)
 *          => new shape=[2,4] (rows=3-1=2, cols=4-0=4)
 */
Tensor* tensor_slice(Tensor *src, const size_t *start, const size_t *end);

/**
 * Print basic info (shape, strides, dtype) and some sample values.
 * Useful for debugging.
 */
void tensor_print(const Tensor *t, const char *name);

/* ------------------------------------------------------------------------- */
/*                     LOW-LEVEL OFFSET-BASED ACCESS                         */
/* ------------------------------------------------------------------------- */

/**
 * Read an element from the tensor at a given linear 'offset' (row-major),
 * and return it as a double. 
 * No bounds checking is performed.
 *
 * @param t       The tensor (must not be NULL)
 * @param offset  The zero-based element index into the underlying array
 * @return        The value as double
 */
double tensor_read_at_offset(const Tensor *t, size_t offset);

/**
 * Write a double value into the tensor at a given linear 'offset' (row-major),
 * casting to the tensor’s dtype. No bounds checking is performed.
 *
 * @param t       The tensor (must not be NULL)
 * @param offset  The zero-based element index
 * @param value   The value to store
 */
void tensor_write_at_offset(Tensor *t, size_t offset, double value);

/* ------------------------------------------------------------------------- */
/*                        INDEXING & BROADCASTING                            */
/* ------------------------------------------------------------------------- */

/**
 * Get an element (returned as double) using an array of indices.
 *
 * @param t        The tensor
 * @param indices  Array of length t->ndim
 * @return         The element value cast to double
 */
double tensor_get(const Tensor *t, const size_t *indices);

/**
 * Set an element in the tensor (the value is provided as double, 
 * then cast to the tensor's dtype).
 *
 * @param t        The tensor
 * @param indices  Indices array
 * @param value    The value (double)
 */
void tensor_set(Tensor *t, const size_t *indices, double value);

/**
 * Element-wise add: out = a + b, with broadcasting.
 * Creates a new tensor. 'a' and 'b' remain unchanged.
 *
 * Broadcasting rules: 
 * - If shapes differ in a dimension, one of them must have size 1 or the same size as the other.
 */
Tensor* tensor_add(const Tensor *a, const Tensor *b);

/**
 * Element-wise subtract: out = a - b, with broadcasting.
 */
Tensor* tensor_sub(const Tensor *a, const Tensor *b);

/**
 * Element-wise multiply: out = a * b, with broadcasting.
 */
Tensor* tensor_mul(const Tensor *a, const Tensor *b);

/**
 * Element-wise division: out = a / b, with broadcasting.
 */
Tensor* tensor_div(const Tensor *a, const Tensor *b);

/* ------------------------------------------------------------------------- */
/*                       REDUCTIONS & LINEAR ALGEBRA                         */
/* ------------------------------------------------------------------------- */

/**
 * Sum all elements in the tensor, returned as double.
 */
double tensor_sum(const Tensor *t);

/**
 * Compute the mean (average) of all elements.
 */
double tensor_mean(const Tensor *t);

/**
 * Dot product for 1D tensors (vectors). 
 * Both must be 1D of the same shape[0].
 */
double tensor_dot(const Tensor *v1, const Tensor *v2);

/**
 * Naive 2D matrix multiply: out = A x B
 * - A: shape=[M, K]
 * - B: shape=[K, N]
 * - out: shape=[M, N]
 *
 * Returns a new allocated tensor with the result.
 */
Tensor* tensor_matmul(const Tensor *A, const Tensor *B);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_H */
