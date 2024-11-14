# HW Chapter 15 "Transfer Learning"

# Non-programming Assignment

## 1. What is spatial separable convolution and how is it different from simple convolution?

Spatial separable convolution is a technique used to reduce the computational cost of a standard convolution operation in Convolutional Neural Networks (CNNs). It works by decomposing a 2D kernel into two 1D kernels — one for horizontal convolution and one for vertical convolution. This reduces the number of computations required and makes the process more efficient.<br>

### How Does Spatial Separable Convolution Work?
Instead of directly applying a 2D kernel (such as a `3 x 3` matrix) to an image, spatial separable convolution breaks the kernel into two smaller 1D kernels:
- 1D Horizontal Filter (applied along the rows)
- 1D Vertical Filter (applied along the columns)

### Example: Using a `3 x 3` Kernel:

#### Step 1: 2D Convolution (Simple Convolution)

`K = [ 1 2 1 0 0 0 -1 -2 -1 ]`

#### Step 2: Spatial Separable Convolution

1. Horizontal 1D Filter (size `1 x 3`):
`K_horizontal = [1 2 1]`

2. Vertical 1D Filter (size `3 x 1`):
`K_vertical = [ 1 0 -1 ]`

| **Aspect**                         | **Simple Convolution**                      | **Spatial Separable Convolution**                  |
|------------------------------------|--------------------------------------------|---------------------------------------------------|
| **Kernel Size**                    | `3 x 3` (2D)                        | Decomposed into `1 x 3` and `3 x 1` (2 1D filters) |
| **Computational Complexity**       | `O(k^2)`, for `k x k` kernel      | `O(2k)`, for `k` size kernel (due to 1D convolutions) |
| **Operations per Filter**          | 9 operations for a `3 x 3` kernel   | 6 operations (3 for horizontal, 3 for vertical)     |
| **Memory Usage**                   | Requires storing a full `3 x 3` kernel | Requires storing two 1D filters: `1 x 3` and `3 x 1` |

### Advantages of Spatial Separable Convolution:
1. Reduced Computational Cost: In simple convolution, a `3 x 3` kernel requires 9 operations for each region of the image. In spatial separable convolution, we only need 6 operations (by applying two 1D convolutions).
2. Memory Efficiency: Storing two 1D kernels requires less memory than storing one 2D kernel, especially for larger kernels.


## 2. What is the difference between depthwise and pointwise convolutions?

## 3. What is the sense of 1 x 1 convolution?

## 4. What is the role of residual connections in neural networks?