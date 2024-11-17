# main.py

import numpy as np

def depthwise_convolution(image, kernel):
    """
    Perform depthwise convolution on a multi-channel image.

    :param image: Input image, shape (height, width, channels)
    :param kernel: Depthwise convolution kernel, shape (k_height, k_width, channels)
    :return: Depthwise convoluted image
    """
    h, w, c = image.shape
    k_h, k_w, c_k = kernel.shape

    if c != c_k:
        raise ValueError("Number of channels in image and kernel must match for depthwise convolution.")

    out_h = h - k_h + 1
    out_w = w - k_w + 1
    output = np.zeros((out_h, out_w, c))

    for channel in range(c):
        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i+k_h, j:j+k_w, channel]
                output[i, j, channel] = np.sum(region * kernel[:, :, channel])

    return output


def pointwise_convolution(image, kernel):
    """
    Perform pointwise convolution (1x1 convolution) on a multi-channel image.

    :param image: Input image, shape (height, width, channels)
    :param kernel: Pointwise kernel, shape (channels, num_filters)
    :return: Pointwise convoluted image
    """
    h, w, c = image.shape
    c_k, num_filters = kernel.shape

    if c != c_k:
        raise ValueError("Number of input channels in the image must match the channels in the kernel for pointwise convolution.")

    output = np.zeros((h, w, num_filters))

    for i in range(h):
        for j in range(w):
            region = image[i, j, :]
            output[i, j, :] = np.dot(region, kernel)

    return output


def convolution(image, kernel, mode="depthwise"):
    """
    General function to perform convolution based on mode.

    :param image: Input image, shape (height, width, channels)
    :param kernel: Convolution kernel
    :param mode: 'depthwise' or 'pointwise'
    :return: Convoluted image
    """
    if mode == "depthwise":
        return depthwise_convolution(image, kernel)
    elif mode == "pointwise":
        return pointwise_convolution(image, kernel)
    else:
        raise ValueError("Invalid mode. Choose 'depthwise' or 'pointwise'.")

image = np.random.rand(5, 5, 3)  # Random 5x5 image with 3 channels
depthwise_kernel = np.random.rand(3, 3, 3)  # 3x3 kernel for depthwise convolution
pointwise_kernel = np.random.rand(3, 2)  # 1x1 kernel for pointwise convolution

# Depthwise convolution
depthwise_result = convolution(image, depthwise_kernel, mode="depthwise")
print("Depthwise Convolution Result:")
print(depthwise_result)

# Pointwise convolution
pointwise_result = convolution(image, pointwise_kernel, mode="pointwise")
print("Pointwise Convolution Result:")
print(pointwise_result)