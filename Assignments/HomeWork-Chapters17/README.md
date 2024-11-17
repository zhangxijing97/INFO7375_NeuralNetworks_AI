# HW to Chapter 17 “Overlapping Objects and Semantic Segmentation”

# Non-programming Assignment

## 1. What are anchor boxes and how do they work?

**Anchor boxes** are predefined bounding boxes of various aspect ratios and scales used in object detection models, such as YOLO and Faster R-CNN. They help predict objects with different shapes and sizes within an image. 

Each anchor box serves as a reference for the network to predict offsets to refine the box's position and size. During training, ground truth bounding boxes are matched with the most overlapping anchor boxes using Intersection over Union (IoU). The model then adjusts the anchor boxes to better fit the actual objects.

---

## 2. What is bounding box prediction and how does it work?

**Bounding box prediction** involves identifying the coordinates (x, y, width, height) of a rectangular region enclosing an object in an image. This is a fundamental task in object detection.

The process typically works as follows:
1. **Feature Extraction**: A convolutional neural network (CNN) extracts feature maps from the input image.
2. **Region Proposal**: Regions of interest are identified either through predefined anchor boxes or using region proposal networks.
3. **Box Refinement**: The network predicts offsets (dx, dy, dw, dh) relative to the anchor boxes or proposals to adjust their size and position for a tighter fit.

The goal is to minimize the difference between the predicted box and the ground truth box during training.

---

## 3. Describe R-CNN

**R-CNN (Region-based Convolutional Neural Networks)** is a pioneering object detection algorithm. It works as follows:
1. **Region Proposal**: Generates ~2000 region proposals per image using selective search.
2. **Feature Extraction**: Each proposal is resized and passed through a CNN to extract features.
3. **Classification**: The extracted features are fed into a support vector machine (SVM) to classify the object in the region.
4. **Bounding Box Regression**: A regression model refines the coordinates of the bounding box.

---

## 4. What are advantages and disadvantages of R-CNN?

### Advantages:
- High detection accuracy due to the combination of CNNs for feature extraction and SVMs for classification.
- Introduced the use of deep learning for object detection.

### Disadvantages:
- **Slow**: Each region proposal is processed individually, resulting in significant computational overhead.
- **High Storage Requirement**: Features for all regions must be stored, leading to high memory usage.
- **Not End-to-End**: Separate stages for region proposal, feature extraction, classification, and bounding box regression.

---

## 5. What is semantic segmentation?

**Semantic segmentation** involves classifying each pixel in an image into a category. Unlike object detection, which provides bounding boxes, semantic segmentation provides pixel-level masks, allowing precise delineation of objects.

For example, in an image of a street, semantic segmentation might label pixels as road, car, pedestrian, building, or sky.

---

## 6. How does deep learning work for semantic segmentation?

Deep learning for semantic segmentation typically uses convolutional neural networks (CNNs) and works as follows:
1. **Feature Extraction**: A CNN extracts hierarchical features from the image.
2. **Pixel Classification**: Each pixel is classified into a category based on its features.
3. **Upsampling**: Low-resolution feature maps are upsampled using transposed convolutions or other techniques to restore the original image resolution.

Popular architectures like U-Net, SegNet, and DeepLab are widely used for semantic segmentation tasks.

---

## 7. What is transposed convolution?

**Transposed convolution**, also called **deconvolution** or **up-convolution**, is a technique to increase the spatial resolution of feature maps. It is commonly used in tasks like semantic segmentation and image generation.

It works by:
1. Expanding the input feature map by inserting zeros between elements.
2. Applying a convolution operation on the expanded feature map with a kernel to fill in values.

This process effectively "reverses" the convolution operation, hence the name transposed convolution.

---

## 8. Describe U-Net

**U-Net** is a popular architecture for semantic segmentation. It consists of two main parts:
1. **Encoder**: A downsampling path (contracting path) that extracts features and reduces spatial resolution using convolutions and pooling.
2. **Decoder**: An upsampling path (expanding path) that restores spatial resolution using transposed convolutions, combining low-level features from the encoder via skip connections.

The skip connections help recover fine-grained spatial details, making U-Net effective for tasks requiring high-resolution outputs, such as medical image segmentation.

---