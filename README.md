# INFO7375_NeuralNetworks_AI

## Table of Contents

- [Chapter 1: Human Brain and Neural Networks](#chapter-1-human-brain-and-neural-networks)
  - [Python Environment](#python-environment)
  - [Human Brain and Biological Neurons](#human-brain-and-biological-neurons)
  - [Neural Networks Basics](#neural-networks-basics)
  - [McCulloch and Pitts Neuron Model](#mcculloch-and-pitts-neuron-model)

- [Chapter 2: The Perceptron](#chapter-2-the-perceptron)
  - [Introduction to the Perceptron](#introduction-to-the-perceptron)
  - [History of the Perceptron](#history-of-the-perceptron)

- [Chapter 3: Supervised Training and Logistic Regression](#chapter-3-supervised-training-and-logistic-regression)
  - [Perceptron for Logistic Regression](#perceptron-for-logistic-regression)
  - [Neural Networks for Logistic Regression and Classification](#neural-networks-for-logistic-regression-and-classification)
  - [Linear Binary Classifier](#linear-binary-classifier)
  - [Loss (Cost) Function](#loss-cost-function)

- [Chapter 4: Advanced Topics in Logistic Regression](#chapter-4-advanced-topics-in-logistic-regression)
  - [Perceptron for Logistic Regression with Sigmoid Activation Function](#perceptron-for-logistic-regression-with-sigmoid-activation-function)
  - [Vectors and Matrices](#vectors-and-matrices)
  - [Perceptron for Logistic Regression with Many Training Samples](#perceptron-for-logistic-regression-with-many-training-samples)
  - [Gradient Descent Optimization](#gradient-descent-optimization)
  - [Logistic Regression](#logistic-regression)
  - [Neural Networks for Logistic Regression](#neural-networks-for-logistic-regression)

## Chapter 1: Human Brain and Neural Networks

### Python Environment
Create a virtual environment:
```
python -m venv env_name
```

Activate the virtual environment:
```
source env_name/bin/activate
```

Install required packages (if any):
```
pip install <package_name>
```

### Human Brain and Biological Neurons
![Neuron Structure](./Image/neuron_structure.png)
1. **Neuron**: The basic unit of the nervous system, responsible for processing and transmitting information.
2. **Dendrite**: Receives signals toward the cell body, many per neuron, short and branched, not myelinated, tree-like structure.
3. **Neuron Cell Body**: The signal travels toward the neuron's cell body (soma), where it is processed.
4. **Axon**: Transmits signals away from the cell body, usually one per neuron, can be long, often myelinated, smooth structure.
5. **Synapse**: The junction between two neurons where information is transmitted from one neuron to another.
6. **Neuromorphic**: Referring to the design and development of hardware and software systems inspired by the structure and function of the human brain.
7. **Synaptic Plasticity**: The ability of synapses to strengthen or weaken over time, in response to increases or decreases in their activity.

### Neural Networks Basics
#### Neurons
Neurons: a thing that holds a number between 0.0 and 1.0.<br>
Digit images have 28 × 28 = 784 pixels, so we create a the first layer with 784 neurons.<br>

<img src="./Image/highlight-first-layer.png" alt="First Layer" width="400"/>

The output layer of our network has 10 neurons, corresponds 1 - 10.<br>

<img src="./Image/output-layer.png" alt="Output Layer" width="400"/>

#### Neural Network Architecture
In a perfect world, A loop can be broken down into several small edges.<br>

<img src="./Image/loop-edges.png" alt="Image" width="400"/><br>
<img src="./Image/upper-loop-neuron.png" alt="Image" width="400"/>

#### Weighted Sum Formula
Weighted Sum = w1*a1 + w2*a2 + w3*a4 + ... + wn*an<br>
- a (activation) refers to the pixel values from the image. These values, typically between 0 and 1 (e.g., grayscale intensity)<br>
- w (weight) represents how important each pixel(neuron) from the image is for a neuron in the next layer.<br>

Since there are 784 neurons in the first layer, each neuron in the second layer has 784 weights.<br>

<p align="left">
  <img src="./Image/weights-blue.png" alt="Image" width="400"/>
  <img src="./Image/weights-square-blue.png" alt="Image" width="400"/>
</p>

#### Neural Network Structure: Neurons, Weights, and Biases
| Layer            | Number of Neurons | Weights per Neuron | Total Weights | Bias per Neuron | Total Biases |
|------------------|-------------------|--------------------|---------------|-----------------|--------------|
| Input Layer      | 50                | 0                  | 0             | 0               | 0            |
| Second Layer     | 100               | 50                 | 5000          | 1               | 100          |
| Third Layer      | 64                | 100                | 6400          | 1               | 64           |
| Output Layer     | 10                | 64                 | 640           | 1               | 10           |

#### ReLU (Rectified Linear Unit)
The **ReLU** activation function outputs the input directly if it is positive; otherwise, it returns 0. This function is widely used in hidden layers for its efficiency.

The formula for the ReLU function is:

`ReLU(x) = max(0, x)`

#### Sigmoid Function
The **sigmoid** activation function maps input values to a range between 0 and 1, which is useful for binary classification tasks.

The formula for the sigmoid function is:

`σ(x) = 1 / (1 + exp(-x))`

<img src="./Image/sigmoid.png" alt="Image" width="400"/>

#### Vanishing Gradient Problem
For an input that results in large positive or negative values, the sigmoid function will output values near 0 or 1, and the derivative (gradient) will be close to 0. When this gradient is backpropagated through many layers, it shrinks even further.<br>

#### Bias in Neural Networks
**Bias** allows the model to adjust the relationship between inputs (x) and outputs (y). Without bias, the network is constrained to model a **linear relationship** passing through the origin (when x = 0, y = 0). This limits its ability to capture complex patterns. With bias, the model can better fit data where the output isn't directly proportional to the input.

#### Softmax Function
To convert the output values of the neurons in the output layer into probabilities that sum to 1, we apply the **softmax** function. The softmax function is defined as:

\[
P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

Where:
- \( z_i \) is the raw output (weighted sum + bias) of the neuron.
- The exponentials \( e^{z_i} \) ensure that all output values are positive.
- The result is a set of probabilities, and the neuron with the highest probability is considered the predicted class.



### McCulloch and Pitts Neuron Model

#### Logic Gates using McCulloch-Pitts Neuron Model
The **AND gate** outputs 1 only when both inputs are 1.

| x1  | x2  | AND Output (y) |
| --- | --- | -------------- |
| 0   | 0   | 0              |
| 0   | 1   | 0              |
| 1   | 0   | 0              |
| 1   | 1   | 1              |

The **OR gate** outputs 1 if at least one input is 1.

| x1  | x2  | OR Output (y)  |
| --- | --- | -------------- |
| 0   | 0   | 0              |
| 0   | 1   | 1              |
| 1   | 0   | 1              |
| 1   | 1   | 1              |

The **NOT gate** inverts the input. If the input is 0, the output is 1. If the input is 1, the output is 0.

| z   | NOT Output (y) |
| --- | -------------- |
| 0   | 1              |
| 1   | 0              |

The **NOR gate** outputs 1 only when both inputs are 0 (inversion of OR gate).

| x1  | x2  | NOR Output (y) |
| --- | --- | -------------- |
| 0   | 0   | 1              |
| 0   | 1   | 0              |
| 1   | 0   | 0              |
| 1   | 1   | 0              |

The **XOR gate** outputs 1 only when the inputs are different (i.e., one input is 1 and the other is 0).

| x1  | x2  | XOR Output (y) |
| --- | --- | ---------------|
| 0   | 0   | 0              |
| 0   | 1   | 1              |
| 1   | 0   | 1              |
| 1   | 1   | 0              |

XOR Gate Construction:
XOR can be constructed using **AND**, **OR**, and **NOT** gates:<br>
`XOR(x1, x2) = (x1 OR x2) AND NOT(x1 AND x2)`