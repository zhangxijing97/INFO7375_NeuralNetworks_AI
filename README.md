# INFO7375_NeuralNetworks_AI

## Table of Contents

- [Chapter 1: Human Brain and Neural Networks](#chapter-1-human-brain-and-neural-networks)
  - [Python Environment](#python-environment)
  - [Human Brain and Biological Neurons](#human-brain-and-biological-neurons)
  - [Neural Network Basics](#neural-metwork-basics)
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

### Neural Network Basics
Neurons: a thing that holds a number between 0.0 and 1.0.<br>
Digit images have 28 × 28 = 784 pixels, so we create a the first layer with 784 neurons<br>
<p align="left">
  <img src="./Image/highlight-first-layer.png" alt="First Layer" width="400"/>
</p>

The output layer of our network has 10 neurons, corresponds 1 - 10.<br>
<br>
<p align="left">
  <img src="./Image/output-layer.png" alt="Output Layer" width="400"/>
</p>

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