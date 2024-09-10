# INFO7375_NeuralNetworks_AI

## Table of Contents

- [Chapter 1: Human Brain and Neural Networks](#chapter-1-human-brain-and-neural-networks)
  - [Python Environment](#python-environment)
  - [Human Brain and Biological Neurons](#human-brain-and-biological-neurons)
  - [Synapses and Neural Networks](#synapses-and-neural-networks)
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
1. **Neuron**: The basic unit of the nervous system, responsible for processing and transmitting information.
2. **Dendrite**: Receives signals toward the cell body, many per neuron, short and branched, not myelinated, tree-like structure.
3. **Neuron Cell Body**: The signal travels toward the neuron's cell body (soma), where it is processed.
4. **Axon**: Transmits signals away from the cell body, usually one per neuron, can be long, often myelinated, smooth structure.
5. **Synapse**: The junction between two neurons where information is transmitted from one neuron to another.
6. **Neuromorphic**: Referring to the design and development of hardware and software systems inspired by the structure and function of the human brain.
7. **Synaptic Plasticity**: The ability of synapses to strengthen or weaken over time, in response to increases or decreases in their activity.

### McCulloch and Pitts Neuron Model