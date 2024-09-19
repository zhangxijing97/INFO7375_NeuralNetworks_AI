# HW to Chapter 1 “Brain, Neurons, and Models“

## 1. How Does a Natural Neuron Work?
A natural neuron, also known as a nerve cell, is the fundamental unit of the nervous system. It works by receiving, processing, and transmitting information through electrical and chemical signals. The neuron consists of three main parts:
- **Dendrites**: These branch-like structures receive signals from other neurons.
- **Cell Body (Soma)**: The cell body processes incoming signals and contains the nucleus, which manages the cell’s activities.
- **Axon**: The long, thread-like part of the neuron transmits electrical impulses to other neurons or muscles.

The overall process involves:
1. **Reception**: Signals (in the form of neurotransmitters) are received at the dendrites.
2. **Integration**: The cell body integrates incoming signals to determine if an action potential (a nerve impulse) should be generated.
3. **Transmission**: If the signal exceeds a threshold, an action potential is triggered and transmitted down the axon to the synapse, where it communicates with other neurons or cells.

## 2. How Does a Natural Neuron Transmit Signals to Other Neurons?
Neurons communicate with each other through a process called **synaptic transmission**. The transmission involves both electrical and chemical signals:
1. **Electrical Signal (Action Potential)**: When a neuron is sufficiently stimulated, it generates an action potential, which travels down the axon to the axon terminal.
2. **Chemical Signal (Synapse)**: Once the action potential reaches the end of the axon, neurotransmitters (chemical messengers) are released into the synaptic cleft (the small gap between neurons).
3. **Reception by Neighboring Neurons**: These neurotransmitters bind to receptors on the dendrites of the neighboring neuron, transmitting the signal to the next neuron. If the signal is strong enough, it triggers an action potential in the receiving neuron, continuing the process.

This method allows neurons to form complex communication networks within the brain and nervous system.

## 3. Describe the McCulloch and Pitts Model of Artificial Neuron
The **McCulloch and Pitts model** (1943) is one of the earliest mathematical models of a neuron, designed to simulate the functioning of a biological neuron in a simplified way. It is a **binary model** of an artificial neuron and operates as follows:
1. **Input**: The model receives several binary inputs (either 0 or 1), representing signals from other neurons.
2. **Weights**: Each input is associated with a weight, which determines the influence of that input on the output.
3. **Summation**: The weighted inputs are summed up to produce a total input value.
4. **Threshold**: The neuron fires if the total input exceeds a certain threshold value. In this case, the output is 1. If the input is below the threshold, the output is 0.

Mathematically, the McCulloch and Pitts neuron is represented as:
- **Output** = 1 if (Σ(weight × input) ≥ threshold)
- **Output** = 0 otherwise.

This model laid the foundation for more complex neural network models by illustrating how neurons can be modeled using simple binary logic gates (AND, OR, NOT).