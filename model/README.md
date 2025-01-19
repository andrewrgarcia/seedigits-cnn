# CNN Architecture

**Layers:**

1. **Conv1:** 
    * Input channels: 1
    * Output channels (filters): 16
    * Kernel size: 3x3
    * Stride: 1
    * Padding: 1
    * Activation: ReLU

2. **Conv2:** 
    * Input channels: 16
    * Output channels (filters): 32
    * Kernel size: 3x3
    * Stride: 1
    * Padding: 1
    * Activation: ReLU

3. **MaxPool:**
    * Kernel size: 2x2
    * Stride: 2 

4. **FC1:** 
    * Fully connected layer with 64 units
    * Activation: ReLU

5. **FC2:** 
    * Fully connected layer with 10 output classes 

**Note:**

* ReLU stands for Rectified Linear Unit, an activation function that introduces non-linearity.
* MaxPool performs downsampling by selecting the maximum value within each pooling window.
* FC1 and FC2 are fully connected layers, where each neuron is connected to every neuron in the previous layer.
