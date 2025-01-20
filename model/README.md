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

4. **Conv3:** 
    * Input channels: 32
    * Output channels (filters): 64
    * Kernel size: 3x3
    * Stride: 1
    * Padding: 1
    * Activation: ReLU

5. **AvgPool (Global Pooling):**
    * Kernel size: 3x3
    * Stride: 3
    * Purpose: Reduces spatial dimensions to 4x4.

6. **Flatten:**
    * Converts the output of the previous layer (shape: `[batch_size, 64, 4, 4]`) into a flat vector (shape: `[batch_size, 1024]`).

7. **FC1:** 
    * Fully connected layer with 128 units.
    * Activation: ReLU

8. **FC2:** 
    * Fully connected layer with 10 output classes.
