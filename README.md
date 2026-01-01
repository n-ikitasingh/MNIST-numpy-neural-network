# MNIST-numpy-neural-network
## Neural network from scratch using NumPy to classify MNIST handwritten digits

This project implements a fully-connected neural network **from scratch using only NumPy** to classify handwritten digits from the MNIST dataset. No deep-learning libraries like TensorFlow or PyTorch were used — the goal was to manually build and understand every step of the learning pipeline.

The network supports two hidden layers, ReLU activation, softmax output, cross-entropy loss, and mini-batch gradient descent. Everything — including forward propagation and backpropagation — is written manually using matrix operations.

## Objective

The aim of this project is to:

* Understand how neural networks work internally
* Implement backpropagation mathematically
* Train a real model on MNIST
* Visualize training performance
* Achieve **90%+ accuracy without ML frameworks**

The final model achieved:

> **97.53% accuracy on the MNIST test set**



## Model Architecture

Input → Hidden Layer 1 (ReLU) → Hidden Layer 2 (ReLU) → Softmax Output

Details:

* Input size: **784 neurons** (28×28 flattened image)
* Hidden Layer 1: **configurable (e.g., 128 neurons)**
* Hidden Layer 2: **configurable (e.g., 64 neurons)**
* Output Layer: **10 neurons** (digits 0–9)
* Activation: **ReLU**
* Output Activation: **Softmax**
* Loss Function: **Cross-Entropy**
* Optimizer: **Mini-Batch Gradient Descent**

All weight updates use backpropagation built by hand.


## Training & Results

The model was trained on the MNIST dataset using:

* Normalized pixel values
* Mini-batch updates
* Logged loss and accuracy per epoch

Performance:

```
Final Test Accuracy: 97.53%
```

### Plots Generated

* Training Loss Curve📉
* Test Accuracy Curve📈 

These clearly show smooth convergence during training.


## Dataset

The MNIST handwritten digit dataset was loaded using `sklearn.datasets`.
It contains:

* 60,000 training images
* 10,000 testing images

Each image is grayscale and 28×28 pixels.


## Technologies Used

* Python 3.14
* NumPy
* scikit-learn (for dataset loading & splitting)
* Matplotlib (for visualization)
* Jupyter Notebook


## Key Learnings

By building this from scratch, I strengthened my understanding of:

* Forward propagation
* Gradient flow & derivatives
* Weight updates
* Softmax & cross-entropy
* Batch training
* Numerical stability
* Model evaluation

Instead of relying on a framework, I implemented the math directly — which gave me a much deeper intuition for how neural networks actually learn.

## Project Files

* `mnist_neural_network_numpy.ipynb` — full implementation & training notebook
* `screenshots/` — result & training screenshots


## Future Improvements

Some enhancements I plan to explore:

* Add regularization
* Add momentum / Adam optimizer
* Try different activations
* Convert into reusable Python modules
* Experiment with deeper networks


## Acknowledgement

This project was created as part of an AI/ML task to demonstrate **understanding of neural-network fundamentals beyond high-level libraries.**


## Final Note

Building this from scratch — debugging backprop, tuning learning rate, stabilizing training — was challenging but extremely rewarding. This project reflects both my **technical ability and curiosity for how AI systems actually work under the hood.**



