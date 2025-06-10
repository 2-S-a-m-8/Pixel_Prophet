# Handwritten Digit Recognition Neural Network from Scratch

This project implements a neural network from scratch to recognize handwritten digits. The model is built and trained using Python, primarily leveraging the MNIST dataset for training and testing.

## Files in This Repository

*   **`Hand_Written_NN_FromSratch.ipynb`**: This is the main Jupyter Notebook containing the Python code for the neural network. It includes the implementation of the network architecture, activation functions, forward and backward propagation, training loop, and evaluation on the MNIST dataset.
*   **`Math_For_NN.pdf`**: This PDF document contains the mathematical theory, derivations, and concepts that form the basis of the neural network implemented in the notebook.

## Functionality

The core functionality of the neural network implemented in `Hand_Written_NN_FromSratch.ipynb` includes:

*   **Network Architecture:** A neural network built from scratch.
*   **Activation Functions:** ReLU (Rectified Linear Unit) and its derivative.
*   **Output Layer Activation:** Softmax function for multi-class classification (digits 0-9).
*   **Parameter Initialization:** Random initialization of weights and biases.
*   **Forward Propagation:** Calculates the network's output based on the input and current parameters.
*   **Backward Propagation:** Calculates the gradients of the loss function with respect to the network parameters.
*   **One-Hot Encoding:** Converts numerical labels into a one-hot vector format suitable for the softmax output.
*   **Gradient Descent:** Optimizes the network parameters by updating them in the direction that minimizes the loss.
*   **Prediction & Accuracy:** Functions to predict labels for new data and calculate the model's accuracy.
*   **Loss Function:** Categorical cross-entropy to measure the difference between predicted probabilities and actual labels.

## How to Run the Code

1.  **Environment:** Ensure you have a Python environment with Jupyter Notebook installed.
2.  **Libraries:** You will need the following Python libraries. You can install them using pip:
    *   `numpy`
    *   `matplotlib`
    *   `keras` (specifically for `keras.datasets.mnist` to load the dataset)

    ```bash
    pip install numpy matplotlib tensorflow keras
    ```
    *(Note: Keras is often used with a backend like TensorFlow. Installing TensorFlow will typically include Keras.)*
3.  **Launch Jupyter Notebook:** Navigate to the repository's directory in your terminal and launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4.  **Open and Run:** Open the `Hand_Written_NN_FromSratch.ipynb` file from the Jupyter interface and run the cells sequentially.

## Expected Output

When you run the `Hand_Written_NN_FromSratch.ipynb` notebook, you should expect to see the following:

*   **Training Progress:** The notebook will print the iteration number, training loss, and test loss at regular intervals during the training process.
*   **Example Predictions:** After training, the notebook will display several example handwritten digits from the test set, along with the model's prediction and the actual label for each.
*   **Loss Curves:** A plot will be generated showing the training loss and test loss curves over the training iterations. This helps visualize how the model's performance on both datasets changes over time.
*   **Accuracy Metrics:** Finally, the notebook will print the overall training accuracy and test accuracy of the model on the MNIST dataset.

## Potential Improvements / Future Work

This project provides a solid foundation for understanding and building neural networks. Some potential areas for improvement or future work include:

*   **Different Network Architectures:** Experiment with adding more layers (deeper network) or more neurons per layer.
*   **Different Activation Functions:** Try other activation functions like Sigmoid or Tanh and compare their performance.
*   **Advanced Optimizers:** Implement and test more advanced optimization algorithms like Adam or RMSprop instead of basic gradient descent.
*   **Regularization:** Introduce regularization techniques (e.g., L2 regularization, dropout) to prevent overfitting.
*   **Hyperparameter Tuning:** Systematically tune hyperparameters like the learning rate, number of iterations, and batch size to find optimal settings.
*   **Convolutional Layers:** For image data like MNIST, implementing convolutional neural network (CNN) layers would likely yield significantly better performance.
*   **Different Datasets:** Adapt the network to work with other image datasets.

## Contributing

Contributions to this project are welcome. If you have suggestions for improvements or find any issues, feel free to open an issue or submit a pull request.

## License

This project is open-source. Consider adding a license file (e.g., MIT License) if you wish to specify terms of use.
