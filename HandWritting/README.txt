
🧠 Handwritten Digit Recognition with PyTorch and Pygame

This project implements a neural network **from scratch** using PyTorch tensors (without `nn.Module`), capable of recognizing handwritten digits from the MNIST dataset. It also features a Pygame-based canvas where users can draw digits, have them predicted by the model, and provide the correct label for real-time learning via backpropagation.

---

🔧 Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| **Python 3.10+** | Main programming language |
| **PyTorch** | Tensor operations, GPU acceleration, and gradient-based learning |
| **NumPy** | Array manipulation, data preprocessing |
| **Pygame** | Drawing interface for real-time digit input |
| **Pillow (PIL)** | Image loading, resizing, and conversion |
| **Matplotlib** *(optional)* | Visualization and debugging |

---

📚 What Was Learned

- ✅ **Manual implementation of forward and backward passes** using PyTorch tensors without relying on high-level modules.
- ✅ Using **GPU acceleration** for inference and training when available.
- ✅ Preprocessing real-world data (user-drawn images) into **28x28 grayscale format** for model input.
- ✅ Integration of a **real-time interactive UI** (Pygame) with a learning model.
- ✅ Custom implementation of **cross-entropy loss**, softmax, and ReLU functions.
- ✅ How to **update neural network parameters manually** using computed gradients.
- ✅ Loading and processing the **raw MNIST dataset** directly from `.gz` files.

---

🧠 Math Involved

1. **Forward Propagation**

Each layer performs:

    z = xW + b
    a = ReLU(z) or Softmax(z) (last layer)

Where:
- x is the input
- W are weights
- b are biases
- ReLU is max(0, z)
- Softmax is:

    Softmax(z_i) = exp(z_i) / sum_j(exp(z_j))

2. **Loss Function – Categorical Cross Entropy**

Given true labels y (one-hot encoded) and predicted probabilities y_hat, the loss is:

    L = -sum(y_i * log(y_hat_i))

The average of this loss across samples is used during training.

3. **Backpropagation**

Using the chain rule:

- Compute derivative of the loss w.r.t. output (dLoss/dOut)
- Backpropagate through softmax and ReLU
- Calculate gradients for each weight and bias:

    dL/dW = a^T * delta
    dL/db = sum(delta)

- Update parameters using gradient descent:

    W := W - eta * dL/dW

---

🖼️ How It Works

1. Run the canvas (WriteTest.py) and draw a digit.
2. Press `S` to predict.
3. If the prediction is wrong, you'll be prompted to enter the correct label.
4. The model will **learn from your correction** and update weights using backpropagation.

---

🗂️ Project Structure

    neuralFromScratch/
    ├── NNFromScratch.py         # Model definition, forward/backward pass, parameter updates
    ├── WriteTest.py             # Pygame canvas and prediction interface
    ├── MNISTdata/               # Contains gzipped MNIST image and label files
    ├── modelWeights/            # Pretrained model weights (optional)

---

🧪 Example

    🖱️ Draw a "3", press S.
    📤 Model predicts "5"?
    ✏️ You enter "3", and the model updates!
    ✅ Next time, it might get it right!

---

🚀 Future Improvements

- Convert model into a proper nn.Module class
- Add a training mode from scratch using the full MNIST dataset
- Implement saving/loading learned weights
- Add convolutional layers for better accuracy on noisy inputs

---

🏁 Running the App

    # Install dependencies
    pip install torch numpy pygame pillow matplotlib

    # Run the digit drawer
    python WriteTest.py

---

📬 Credits

Created by Joe Taylor
Inspired by Green Code, deep learning fundamentals, and a love for building things from scratch.
