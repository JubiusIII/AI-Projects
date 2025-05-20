# 🧠 Handwritten Digit Recognition with PyTorch and Pygame

This project implements a neural network **from scratch** using PyTorch tensors (without `nn.Module`), capable of recognizing handwritten digits from the MNIST dataset. It also features a Pygame-based canvas where users can draw digits, have them predicted by the model, and provide the correct label for real-time learning via backpropagation.

---

## 🔧 Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| **Python 3.10+** | Main programming language |
| **PyTorch** | Tensor operations, GPU acceleration, and gradient-based learning |
| **NumPy** | Array manipulation, data preprocessing |
| **Pygame** | Drawing interface for real-time digit input |
| **Pillow (PIL)** | Image loading, resizing, and conversion |
| **Matplotlib** *(optional)* | Visualization and debugging |

---

## 📚 What Was Learned

- ✅ **Manual implementation of forward and backward passes** using PyTorch tensors without relying on high-level modules.
- ✅ Using **GPU acceleration** for inference and training when available.
- ✅ Preprocessing real-world data (user-drawn images) into **28x28 grayscale format** for model input.
- ✅ Integration of a **real-time interactive UI** (Pygame) with a learning model.
- ✅ Custom implementation of **cross-entropy loss**, softmax, and ReLU functions.
- ✅ How to **update neural network parameters manually** using computed gradients.
- ✅ Loading and processing the **raw MNIST dataset** directly from `.gz` files.

---

## 🧠 Math Involved

### 1. **Forward Propagation**

Each layer performs:

\[
z = xW + b \
a = \text{ReLU}(z) \quad \text{or} \quad \text{Softmax}(z) \quad \text{(last layer)}
\]

Where:
- \( x \) is the input
- \( W \) are weights
- \( b \) are biases
- ReLU is \( \max(0, z) \)
- Softmax is:

\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

---

### 2. **Loss Function – Categorical Cross Entropy**

Given true labels \( y \) (one-hot encoded) and predicted probabilities \( \hat{y} \), the loss is:

\[
L = -\sum y_i \log(\hat{y}_i)
\]

The average of this loss across samples is used during training.

---

### 3. **Backpropagation**

Using the chain rule:

- Compute derivative of the loss w.r.t. output (`dLoss/dOut`)
- Backpropagate through softmax and ReLU
- Calculate gradients for each weight and bias:

\[
\frac{\partial L}{\partial W} = a^T \cdot \delta \
\frac{\partial L}{\partial b} = \sum \delta
\]

- Update parameters using gradient descent:

\[
W := W - \eta \cdot \frac{\partial L}{\partial W}
\]

---

## 🖼️ How It Works

1. Run the canvas (`WriteTest.py`) and draw a digit.
2. Press **`S`** to predict.
3. If the prediction is wrong, you'll be prompted to enter the correct label.
4. The model will **learn from your correction** and update weights using backpropagation.

---

## 🗂️ Project Structure

```
neuralFromScratch/
│
├── NNFromScratch.py         # Model definition, forward/backward pass, parameter updates
├── WriteTest.py             # Pygame canvas and prediction interface
├── MNISTdata/               # Contains gzipped MNIST image and label files
├── modelWeights/            # Pretrained model weights (optional)
```

---

## 🧪 Example

> 🖱️ Draw a "3", press `S`.  
> 📤 Model predicts "5"?  
> ✏️ You enter "3", and the model updates!  
> Next time, it might get it right!

---

## 🚀 Future Improvements

- Convert model into a proper `nn.Module` class
- Add a training mode from scratch using the full MNIST dataset
- Implement saving/loading learned weights
- Add convolutional layers for better accuracy on noisy inputs

---

## 🏁 Running the App

```bash
# Install dependencies
pip install torch numpy pygame pillow matplotlib

# Run the digit drawer
python WriteTest.py
```

---

## 📬 Credits

Created by [Your Name]  
Inspired by deep learning fundamentals and a love for building things from scratch.
