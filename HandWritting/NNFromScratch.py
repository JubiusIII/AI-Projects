import numpy as np
import gzip
import torch
import torch.nn.functional as F

# Choose device automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetworkTorch:
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10, device='cpu'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.device = device
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()

        # Input to first hidden layer
        self.weights.append(torch.nn.Parameter(torch.randn(input_size, hidden_layers[0], device=device) * 0.01))
        self.biases.append(torch.nn.Parameter(torch.zeros((1, hidden_layers[0]), device=device)))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(torch.nn.Parameter(torch.randn(hidden_layers[i], hidden_layers[i + 1], device=device) * 0.01))
            self.biases.append(torch.nn.Parameter(torch.zeros((1, hidden_layers[i + 1]), device=device)))

        # Last hidden to output
        self.weights.append(torch.nn.Parameter(torch.randn(hidden_layers[-1], output_size, device=device) * 0.01))
        self.biases.append(torch.nn.Parameter(torch.zeros((1, output_size), device=device)))


    def forward(self, inputs):
        x = inputs
        for i in range(len(self.weights)):
            x = torch.matmul(x, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                x = F.softmax(x, dim=1)
            else:
                x = F.relu(x)
        return x

    def backwards(self, y_true):
        # Number of samples
        samples = len(self.outputs[-1])

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            print("Chaning to Discrete Values")
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        dSoftMaxCrossEntropy = self.outputs[-1].copy()
        # Calculate gradient
        dSoftMaxCrossEntropy[range(samples), y_true] -= 1
        # Normalize gradient
        dSoftMaxCrossEntropy = dSoftMaxCrossEntropy / samples
        # print(dSoftMaxCrossEntropy)

        # print(dSoftMaxCrossEntropy)
        # Calculate gradients -> Calcualte derivative of weights, biases, and inputs (to continue to backpropagate)
        dInputs = np.dot(dSoftMaxCrossEntropy.copy(), self.weights[-1].T)

        dWeights = np.dot(self.outputs[-3].T, dSoftMaxCrossEntropy.copy())
        dBiases = np.sum(dSoftMaxCrossEntropy.copy(), axis=0, keepdims=True)
        self.gradientsWeights = [dWeights] + self.gradientsWeights
        self.gradientsBiases = [dBiases] + self.gradientsBiases


        i = -3
        j = -1
        for _ in range(len(self.hidden_layers)):
            i -= 1
            j -= 1
            
            # ReLU activation Function
            dInputsReLU = dInputs.copy()
            dInputsReLU[self.outputs[i] <= 0] = 0
            
            i -= 1
            dInputs = np.dot(dInputsReLU, self.weights[j].T)
            dWeights = np.dot(self.outputs[i].T, dInputsReLU)
            dBiases = np.sum(dInputsReLU, axis=0, keepdims=True)
            self.gradientsWeights = [dWeights] + self.gradientsWeights
            self.gradientsBiases = [dBiases] + self.gradientsBiases

    def updateParams(self, lr=0.05, decay=1e-7):
        lr = lr * (1. / (1. + decay * self.iterations))

        for i in range(len(self.weights)-1):
            if i != len(self.weights)-1:
                assert self.weights[i].shape == self.gradientsWeights[i].shape
                self.weights[i] += -lr*self.gradientsWeights[i]
        
        for i in range(len(self.biases)-1):
            if i != len(self.biases)-1:
                assert self.biases[i].shape == self.gradientsBiases[i].shape
                self.biases[i] += -lr*self.gradientsBiases[i]
        
        self.iterations += 1

# LossCategoricalCrossEntropy implementation
def LossCategoricalCrossEntropy(yPred, yTrue):
    # If predicted class has a prediction of 0% likelihood this prevents log(0), which would be infinity
    yPred = np.clip(yPred, 1e-10, 1 - 1e-10)

    # We calculate the sum of the log losses
    loss = -np.sum(yTrue * np.log(yPred), axis=1)

    # We calculate the average loss - this depends on the number of samples
    # So the return loss is the average loss not the summed up loss (which took me a while to understand)
    average_loss = np.mean(loss)

    return average_loss

def sparse_to_one_hot(sparse_labels, num_classes):
    one_hot_encoded = np.zeros((len(sparse_labels), num_classes))
    one_hot_encoded[np.arange(len(sparse_labels)), sparse_labels] = 1
    return one_hot_encoded

# Extract MNIST image files
def extract_images(filename):
    with gzip.open(filename, 'rb') as f:
        _ = np.frombuffer(f.read(16), dtype=np.uint8)  # Skip header
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
    images = images.reshape(-1, 28, 28)
    return images

# Extract MNIST label files
def extract_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _ = np.frombuffer(f.read(8), dtype=np.uint8)  # Skip header
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels

# Initialize model
modelMNIST = NeuralNetworkTorch(hidden_layers=[256], device=device)

# Load weights
model_weights = [
    torch.from_numpy(np.load("neuralFromScratch/modelWeights/layer_stack.1.weight.npy")).T.to(device),
    torch.from_numpy(np.load("neuralFromScratch/modelWeights/layer_stack.3.weight.npy")).T.to(device)
]
model_biases = [
    torch.from_numpy(np.load("neuralFromScratch/modelWeights/layer_stack.1.bias.npy")).unsqueeze(0).to(device),
    torch.from_numpy(np.load("neuralFromScratch/modelWeights/layer_stack.3.bias.npy")).unsqueeze(0).to(device)
]

# Set weights and biases
for i in range(len(model_weights)):
    modelMNIST.weights[i] = model_weights[i]
    modelMNIST.biases[i] = model_biases[i]

if __name__ == "__main__":
    # Load test data
    test_images = extract_images("neuralFromScratch/MNISTdata/t10k-images-idx3-ubyte.gz")
    test_labels = extract_labels("neuralFromScratch/MNISTdata/t10k-labels-idx1-ubyte.gz")
