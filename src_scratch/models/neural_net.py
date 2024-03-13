"""Neural network model."""

from typing import Sequence
import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.m = {}  
        self.v = {} 

        self.params = {}
        self.gradients = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            if opt == 'Adam':
                self.m["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.m["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])

                self.v["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.v["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        output = np.dot(X, W) + b

        return output
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg: float, N: int):
        """Gradient of linear layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
            de_dz: gradient of the loss with respect to the output of this layer
            reg: regularization strength
            N: number of samples
        Returns:
            de_dw: Gradient of the loss with respect to the weights
            de_db: Gradient of the loss with respect to the biases
            de_dx: Gradient of the loss with respect to the input
        """
        de_dw = np.dot(X.T, de_dz) / N + reg * W
        de_db = np.sum(de_dz, axis=0) / N
        de_dx = np.dot(de_dz, W.T)
        
        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the gradient
        """
        grad = np.zeros_like(X)
        grad[X > 0] = 1

        return grad


    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.
        Parameters:
            x: Input data.
        Returns:
            Sigmoid activation.
        """
        pos_mask = (x >= 0)
        neg_mask = (x < 0)

        z = np.zeros_like(x, dtype=np.float64)
        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        z[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

        return z


    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of the sigmoid function.
        Parameters:
            X: Input data.
        Returns:
            Gradient of the sigmoid.
        """
        sig = self.sigmoid(X)

        return sig * (1 - sig)


    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Mean Squared Error loss.
        Parameters:
            y: True values.
            p: Predicted values.
        Returns:
            MSE loss.
        """
        return np.mean((y - p) ** 2)

    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradient of the Mean Squared Error loss.
        Parameters:
            y: True values.
            p: Predicted values.
        Returns:
            Gradient of the MSE loss.
        """
        n = y.shape[0]

        return (2/n) * (p - y)

    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradient of MSE loss with respect to the input of the sigmoid function.
        Parameters:
            y: True values.
            p: Predicted values (after applying sigmoid).
        Returns:
            Gradient of the loss with respect to the input of the sigmoid.
        """
        n = y.shape[0]
        sig_grad = p * (1 - p)  
        mse_grad = (2 / n) * (p - y)  

        return mse_grad * sig_grad


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or testing sample.
        Returns:
            Matrix of shape (N, C) where each row contains the output for each sample.
        """
        self.outputs = {}  
        self.outputs['Z0'] = X

        for i in range(1, self.num_layers):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            Z = self.linear(W, self.outputs['Z' + str(i-1)], b)  
            A = self.relu(Z)  
            self.outputs['Z' + str(i)] = Z  
            self.outputs['A' + str(i)] = A  

        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        Z = self.linear(W, self.outputs['A' + str(self.num_layers-1)], b)  
        Y_hat = self.sigmoid(Z) 
        self.outputs['Z' + str(self.num_layers)] = Z
        self.outputs['A' + str(self.num_layers)] = Y_hat

        return Y_hat


    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients for each parameter.
        Parameters:
            y: The true values.
        Returns:
            Total loss for this batch of training samples.
        """
        self.gradients = {}
        dA = self.mse_sigmoid_grad(y, self.outputs['A' + str(self.num_layers)])
        
        for i in reversed(range(1, self.num_layers + 1)):
            if i == self.num_layers:
                dZ = dA * self.sigmoid_grad(self.outputs['Z' + str(i)])
            else:
                dZ = dA * self.relu_grad(self.outputs['Z' + str(i)])
            
            if i == 1:
                A_prev = self.outputs['Z0']
            else:
                A_prev = self.outputs['A' + str(i-1)]
            
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            if i > 1:
                dA_prev = np.dot(dZ, self.params['W' + str(i)].T)
                dA = dA_prev
            
            self.gradients['W' + str(i)] = dW
            self.gradients['b' + str(i)] = db.squeeze()

        loss = self.mse(y, self.outputs['A' + str(self.num_layers)])

        return loss


    def update(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, opt="SGD"):
        if opt == "SGD":
            for i in range(1, self.num_layers + 1):
                self.params['W' + str(i)] -= lr * self.gradients['W' + str(i)]
                self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)]
        elif opt == "Adam":
            if not hasattr(self, 'm'):
                self.m, self.v = {}, {}
                for i in range(1, self.num_layers + 1):
                    self.m['W' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                    self.m['b' + str(i)] = np.zeros_like(self.params['b' + str(i)])
                    self.v['W' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                    self.v['b' + str(i)] = np.zeros_like(self.params['b' + str(i)])
                self.t = 0

            self.t += 1
            for i in range(1, self.num_layers + 1):
                self.m['W' + str(i)] = b1 * self.m['W' + str(i)] + (1 - b1) * self.gradients['W' + str(i)]
                self.m['b' + str(i)] = b1 * self.m['b' + str(i)] + (1 - b1) * self.gradients['b' + str(i)]
                self.v['W' + str(i)] = b2 * self.v['W' + str(i)] + (1 - b2) * (self.gradients['W' + str(i)] ** 2)
                self.v['b' + str(i)] = b2 * self.v['b' + str(i)] + (1 - b2) * (self.gradients['b' + str(i)] ** 2)

                m_hat_w = self.m['W' + str(i)] / (1 - b1 ** self.t)
                m_hat_b = self.m['b' + str(i)] / (1 - b1 ** self.t)
                v_hat_w = self.v['W' + str(i)] / (1 - b2 ** self.t)
                v_hat_b = self.v['b' + str(i)] / (1 - b2 ** self.t)

                self.params['W' + str(i)] -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
                self.params['b' + str(i)] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        return 

