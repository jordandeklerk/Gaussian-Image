import numpy as np
from tqdm import tqdm
from utils import psnr
from models import NeuralNetwork

def NN_experiment(X_train, y_train, X_test, y_test, input_size, num_layers, \
                  hidden_size, hidden_sizes, output_size, epochs, \
                  learning_rate, opt):

    net = NeuralNetwork(input_size, hidden_sizes, output_size, num_layers, opt)

    train_loss = np.zeros(epochs)
    train_psnr = np.zeros(epochs)
    test_psnr = np.zeros(epochs)
    predicted_images = np.zeros((epochs, y_test.shape[0], output_size))

    for epoch in tqdm(range(epochs)):

        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        prediction = net.forward(X_train_shuffled)
        loss = net.backward(y_train_shuffled)
        net.update(learning_rate)

        train_loss[epoch] = loss
        train_psnr[epoch] = psnr(y_train_shuffled, prediction)

        test_prediction = net.forward(X_test)
        test_psnr[epoch] = psnr(y_test, test_prediction)
        predicted_images[epoch] = test_prediction

    return net, train_psnr, test_psnr, train_loss, predicted_images