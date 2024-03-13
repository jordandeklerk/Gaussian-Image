from utils import *

size = 32
train_data, test_data = get_image(size)

# Low Resolution - SGD - None Mapping
B_dict = get_B_dict(size)
X_train, y_train, X_test, y_test = get_input_features(B_dict, 'none')

input_size = X_train.shape[1]
opt = "SGD"

net, train_psnr, test_psnr, train_loss, predicted_images = NN_experiment(X_train, y_train, X_test, y_test, input_size, num_layers, hidden_size, hidden_sizes, output_size, epochs, learning_rate, opt)

plot_training_curves(train_loss, train_psnr, test_psnr)
plot_reconstruction(net.forward(X_test), y_test)
plot_reconstruction_progress(predicted_images, y_test)

# Low Resolution - Adam - None Mapping
B_dict = get_B_dict(size)
X_train, y_train, X_test, y_test = get_input_features(B_dict, 'none')

input_size = X_train.shape[1]
opt = "Adam"

net, train_psnr, test_psnr, train_loss, predicted_images = NN_experiment(X_train, y_train, X_test, y_test, input_size, num_layers, hidden_size, hidden_sizes, output_size, epochs, learning_rate, opt)

plot_training_curves(train_loss, train_psnr, test_psnr)
plot_reconstruction(net.forward(X_test), y_test)
plot_reconstruction_progress(predicted_images, y_test)

def train_wrapper(mapping, size, opt):

    B_dict = get_B_dict(size)
    X_train, y_train, X_test, y_test = get_input_features(B_dict, mapping)

    input_size = X_train.shape[1]

    net, train_psnr, test_psnr, train_loss, predicted_images = NN_experiment(
        X_train, y_train, X_test, y_test,
        input_size, num_layers, hidden_size, hidden_sizes,
        output_size, epochs, learning_rate, opt
    )

    return {
        'net': net,
        'train_psnrs': train_psnr,
        'test_psnrs': test_psnr,
        'train_loss': train_loss,
        'pred_imgs': predicted_images
    }

# All Mappings and Optimizers - Low Resolution 
mappings = ['none', 'basic', 'gauss_1.0']
opts = ['SGD', 'Adam']
size = 32

results = {}
for mapping in mappings:
    for opt in opts:
        key = f"{mapping}_{opt}"
        results[key] = train_wrapper(mapping, size, opt)

# All Mappings and Optimizers - High Resolution
size = 128
train_data, test_data = get_image(size)

epochs = 4000
mappings = ['none', 'basic', 'gauss_1.0']
opts = ['SGD', 'Adam']
size = 128

results = {}
for mapping in mappings:
    for opt in opts:
        key = f"{mapping}_{opt}"
        results[key] = train_wrapper(mapping, size, opt)