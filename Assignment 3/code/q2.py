import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

# Constants for Q 2_1
ALPHA = 1.0
B_0 = 0.0
B_1 = 0.9

# Constants for Q 2_2 and Q 2_3
B_00 = 0.0
B_01 = 0.1
ALPHA_1 = 0.05
C = 1.0
BATCH_SIZE = 100
ITERATIONS = 500

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return

        self.vel = ((-1 * self.lr) * grad) + (self.beta * self.vel)
        params += self.vel

        # the updated parameters
        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count + 1)
        self.mean_hinge_loss = []
        self.myvar = 0

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        hinge_loss = []

        # Obtain w * X
        w_x = np.dot(X, self.w)

        for i in range(X.shape[0]):
            # Loss is the maximum b/w two inputs below
            hinge_loss.append(np.amax([1 - w_x[i] * y[i], 0]))

        # Append to mean array for later usage (loss calculation)
        self.mean_hinge_loss.append(np.mean(hinge_loss))

        return hinge_loss

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        hinge_losses = np.asarray(self.hinge_loss(X,y))

        w_star = []
        for i in range(hinge_losses.shape[0]):
            if hinge_losses[i] > 0:
                w_star.append(np.dot(y[i], X[i]))
            else:
                w_star.append(np.zeros(X.shape[1]))

        sum = np.sum(np.asarray(w_star), axis=0)

        return self.w - (self.c / X.shape[0]) * sum

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1

        # Declare a fixed array of integers
        classifier = np.ndarray(shape=(X.shape[0]), dtype=int)

        # For entire data matrix X
        for i in range(X.shape[0]):
            # Calculate dot product of w_T and x
            w_T_x = np.dot(np.transpose(self.w[1:]), X[i])

            # If less than 0, then classify as 1
            if w_T_x >= 0 :
                classifier[i] = 1

            # If greater than 0, classify as -1
            else:
                classifier[i] = -1

        # Return Classified Array
        return classifier


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''

    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history

        # Obtain previous iteration's history's weight
        previous_history = w_history[_]

        # Obtain previous iteration's gradient weight
        previous_gradient = func_grad(previous_history)

        # Run the optimization function to obtain current iteration's weight
        current = optimizer.update_params(previous_history, previous_gradient)

        # Add it to the history
        w_history.append(current)

    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''

    # Make a batch sampler
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)

    # Make an SVM
    svm = SVM(penalty, train_data.shape[1])

    # Go through every iteration to get a batch
    for i in range(iters):
        x_batch, y_batch = batch_sampler.get_batch()
        svm.w = optimizer.update_params(svm.w, svm.grad(np.hstack((np.ones((x_batch.shape[0], 1)), x_batch)), y_batch))

    return svm

# Taken from Previous Assignment: Calculate accuracy based on labels
def calculate_accuracy(predicted_labels, actual_labels):
    total_labels = len(predicted_labels)
    correct_labels = 0

    # Go through entire array and add up correct label predictions
    for i in range(predicted_labels.shape[0]):
        if predicted_labels[i] == actual_labels[i]:
            correct_labels = correct_labels + 1

    accuracy = correct_labels / total_labels
    return accuracy


def q2_1():
    # 2.1 SGD With Momentum Plot

    # Optimize for first gradient
    gradient_1 = GDOptimizer(ALPHA, B_0)
    weights_1 = optimize_test_function(gradient_1)

    # Optimize for second gradient
    gradient_2 = GDOptimizer(ALPHA, B_1)
    weights_2 = optimize_test_function(gradient_2)

    # Plot Values
    label_1 = 'BETA = ' + str(B_0)
    plt.plot(weights_1, label=label_1, color='black')

    label_2 = 'BETA = ' + str(B_1)
    plt.plot(weights_2, label=label_2, color='grey')

    # plt.title("Gradient Descent Function for f(w) = 0.01w^2")
    plt.title(r'Gradient Descent for $f(w) = 0.01w^2$')
    plt.xlabel("Time Step")
    plt.ylabel(r'$w_t$')
    plt.legend()
    plt.show()

    return


# Plot Map for Q 2.3
def plot_q_2_3(svm_w, beta):

    # Plot Title
    plt.title(r'SVM Weights for $\beta = $'+ str(beta))
    plt.imshow(svm_w, cmap='Greys_r')
    plt.colorbar()
    plt.show()

    return

# Generates SVM based on data, targets and predefined constants. Returns train, test loss and accuracy
def model_generator(train_data, train_targets, test_data, test_targets, beta):

    # Obtain an SVM for training data and test data(s)

    svm_training = optimize_svm(train_data, train_targets, C , GDOptimizer(lr= ALPHA_1 , beta=beta),
                       BATCH_SIZE, ITERATIONS)

    svm_test = optimize_svm(test_data, test_targets, C , GDOptimizer(lr= ALPHA_1, beta=beta),
                            BATCH_SIZE,ITERATIONS)

    # Obtain training predictions
    training_predictions = svm_training.classify(train_data)

    # Obtain test predictions
    test_predictions = svm_training.classify(test_data)

    # Calculate Training Loss
    training_loss = np.mean(svm_training.mean_hinge_loss)

    # Calculate Test Loss
    test_loss = np.mean(svm_test.mean_hinge_loss)

    # Obtain Classification Accuracy for Training Data
    classification_accuracy_train = calculate_accuracy(training_predictions, train_targets)

    # Obtain Classification Accuracy for Test Data
    classification_accuracy_test = calculate_accuracy(test_predictions, test_targets)

    # Obtain SVM w values for plot in 28 by 28 grid
    svm_w = np.reshape(svm_training.w[1:], (-1, 28))

    return svm_w, training_loss, test_loss, classification_accuracy_train, classification_accuracy_test


def q_2_3():

    # Load data
    train_data, train_targets, test_data, test_targets = load_data()

    # Generate the models for BETA = 0 and BETA = 0.1

    m1_svm_w, m1_training_loss, m1_test_loss, m1_classification_accuracy_train, m1_classification_accuracy_test = \
        model_generator(train_data, train_targets, test_data, test_targets, B_00)

    m2_svm_w, m2_training_loss, m2_test_loss, m2_classification_accuracy_train, m2_classification_accuracy_test = \
        model_generator(train_data, train_targets, test_data, test_targets, B_01)

    print("Training Loss for Model 1: ", m1_training_loss)
    print("Training Loss for Model 2: ", m2_training_loss)

    print("Test Loss for Model 1: ", m1_test_loss)
    print("Test Loss for Model 2: ", m2_test_loss)

    print("Training Classification Accuracy for Model 1: ", m1_classification_accuracy_train)
    print("Training Classification Accuracy for Model 2: ", m2_classification_accuracy_train)

    print("Test Classification Accuracy for Model 1: ", m1_classification_accuracy_test)
    print("Test Classification Accuracy for Model 2: ", m2_classification_accuracy_test)

    plot_q_2_3(m1_svm_w, B_00)
    plot_q_2_3(m2_svm_w, B_01)

    return

def main():

    # Perform Plot for Q 2.1
    q2_1()

    # Perform Training, Test and Plots for Q 2.3
    q_2_3()

    return


if __name__ == '__main__':
    main()
