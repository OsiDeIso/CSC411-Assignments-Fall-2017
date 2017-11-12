'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

IMAGE_SIZE = 8
BETA_ALPHA = 2
BETA_BETA = 2

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))

    for i in range(eta.shape[0]):

        # Get the indices that have the digit value
        indices = []
        for j in range(train_labels.shape[0]):
            if train_labels[j] == i:
                indices.append(j)

            # Sum the binary features for every feature
            total_binarized_features = np.sum(train_data[indices], axis=0)

        for j in range(total_binarized_features.shape[0]):

            # Getting the n for bernoulli's classifier
            top = total_binarized_features[j] + 1
            bottom = np.asarray(indices).shape[0] + BETA_ALPHA
            eta[i][j] = top / bottom

    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    # Copied Entirely from Q2_0 where explained
    means = []
    for i in range(10):
        img_i = class_images[i]
        img_i = img_i.reshape(IMAGE_SIZE,IMAGE_SIZE)
        means.append(img_i)

    means=np.asarray(means)
    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

    return


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = binarize_data(eta)
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''

    likelihood = np.zeros((len(bin_digits), 10))
    for i, bin_digit in enumerate(bin_digits):
        for j, e in enumerate(eta):
            likelihood[i][j] = np.sum(np.log((e**bin_digit) * (1-e)**(1-bin_digit)))
    return likelihood


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    # Can be copied from
    # p(y|x) = p(x|y)p(y) / p(x)
    # p(y) = 1/10
    # p(x|y) = generative_likelihood
    # p(x) = sum  of p(x|y) * p(y)
    likelihood = np.zeros((len(bin_digits), 10))
    gl = generative_likelihood(bin_digits, eta)
    for i, g in enumerate(gl):
        p_x = sum(np.exp(gf) for gf in g) * (1 / 10)
        for j, gen_feature in enumerate(g):
            likelihood[i][j] = gen_feature + np.log(1 / 10) - np.log(p_x)

    return likelihood


def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return

    # Can be copied from

    cond_likelihood = conditional_likelihood(bin_digits, eta)
    all = []
    for i in range(len(bin_digits)):
        all.append(cond_likelihood[i][labels[i].__int__()])

    # Compute as described above and return
    return np.mean(all)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class

    # Can be copied

    return np.argmax(cond_likelihood, axis=1)

def calculate_accuracy(predicted_labels, actual_labels):

    return (predicted_labels == actual_labels).sum() / len(actual_labels)


def main():
    # train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    # train_data, test_data = binarize_data(train_data), binarize_data(test_data)
    #
    # # Fit the model
    # eta = compute_parameters(train_data, train_labels)
    #
    # # Evaluation
    # plot_images(eta)
    #
    # generate_new_data(eta)

    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    # Compute Average Likelihoods
    average_likelihood_training = avg_conditional_likelihood(train_data, train_labels, eta)
    average_likelihood_test = avg_conditional_likelihood(test_data, test_labels, eta)
    # Compute classification predictions
    prediction_training = classify_data(train_data, eta)
    prediction_test = classify_data(test_data, eta)
    # Compute accuracy
    accuracy_training = calculate_accuracy(prediction_training, train_labels)
    accuracy_test = calculate_accuracy(prediction_test, test_labels)
    print("Average Conditional Likelihood:", "\nTraining:", average_likelihood_training, "\nTest:",
          average_likelihood_test)
    print("Classification Accuracy:", "\nTraining:", accuracy_training, "\nTest:", accuracy_test)

    # plot data
    plot_images(eta)
    generate_new_data(eta)


if __name__ == '__main__':
    main()
