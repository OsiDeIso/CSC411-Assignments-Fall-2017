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
P_Y = 0.10

# Copied from as Q2_1: go through each label and see if it matches
# If it does, increase the count. At the end, get the ratio for accuracy
def classification_accuracy(pred_labels, eval_labels):

    number_of_evaluated_samples = 0
    number_of_correctly_evaluated_samples = 0

    for i in range(eval_labels.shape[0]):
        if pred_labels[i] == eval_labels[i]:
            number_of_correctly_evaluated_samples += 1

        number_of_evaluated_samples += 1

    accuracy = number_of_correctly_evaluated_samples/number_of_evaluated_samples

    return accuracy

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
            # Note constants have been calculated out
            top = total_binarized_features[j] + 1
            bottom = np.asarray(indices).shape[0] + 2
            eta[i][j] = top / bottom

    return eta

# Copied Entirely from Q2_0 where explained
def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
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
    # generated_data = binarize_data(eta)

    # binarize based on the piazza post statements (np.binomial)
    generated_data = np.zeros((eta.shape[0],eta.shape[1]))

    for i in range(eta.shape[0]):
        for j in range(eta.shape[1]):
            # Using a binomial distribution to generate data from the eta with a value of 1
            generated_data[i][j] = np.random.binomial(1, eta[i][j])

    plot_images(generated_data)
    return

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    gen_likelihood = np.zeros((bin_digits.shape[0], 10))

    for i in range(bin_digits.shape[0]):
        # Equation is log
        bin_digits_i = bin_digits[i]

        for j in range(eta.shape[0]):
            # part 1: ((nkj)^(bj)
            part_1 = (eta[j] ** bin_digits_i)

            # part 2:* (1 - nkj)(1-bj))
            part_2 = ((1 - eta[j]) ** (1 - bin_digits_i))

            part_before_multiplicand = part_1 * part_2
            part_before_multiplicand = np.log(part_before_multiplicand)
            gen_likelihood[i][j] = np.sum(part_before_multiplicand)

    return gen_likelihood

# Same as q2_2 conditional likelihood - replace means, covariances with eta
def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    # Make a conditional likelihood array
    cond_likelihood = np.zeros((bin_digits.shape[0], 10))

    # Obtain the generative likelihood
    gen_likelihood = generative_likelihood(bin_digits,eta)

    for i in range(gen_likelihood.shape[0]):
        p_x_mu_sigma = 0

        for j in range(gen_likelihood[i].shape[0]):
            p_x_mu_sigma += np.exp(gen_likelihood[i][j])

        p_x_mu_sigma *= P_Y

        for j in range(gen_likelihood[i].shape[0]):
            cond_likelihood[i][j] = gen_likelihood[i][j] - np.log(p_x_mu_sigma) + np.log(P_Y)

    return cond_likelihood

# Similar to q2_2's conditional likelihood
def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return

    # Can be copied from

    # Make a likelihood list to populate conditional likelihoods in
    likelihoods = []

    # Make the output classes into integers for indexing purposes
    # within the conditional likelihood
    integer_labels = []
    for i in range(labels.shape[0]):
        integer_labels.append(int(labels[i]))

    # Iterate through every conditional likelihood matrix
    for i in range(cond_likelihood.shape[0]):
        # Add the likelihoods for a given iteration for a given iteration's label
        # What is the likelihood for a given a given digit sample, for a given digit outcome
        likelihoods.append(cond_likelihood[i][integer_labels[i]])

    # Take the average of the obtained list
    avg_cond_likelihood = np.average(likelihoods)

    return avg_cond_likelihood


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class

    # Get the highest probability from across all potential digit labels (axis -->)
    most_likely_class = np.argmax(cond_likelihood, axis=1)

    return most_likely_class


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation

    # Obtain average likelihoods for train and test data sets
    average_likelihood_for_training_data = avg_conditional_likelihood(train_data, train_labels, eta)
    print("Average Conditional (Log) Likelihood for Training data: ",average_likelihood_for_training_data)

    average_likelihood_for_test_data = avg_conditional_likelihood(test_data, test_labels, eta)
    print("Average Conditional (Log) Likelihood for Test data: ", average_likelihood_for_test_data)

    # Make Predictions for the train and test sets
    prediction_training = classify_data(train_data, eta)
    accuracy_for_training_data = classification_accuracy(prediction_training,train_labels)
    print("Classification Accuracy for Training Data: ",accuracy_for_training_data)

    prediction_testing = classify_data(test_data, eta)
    accuracy_for_testing_data = classification_accuracy(prediction_testing,test_labels)
    print("Classification Accuracy for Testing Data: ",accuracy_for_testing_data)

    # Plot the Covariances Diagonals at the end as it blocks all other executions
    plot_images(eta)
    generate_new_data(eta)


if __name__ == '__main__':
    main()
