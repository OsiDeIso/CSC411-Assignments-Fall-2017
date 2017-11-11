'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

IMAGE_SIZE = 8

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))

    # Compute means for every feature (64) for all samples
    # for a given digit
    for i in range(len(means)):

        # Get the indicies for every digit i
        digit_indicies = np.where(train_labels == i)

        # Find the mean of the each feature (downwards -> axis = 0)
        # where the sample data has an output of digit i
        means[i] = np.mean(train_data[digit_indicies], axis=0)

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''

    # Obtain means
    means = compute_mean_mles(train_data, train_labels)

    covariances = np.zeros((10, 64, 64))

    # Compute covariances

    # Iterate through every digit
    for i in range(covariances.shape[0]):

        # Obtain the the indicies in the training set
        # for the samples that contain the digit of interest
        digit_i_indcies = np.where(train_labels == i)
        digit_i_indcies = np.squeeze(digit_i_indcies)

        # For a given digit, iterate through every feature
        for j in range(covariances.shape[1]):

            # Obtain the mean for feature 1
            mean_for_digit_and_feature_1 = means[i][j]
            # print(i, j, mean_for_digit_and_feature_1)

            for k in range(covariances.shape[2]):

                # Obtain the mean for feature 2
                mean_for_digit_and_feature_2 = means[i][k]
                # print(i, j, k, mean_for_digit_and_feature_2)

                # For every sample calculate the covariance and add to sum
                sum = 0
                # print(digit_i_indcies.shape[0])
                for l in range(digit_i_indcies.shape[0]):

                    # Obtain [(x1 - mu1)]
                    difference_in_feature_1 = train_data[digit_i_indcies[l]][j] - mean_for_digit_and_feature_1

                    # Obtain [(x2 - mu2)]
                    difference_in_feature_2 = train_data[digit_i_indcies[l]][k] - mean_for_digit_and_feature_2

                    # Sum is (x1-mu1)*(x2-mu2)
                    sum += difference_in_feature_1 * difference_in_feature_2

                # Obtain the average by dividing by the number of samples taken E[(x1-mu1)*(x2-mu2)]
                sum = sum / digit_i_indcies.shape[0]

                # Put the sum in the covariance matrix
                covariances[i][j][k]= sum

    # Plotting for testing
    plot_cov_diagonal(covariances)

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side

    # Make an array to append final values into
    variances = []

    for i in range(covariances.shape[0]):
        # Obtain the diagonals - i.e. the variances
        diagonals = np.diag(covariances[i])

        # Reshape them
        diagonals = diagonals.reshape(IMAGE_SIZE,IMAGE_SIZE)

        # Append them to the Variances array
        variances.append(diagonals)

    # Plot using q2_0 style
    all_concat = np.concatenate(variances, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

    return

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    return None

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation

if __name__ == '__main__':
    main()