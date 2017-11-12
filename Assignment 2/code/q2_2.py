'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

IMAGE_SIZE = 8
P_Y = 0.10

def classification_accuracy(pred_labels, eval_labels):

    # Same thing as Q2_1: go through each label and see if it matches
    # If it does, increase the count. At the end, get the ratio for accuracy

    number_of_evaluated_samples = 0
    number_of_correctly_evaluated_samples = 0

    for i in range(eval_labels.shape[0]):
        if pred_labels[i] == eval_labels[i]:
            number_of_correctly_evaluated_samples += 1

        number_of_evaluated_samples += 1

    accuracy = number_of_correctly_evaluated_samples/number_of_evaluated_samples

    return accuracy

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

    return covariances + 0.01 * np.identity(IMAGE_SIZE * IMAGE_SIZE)

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

    # Provided Equation divided into 3 parts for debugging and simplification:
    # 0: generative_likelihood = log (part 1 * part 2 * part 3)
    # 1: (2 * pi)^(-d/2) *
    # 2: |sigma(k)|^(-1/2) *
    # 3: exp ^ (-0.5) * (x - mu(k)) * (sigma(k)^(-1)) * (x - mu(k))

    # 0: generative_likelihood
    gen_likelihood = np.zeros((digits.shape[0], 10))

    # 1: (2 * pi) ^ (-d / 2) where d = (# of features)
    part_1 = 2 * np.pi * ((IMAGE_SIZE ** 2)/2)

    # iterate over every data sample (n)
    for i in range(digits.shape[0]):
        # 3: (x - mu(k))
        x_minus_mu = digits[i] - means
        for k in range(x_minus_mu.shape[0]):

            # 2: |sigma(k)|^(-1/2)
            sigma_k_determinant = np.linalg.det(covariances[k])
            sigma_k_determinant_inverse_root = (sigma_k_determinant ** -0.5)
            part_2 = sigma_k_determinant_inverse_root

            # 3: (x - mu(k))
            x_minus_mu_k = x_minus_mu[k]

            # 3: (sigma(k)^(-1))
            sigma_k_inverse = np.linalg.inv(covariances[k])

            # 3: (-0.5) * (x - mu(k)) * (sigma(k)^(-1)) * (x - mu(k))
            power_value = np.dot(np.transpose(x_minus_mu_k),sigma_k_inverse)
            power_value = np.dot(power_value, x_minus_mu_k)
            power_value = power_value * -0.5
            part_3 = np.exp(power_value)

            # Get the log liklehood
            # 0: generative_likelihood = log (part 1 * part 2 * part 3)
            gen_likelihood[i][k] = np.log(part_1 * part_2 * part_3)

    return gen_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    # From Question 1:
    # p(y|x, mu, Sigma) = p(x|y, mu, Sigma) * p(y) / p(x, mu, Sigma)
    # Applying logs makes:
    # log p(y|x, mu, Sigma) = log p(x|y, mu, Sigma) - log p(x, mu, Sigma) + log p(y)
    # Conditional Likelihood = Generative Likelihood - log (p(x, mu, Sigma) + log p(y)

    # Splitting into multiple parts:
    # 1: p(x|y, mu, Sigma) * p(y)
    # 2: p(x, mu, Sigma)



    # Make a conditional likelihood array
    cond_likelihood = np.zeros((digits.shape[0], 10))

    # Obtain the generative likelihood
    gen_likelihood = generative_likelihood(digits,means,covariances)

    for i in range(gen_likelihood.shape[0]):
        p_x_mu_sigma = 0

        for j in range(gen_likelihood[i].shape[0]):
            p_x_mu_sigma += np.exp(gen_likelihood[i][j])

        p_x_mu_sigma *= P_Y

        for j in range(gen_likelihood[i].shape[0]):
            cond_likelihood[i][j] = gen_likelihood[i][j] - np.log(p_x_mu_sigma) + np.log(P_Y)

    return cond_likelihood

# Henrys
def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return

    # Make a likelihood list to populate conditional liklihoods in
    likelihoods = []

    # Iterate through every conditional likelihood matrix
    integer_labels = []
    for i in range(labels.shape[0]):
        integer_labels.append(int(labels[i]))

    for i in range(cond_likelihood.shape[0]):
        # Add the likelihoods for a given iteration for a given iteration's label
        likelihoods.append(cond_likelihood[i][integer_labels[i]])

    # Take the average of the obtained list
    avg_cond_likelihood = np.average(likelihoods)

    return avg_cond_likelihood


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute and return the most likely class
    # Get the highest probability from across all potential digit labels (axis -->)
    most_likely_class = np.argmax(cond_likelihood, axis=1)

    return most_likely_class


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    # means = compute_mean_mles(train_data, train_labels)
    # covariances = compute_sigma_mles(train_data, train_labels)
    # generative_likelihood(train_data, means, covariances)

    # Evaluation

    #n2c

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Compute Average Likelihoods
    average_likelihood_training = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    average_likelihood_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    # Compute classification predictions
    prediction_training = classify_data(train_data, means, covariances)
    prediction_test = classify_data(test_data, means, covariances)
    # Compute accuracy
    accuracy_training = classification_accuracy(prediction_training, train_labels)
    accuracy_test = classification_accuracy(prediction_test, test_labels)
    print("Average Conditional Likelihood:", "\nTraining:", average_likelihood_training, "\nTest:", average_likelihood_test)
    print("Classification Accuracy:", "\nTraining:", accuracy_training, "\nTest:", accuracy_test)
    # Plot covariance diagonals
    plot_cov_diagonal(covariances)


if __name__ == '__main__':
    main()