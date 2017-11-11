'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

IMAGE_SIZE = 8


def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i

        # Take means for each feature for all samples
        verticalMeans = np.mean(i_digits,axis=0)

        # Put the average back to an 8-by-8 pixel array
        verticalMeans = verticalMeans.reshape(IMAGE_SIZE,IMAGE_SIZE)

        # Add them to the means array
        means.append(verticalMeans)

    means=np.asarray(means)
    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

if __name__ == '__main__':
    train_data, train_labels, _, _ = data.load_all_data_from_zip('a2digits.zip', 'data')
    plot_means(train_data, train_labels)
