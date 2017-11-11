'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):

        # Query a single test point using the k-NN algorithm

        # Get the distances from self to every test point in the dataset
        distances = self.l2_distance(test_point)

        # Sort the distances to find the points with least distance -> greatest distance
        ordered_indices = np.argsort(distances)

        # Populate k nearest neighbor inside an array
        k_nearest_digits = []

        # Iterate for upto k times and add the nearest digit value
        for i in range (0,k):
            # Get the index from the order
            nearest_neighbor_index = ordered_indices[i]

            # Get the nearest neighbor digit for the index
            nearest_neighbor_digit= self.train_labels[nearest_neighbor_index]

            # Push the nearest neighbor digit into the array
            k_nearest_digits.append(nearest_neighbor_digit)

        # Decide what the answer is by using calculating the most frequent digit among the k neighbors
        digit = self.digit_decider(k_nearest_digits, k)

        return digit

    def digit_decider(self, k_nearest_digits, k):

        # Obtain the unique neighbors list and their frequency (counts) -- sorted
        unique_digits, unique_indices, unique_digit_counts = np.unique(k_nearest_digits, return_counts=True,
                                                                return_inverse = True)
        # Find the most frequent neighbor
        maximum_digit_count = np.amax(unique_digit_counts)

        # Check if there are more values that are as frequent (i.e. is there a tie ?)
        number_of_maximum_frequency_occurences = list(unique_digit_counts).count(maximum_digit_count)

        # If there happens to be a a tie
        if number_of_maximum_frequency_occurences != 1:
            # Perform the analysis again for the last neighbor removed
            return self.digit_decider(k_nearest_digits[:k-1],k-1)

        # If there is no tie, return the unique digit with the highest frequency count
        else:
            return unique_digits[np.argmax(np.bincount(unique_indices))]


def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        pass

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    number_of_evaluated_samples = 0
    number_of_correctly_evaluated_samples = 0

    for i in range (len(eval_data)):

        # Make a prediction for the given data sample
        prediction = knn.query_knn(eval_data[i],k)

        # If your answer is correct, add to the correct count
        if prediction == eval_labels[i]:
            number_of_correctly_evaluated_samples += 1

        # Increase the evaulated samples count
        number_of_evaluated_samples +=1

    return (number_of_correctly_evaluated_samples/number_of_evaluated_samples)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    a = classification_accuracy(knn, 15, test_data, test_labels)
    print(a)


if __name__ == '__main__':
    main()