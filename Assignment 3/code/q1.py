'''
Question 1 Skeleton Code


'''

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

SGD_ITERATIONS = 50
NUM_FEATURES = 20
MAX_VALUE = 100
SGD_ITERATIONS_1 = 100

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

# Classification Accuracy from Assignment 1 and 2
def accuracy(predictions, targets):
    total = 0
    hit = 0
    for i in range(predictions.shape[0]):
        total = total + 1
        if predictions[i] == targets[i]:
            hit = hit + 1

    # Make accuracy a percentage (easier to compare)
    accuracy = (100.0 * hit) / total
    return accuracy

# Compute the confusion matrix, get the bigest confusion and the most confused pair
def confusion_matrix_processing(test_predictions, test_targets):

    # Make an empty matrix for computing confusion matrix
    confusion_matrix = np.zeros((NUM_FEATURES, NUM_FEATURES))

    # Make an empty matrix for computing the number of misclassified cases
    num_misclassified = np.zeros((NUM_FEATURES,))

    # Compute the Confusion Matrix

    # For the every box in the matrix
    for i in range(NUM_FEATURES):
        for j in range(NUM_FEATURES):

            # For every class, check if the predictions and targets are the row and column respectively
            for k in range(test_predictions.shape[0]):
                if test_targets[k] == j:
                    if test_predictions[k] == i:

                        # If so, increase the count
                        confusion_matrix[i][j] += 1


    # Search for the biggest confusion value in the matrix and the most confused pair
    max_associated_confusion = 0
    associated_i = -1
    associated_j = -1

    # For every Feature combination
    for j in range(NUM_FEATURES):
        for i in range(NUM_FEATURES):
            # If the features are not the same
            if i != j:
                # Biggest Confusion Values
                num_misclassified[j] += confusion_matrix[i][j]

                # Most confused pair

                # If the matrix has a higher count, reset the values
                if confusion_matrix[i][j] > max_associated_confusion:
                    max_associated_confusion = confusion_matrix[i][j]
                    associated_i = i
                    associated_j = j

    # Most confused feature class is the one with the highest mistakes
    most_confused_class = num_misclassified.argmax()

    # The number of mistakes
    number_of_confused_cases = num_misclassified.max()

    # Return All the information
    return confusion_matrix, most_confused_class, number_of_confused_cases, associated_i, associated_j, max_associated_confusion-1



# Plot Confusion Matrix
def plot_confusion_matrix(confusion_matrix):
    # Plotting details
    plt.title("Confusion Matrix of the SGD Classifier")
    plt.ylabel("Predicted Class")
    plt.xlabel("Actual Class")
    plt.imshow(confusion_matrix)

    # Show all indices frpm 1 till NUM_FEATURES
    plt.xticks(np.arange(0, confusion_matrix.shape[0], 1.0))
    plt.yticks(np.arange(0, confusion_matrix.shape[0], 1.0))

    plt.colorbar()
    plt.show()

    return


# Print the accuracy for a given model (Training and Test)
# Loss can be automatically calculated later
def print_acccuracy(model_name, model, train , train_target, test, test_target):

    # Calculate Accuracies
    train_accuracy = accuracy(model.predict(train), train_target)
    test_accuracy = accuracy(model.predict(test), test_target)

    # Print Accuracies
    print(model_name, "train accuracy: ", str(train_accuracy))
    print(model_name, "test accuracy: " + str(test_accuracy))

    return

if __name__ == '__main__':
    train_data, test_data = load_data()
    # train_bow, test_bow, feature_names_bow = bow_features(train_data, test_data)
    train_tf_idf, test_tf_idf, feature_names_tf_idf = tf_idf_features(train_data, test_data)

    # Baseline Model
    bnb_model = bnb_baseline(train_tf_idf, train_data.target, test_tf_idf, test_data.target)

    # Using Scikit learn's Stochastic Gradient Descent Classifier
    sgd_classifier = SGDClassifier(max_iter= SGD_ITERATIONS)
    sgd_classifier.fit(train_tf_idf, train_data.target)

    # Using Scikit learn's Logistic Regression Classifier
    logistic_regr = LogisticRegression()
    logistic_regr.fit(train_tf_idf, train_data.target)

    # Using Scikit learn's Random Forest Classifier
    random_forest = RandomForestClassifier()
    random_forest.fit(train_tf_idf, train_data.target)

    # Printing out the results of the models chosen (3)
    print_acccuracy("Stochastic Gradient Descent Classifier", sgd_classifier, train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    print_acccuracy("Logistic Regression Classifier", logistic_regr, train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    print_acccuracy("Random Forest Classifier", random_forest, train_tf_idf, train_data.target, test_tf_idf, test_data.target)

    # BEST CLASSIFIER: Using Scikit learn's SGD Classifier for 100 iterations for the confusion matrix
    sgd_classifier_best = SGDClassifier(max_iter= SGD_ITERATIONS_1)
    sgd_classifier_best.fit(train_tf_idf, train_data.target)
    sgd_classifier_predictions = sgd_classifier_best.predict(test_tf_idf)

    # Obtain the confusion matrix, most confused class, how many confusions said class has
    # the most confused pair and the number of confusions for the given pair
    # the pair confusion value is Cij
    confusion_matrix, most_confused_class, confusion_value, pair_i, pair_j, pair_confusion_value = \
        confusion_matrix_processing(sgd_classifier_predictions, test_data.target)

    print("The BEST Classifier is the Stochastic Gradient Descent Classifier")

    # Most confused class
    print("Most Confused Class Index:", most_confused_class)
    print("Most Confused Class Feature Name:", train_data.target_names[most_confused_class])
    print("Most Confused Class's Number of Cases:", confusion_value)

    # Most confused class pair
    print("Most Confused Class Pair Indices:", pair_i, "and", pair_j)
    print("Most Confused Class Pair Names:", train_data.target_names[pair_i], "and", train_data.target_names[pair_j])
    print("Most Confused Class Pair's Number of Cases:", pair_confusion_value)


    # Plot the Confusion Matrix
    plot_confusion_matrix(confusion_matrix)
