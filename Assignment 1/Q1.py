from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

#inputs: number of indices (samples), what percentage of them to train
#outputs: the training indices, testing indices
def randomize_indices(num_samples,percent_train):

    num_train = round(num_samples * percent_train);
    
    #remaining are left for testing
    num_test = round(num_samples - num_train)
    
    #array with all indices
    indices_total = np.arange(num_samples)
    
    #generate random indices without replacing (i.e. 80% of the total)
    indices_train = (np.random.choice(num_samples,num_train,replace = False))
    
    #difference are the indices used for the test
    indices_test = np.setdiff1d(indices_total, indices_train)    

    return indices_train,indices_test


def huber_loss(epsilon, y, Xw ):
    mod_a = abs(y-Xw)
    squared_case = mod_a < epsilon
    case1 = 0.5 * (mod_a ** 2)
    case2 = (epsilon * abs(mod_a)) - (0.5 * (epsilon ** 2))
    hans = (squared_case) * (case1) + (~squared_case) * (case2)
    return np.mean(hans)

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]
    
    #The y_label is MEDV from the boston.DESCR
    y_label = "MEDV"
    # i: index
    #fig, axs = plt.subplots(3,5, figsize=(15, 6), facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)        
        #plot all rows in X for the column (feature) i
        #against target y
        plt.plot(X[:,i],y, linestyle='None', marker='o', markerfacecolor='None', color="black")
        
        #labeling of Axes and title(s)
        plt.xlabel(features[i])
        plt.ylabel(y_label)
        title_str = y_label + " vs. " + features[i]
        plt.title(title_str)
    

    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    #Using w∗ = (XT X)^(−1)XT y
    X_t  = X.transpose()
    
    XTX = np.dot(X_t, X)
    XTY = np.dot(X_t, Y)
    w = np.linalg.solve(XTX, XTY)
    return w
    # Remember to use np.linalg.solve instead of inverting!
    #raise NotImplementedError()

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #adding Bias to X
    X = np.column_stack((np.ones(len(X)),X))

    #get randomized indices for 80% of the data
    indices_train, indices_test =  randomize_indices(len(y),0.80)

    #Make train, testing data from training and 
    #testing indices    
    X_train = np.take(X, indices_train, axis=0)
    y_train = np.take(y, indices_train)
    X_test = np.take(X, indices_test, axis=0)
    y_test =  np.take(y, indices_test)   

    # Fit regression model
    w = fit_regression(X_train, y_train)
    np.set_printoptions(suppress=True)
    print("weights:", w)
    
    # Compute fitted values, MSE, etc.
    
    #MSE Computation
    mse = np.mean((y_test-np.dot(X_test, w))**2)
    print ("MSE Loss:", mse)
    
    #L1 Loss Computation
    l1 = np.mean(abs(y_test-np.dot(X_test, w)))
    print ("L1 Absolute Loss:", l1)
    
    #huber loss     
    lhuber = huber_loss(1.35, y_test, np.dot(X_test, w))
    print("Huber Loss: ", lhuber)

if __name__ == "__main__":
    main()
