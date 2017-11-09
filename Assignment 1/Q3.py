import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50
K = 500
chosen_weight = 5

#Plot the graph
def plot_graph_of_sigmas(batch_sampler,w):
    title_str = "Gradient Range vs. Variance for weight #" + str(chosen_weight)
    print(title_str)
    var = KVar(batch_sampler, w)
    plt.semilogx(range(1,400), var)
    plt.grid(True, which="both", ls="-")
    plt.xlabel("m")
    plt.ylabel("Variance")
    plt.title(title_str)
    plt.show()
    return

def squareDistanceMetric(vec1, vec2):
    diffvec = np.subtract(vec1, vec2)
    squarediff = np.square(diffvec)
    #print(diffvec)
    sum = np.sum(squarediff)
    rootsum = (sum ** 0.5)
    #print(sum, rootsum)
    return rootsum

def KVar(batch_sampler, w):
    
    var = []
    for m in range(1,400):
        saved_values = []    
        for i in range (K):
            X_b, y_b = batch_sampler.get_batch(m)
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            saved_values.append(batch_grad[chosen_weight])
        gradmean = np.mean(saved_values, axis=0)
        var.append(sum([(p - gradmean)**2 for p in saved_values]) / len(saved_values))

    print (var)
    return var



def K_Gradient_Average(batch_sampler, w):
    
    saved_values = []    
    for i in range (K):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        saved_values.append(batch_grad)
    
    gradmean = np.mean(saved_values, axis=0)
    return gradmean
    
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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    X_t  = np.transpose(X)
    n = X.shape[0]
    XTX = np.dot(X_t, X)
    XTXw = np.dot(XTX, w) 
    XTY = np.dot(X_t, y)
    gradient = np.subtract(XTXw, XTY)
    gradient = (2/n) * gradient
    #gradient = np.expand_dims(gradient, axis=1)
    
    return gradient
    #raise NotImplementedError()

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    
    #Perform mean on the batch using sample and weights
    meangrad = K_Gradient_Average(batch_sampler, w) 
    
    #Get true gradient by performing grad on whole set
    truegrad = lin_reg_gradient(X, y, w)
    
    #Similarity Checks
    cossim = cosine_similarity(meangrad,truegrad)
    print("Cosine Similarity:", cossim)
    
    sqd = squareDistanceMetric(meangrad, truegrad)
    print("Square Distance Metric:", sqd)
    
    plot_graph_of_sigmas(batch_sampler,w)

    #Given Code Commented
    # Example usage
    #X_b, y_b = batch_sampler.get_batch()
    #batch_grad = lin_reg_gradient(X_b, y_b, w)




if __name__ == '__main__':
    main()