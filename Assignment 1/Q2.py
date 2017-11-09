# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import scipy as sp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#print the end graph
def print_tau_plot(losses):    
    plt.plot(np.asarray(losses).mean(0), 'k-' )
    plt.grid(True, which="both", ls="-")
    plt.xlabel("Tau")
    plt.ylabel("Loss")
    title_str = "Tau vs. Loss"
    plt.title(title_str)
    plt.show()
    #print("min loss = {}".format(losses.min()))
    
    return

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train,tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    #Solving Based on the equation provided
    #Where X is X_train and X_i is test datum
    loss = l2(x_train,np.transpose(test_datum))
    common = (-1 * loss / (2 *(tau ** 2)))
    top = np.exp(common)
    bottom = np.exp(sp.misc.logsumexp(common))
    
    #Dividing the top and bottom of the eqn to get a_i    
    a_i = np.divide(top, bottom)
    #Diagonalizing for A 
    A = np.diag(a_i[:,0])
    
    #Using A with the proven equation to get w*
    x_t = np.transpose(x_train)
    x_t_A = np.dot(x_t,A)
    x_t_A_x = np.dot(x_t_A, x_train)
    
    I = np.identity(len(x_train[1]))
    lambdaI = np.dot(I,lam)
    
    part1= x_t_A_x + lambdaI
    
    part2 = np.dot(x_t_A,y_train)
    part2 = np.expand_dims(part2, axis=1)
    
    #Similar to part1, using linalg solve to perform the solve
    w_star = np.linalg.solve(part1, part2)
    y_hat = np.dot(np.transpose(w_star), test_datum)
    
    return y_hat




def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    
    #The splitting is done by indices
    #We randomize the indices, then pull n/k each time
    #at the end of the kth loop, we should have k folds
    #with random indices
    
    #total samples
    num_samples = len(y)
    
    #number to take per fold
    pull_out = round(len(y)/k)
    
    #Make an array to pull out from
    indices_total = np.arange(num_samples)    

    #Make an array to stored out the values you pulled out
    k_fold_indices = []
    
    for i in range (k):
        #Shuffle - It's ineffective so do it inside the for loop
        np.random.shuffle(indices_total)
        
        #pull out and append to the first indiex of the storage array
        k_fold_indices.append(indices_total[0:pull_out])
        
        #remove the pulled out indices
        indices_total = np.setdiff1d(indices_total, k_fold_indices[i]) 
        
    #Once the indices have been pulled out, need to store the losses
    losses = []
    #For K indices, there are K number of test to perform
    for i in range (k):
        #Get the total number of folds in an array
        total_folds = np.arange(k)

        #Pull out the index as the test fold
        test_fold = np.array([i])
        
        #Let the remaining be the training folds (K-1)
        training_folds = np.setdiff1d(total_folds, test_fold)

        #Accumulate the values from each fold index from the randomized indices
        #Using the combined array of randomized indices, take the values to make
        #the concatenated test and training sets        
        train_set = np.take(k_fold_indices,training_folds , axis=0)
        train_set = np.concatenate(train_set).ravel()
        
        test_set = np.take(k_fold_indices,test_fold, axis=0)
        test_set = np.concatenate(test_set).ravel()

        #Use that array to pull out the values common across X, Y        
        X_train = np.take(x,train_set, axis=0)
        Y_train = np.take(y,train_set, axis=0)
        
        X_test = np.take(x,test_set, axis=0)
        Y_test = np.take(y,test_set, axis=0) 
        
        #Use the given function to pull out loss
        losses.append(run_on_fold(X_test, Y_test, X_train, Y_train, taus))

    return losses

    ## TODO
    #X_train = np.take(X, indices_train, axis=0)
    ##run_on_fold(x_test, y_test, x_train, y_train, taus):


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,10)
    losses = run_k_fold(x,y,taus,k=5)    
    print_tau_plot(losses)
