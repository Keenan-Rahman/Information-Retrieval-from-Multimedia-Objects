# pca.py
# CSE 515
# Project Phase II
# This program applies the PCA dimensionality reduction technique to retrieve the latent semantics of a dataset ranked by data preservation

import numpy as np
import scipy
import scipy.linalg

def pca(k, dataset): #PCA method. Input should be provided as integer k latent features to select and an n x p numpy array where n is the number of data vectors and p is the numer of elements per data vector
    
    #Calculate mean of each column
    means = np.mean(dataset, axis = 0)
    
    #Calculate deviations from mean
    deviationsFromMean = dataset - means
    
    #Calculate covariance matrix of deviations from mean deviated matrix
    covarianceMatrix = ((np.dot(deviationsFromMean.T, deviationsFromMean)) / (deviationsFromMean.shape[0] - 1)) #covariance is taken by taking the dot product of the transpose of the means deviated matrix with the regular matrix and then dividing it by n - 1
    
    #Find eigendecomposition of covariance matrix
    eigendecomposition = scipy.linalg.eigh(np.array(covarianceMatrix, dtype=float))
    # np.array(covarianceMatrix, dtype=float)
    eigenvalues = eigendecomposition[0]
    eigenvectors = eigendecomposition[1]

    #sort eigenvalues in descending order by extracting the indices of the descending order eigenvalues and then sorting the eigenvalue and eigenvevtor lists on those indices
    sortIndices = eigenvalues.argsort()[::-1] 
    sortedEVals = eigenvalues[sortIndices]
    sortedEVectors = eigenvectors[:,sortIndices]

    #select the k highest value eigenvectors
    finalEVectors = sortedEVectors[:, 0:k]

    #apply dimensional redution transformation to dataset (dataset represented as the means deviated matrix)
    projection = np.dot(deviationsFromMean, finalEVectors) 
    
    return [projection, finalEVectors.T, sortedEVals, sortedEVectors] #return a list containing the PCA-applied transformation of the input matrix as well as the selected bases vecotrs and the full set of sorted eigenvalues and eigenvectors.

#pca(5)