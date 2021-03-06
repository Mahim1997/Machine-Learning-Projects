# -*- coding: utf-8 -*-
"""ML-Offline-3-PCA-EM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_0pMk0csXTYC6NJ4Jryp0kVZqwpYmTuW

## 1505022 - ML Offline 3 on PCA and EM algorithm
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
FILE_NAME = '/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-3-PCA-EM/data.txt'

# data_np = np.array([
#     [1, 1, 1],
#     [1, 2, 1],
#     [1, 3, 2],
#     [1, 4, 3]
# ])
# print(data_np.shape)
# print(data_np)

data_np = np.loadtxt(FILE_NAME, dtype='float') # load data into 2D numpy array.

print(data_np.shape)
print(data_np[0].shape)

## https://stackoverflow.com/questions/31152967/normalise-2d-numpy-array-zero-mean-unit-variance
# X_standardized = data_np
X_standardized = StandardScaler().fit_transform(data_np)
print(f"X_standardized.shape = {X_standardized.shape}")

mean_X = np.mean(X_standardized, axis=0)
print(f"mean_X.shape = {mean_X.shape}")
# print(mean_X)

covariance_mat = (X_standardized - mean_X).T.dot((X_standardized - mean_X)) / (X_standardized.shape[0] - 1)
# cov2 = np.cov(X_standardized.T) ## SAME as above
print(f"covariance_mat.shape = {covariance_mat.shape}")
# print(covariance_mat)

del data_np

"""### Obtain eigen-decomposition"""

## https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
eig_values, eig_vectors = np.linalg.eig(covariance_mat)
print(eig_vectors.shape)

# print('Eigenvectors \n%s' %eig_vectors)
# print('\nEigenvalues \n%s' %eig_values)

########## Sort ############

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0], i[1])

# for i in range(0, 10):
#     if i >= len(eig_pairs):
#         break
#     print(eig_pairs[i][0])

explained_variances = []

for i in range(len(eig_pairs)):
    explained_variances.append(eig_pairs[i][0])
 
explained_variances = np.asarray(explained_variances)
explained_variances = explained_variances/np.sum(explained_variances)

# print("Total explained_variances.sum = ", np.sum(explained_variances))
# print("In % ", explained_variances*100)

print(f"len(eig_pairs) = {len(eig_pairs)}")

## Top 2 taken ##
num_components = 2

matrix_w = eig_pairs[0][1].reshape(-1, 1) ## start with first

for i in range(1, num_components):
    matrix_w = np.concatenate((matrix_w, eig_pairs[i][1].reshape(-1, 1)), axis=-1)

# matrix_w = np.hstack(
#     (
#     eig_pairs[0][1].reshape(-1,1), ## explicit num_vectors * 1 shape
#     eig_pairs[1][1].reshape(-1,1)
#     )
# )

print('Matrix W shape :\n', matrix_w.shape)
# print('Matrix W:\n', matrix_w)

convertedInput = np.dot(X_standardized, matrix_w)
# print(convertedInput.shape)
# print(convertedInput[0:10])

# print(convertedInput.T[0])

"""## Plot 2D visualization"""

import seaborn as sns

# projected_1 = X_scaled.dot(vectors.T[0])
# projected_2 = X_scaled.dot(vectors.T[1])

projected_1 = convertedInput.T[0]
projected_2 = convertedInput.T[1]

res = pd.DataFrame(projected_1, columns=['PC1'])
res['PC2'] = projected_2

# plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 8))
sns_plot = sns.scatterplot(res['PC1'], res['PC2'], [0] * len(res), s=50, palette='dark', legend=False) # hue=res['Y']

# sns_plot.figure.savefig("/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-3-PCA-EM/PCA-2D.png")

"""## EM algorithm"""

np.random.seed(22)

K = 3 ## no. of Gaussians

X = convertedInput
print(X.shape)

"""## Step 1. Initialize values and compute log likelihood"""

DEBUG = False

"""
Parameters:
    i)   weights
    ii)  means
    iii) covariances

Hidden variables:
    i) 
"""
class GaussianEM:
    def __init__(self, K, epsilon=0.001):
        self.K = K
        self.epsilon = epsilon
        print("GaussianEM __init__(K = {})".format(self.K))
        
    ### Initializes parameters and hidden variables
    def initialize_variables(self, X):
        print(f"X.shape = {X.shape}")
        self.X = X
        self.N, self.D = X.shape
        ## np.random.randn(self.D, )  np.zeros(self.D)
        self.means = np.array([np.zeros(self.D) for _ in range(self.K)]) ## list of K means, each means_k => (D, 1) or (D, ) size
        self.covariances = [np.random.randn(self.D, self.D) for _ in range(self.K)] ## list of K covariances, each cov_k => (D, D) size
        self.weights = np.array([(1/K) for _ in range(self.K)]) ## initialize as 1/K => use np array for easier normalization purposes

        print(f"self.means.shape = {self.means.shape}, self.covariances.len = {len(self.covariances)}, self.weights.shape = {self.weights.shape}")

        ## initially, compute once.
        self.compute_conditional_prob_ALL_Gauss()
    
    ### For each k_th Gaussian, return (N, 1) sized probability for ALL examples of data X i.e. N(X|mean_k, cov_k).
    def compute_conditional_prob_kth_Gauss(self, k):
        ## https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
        mean_k = self.means[k]
        cov_k = self.covariances[k]

        var_1 = X - mean_k ## (x_i - u_k) for all x_i
        var_2 = np.linalg.inv(cov_k) ## E_k inverse
        var_3 = (X - mean_k).T ## (x_i - u_k).T for all x_i

        product_left = np.dot(var_1, var_2)
        exponent = np.einsum('ij,ij->i', product_left, var_3.T) # np.sum(a*b_T, axis=1) # np.einsum('ij,ij->i', a, b_T)
        exp_vector = np.exp(
            -0.5 * exponent
        )
        denominator = np.sqrt(
            ((2*np.pi)**self.D) * np.abs(np.linalg.det(cov_k)) ## take absolute sign of determinant
        )
        return exp_vector/(denominator + self.epsilon)

    ## Computes conditional probabilities for ALL Gaussians
    def compute_conditional_prob_ALL_Gauss(self):
        self.conditional_prob_all_Gauss = np.asarray([self.compute_conditional_prob_kth_Gauss(k=k) for k in range(0, K)]) ## q
        self.conditional_prob_all_Gauss = self.conditional_prob_all_Gauss.T

        self.weighted_conditional_prob_all_Gauss = np.asarray([self.compute_conditional_prob_kth_Gauss(k=k)*self.weights[k] for k in range(0, K)]) ## q
        self.weighted_conditional_prob_all_Gauss = self.weighted_conditional_prob_all_Gauss.T

    ### Normalize each weights
    def normalize_weights(self):
        self.weights = self.weights/np.sum(self.weights)

    ### Expectation -> E-step, update hidden probabilities 'p'
    def step_expectation_E(self):
        self.compute_conditional_prob_ALL_Gauss()  ## compute q and wq matrices.
        self.hidden_probabilities = normalize(self.weighted_conditional_prob_all_Gauss, axis=1, norm='l1') ## l1 to make sum = 1, axis=1 for row wise ## axis=1: row
        if DEBUG:
            print(f"self.hidden_probabilities.shape = {self.hidden_probabilities.shape}")

    
    ### Maximization -> M-step, update each parameters with respect to hidden probabilities
    def step_maximization_M(self):
        ## Update means.
        mean_matrix = self.hidden_probabilities.T@self.X ## take transpose to get (K, N) * (N, D) -> (K, D)
        mean_matrix = mean_matrix / np.sum(self.hidden_probabilities.T, axis=1).reshape(-1, 1) ## row-wise sum and reshape to broadcast
        self.means = mean_matrix
        if DEBUG:
            print(f"self.means.shape = {self.means.shape}")

        # k = 0
        ## Update covariances for each k_th Gaussian
        for k in range(0, K):
            diff = self.X - self.means[k]
            
            diff = diff.reshape(diff.shape[0], 1, -1)
            b = diff.reshape(diff.shape[0], diff.shape[2], 1)

            c = b@diff
            
            probs = self.hidden_probabilities.T[k]
            probs = probs.reshape(probs.shape[0], 1, 1)
            
            p_x_product = probs*c
            numerator = np.sum(p_x_product, axis=0)
            denominator = np.sum(self.hidden_probabilities.T[k]).reshape(-1, 1)
            self.covariances[k] = numerator/(denominator + self.epsilon)

        ## Update weights and normalize.
        self.weights = np.sum(self.hidden_probabilities.T, axis=1) ## 1->row
        self.weights = self.weights/(self.hidden_probabilities.shape[0] + self.epsilon)


    ### Computes log likelihood using all Gaussian's params
    def compute_log_likelihood(self):
        if DEBUG:
            print(f"self.conditional_prob_all_Gauss.shape = {(self.conditional_prob_all_Gauss.shape)}")
            print(f"self.weighted_conditional_prob_all_Gauss.shape = {(self.weighted_conditional_prob_all_Gauss.shape)}")            

        row_wise_sum = np.sum(self.weighted_conditional_prob_all_Gauss, axis=1) ## axis=1 : row, axis=0 : column
        
        if DEBUG:
            print(f"row_wise_sum.shape = {row_wise_sum.shape}")

        log_likelihood = np.sum(np.log(row_wise_sum))
        return log_likelihood

#################################### Testing ####################################

def appendAndReturn_DataframeRes(df_old, metrics_list):
    a_series = pd.Series(metrics_list, index=df_old.columns)
    df_old = df_old.append(a_series, ignore_index=True)
    return df_old

column_names =["Itr", "PrevLoss", "CurrLoss", "DelLoss"]

df_metrics = pd.DataFrame(columns = column_names)
print(df_metrics.head(5))

"""## Fitting script"""

# Commented out IPython magic to ensure Python compatibility.

MAX_ITER = 10000
LIM_DELTA_LOSS = 0.01

em = GaussianEM(K=3) ## initialize with num of gaussians

em.initialize_variables(X=X) ## initialize data
em.normalize_weights() ## to make sum(w_k) = 1

itr = 0
log_likelihood = em.compute_log_likelihood() ## Compute iniital log likelihood
prev_log_likelihood = log_likelihood

while itr <= MAX_ITER:

    em.step_expectation_E() ## Expectation step
    em.step_maximization_M() ## Maximization step
    
    prev_log_likelihood = log_likelihood
    log_likelihood = em.compute_log_likelihood() ## Compute log likelihood
    
    change_loss = np.abs(np.abs(log_likelihood) - np.abs(prev_log_likelihood))
    # print(f"itr = {itr}, prev_log_likelihood = {prev_log_likelihood}, log_likelihood = {log_likelihood}, change_loss = {change_loss}")
    df_metrics = appendAndReturn_DataframeRes(df_old=df_metrics, metrics_list=[itr, prev_log_likelihood, log_likelihood, change_loss])

    if itr > 0 and change_loss < LIM_DELTA_LOSS:
        break

    itr += 1

display(df_metrics.head(5))

print(em.hidden_probabilities.shape, X.shape)
# gaussian_indices = []
# for i in range(0, len(X)):
#     max_idx = np.argmax(em.hidden_probabilities[i])
#     gaussian_indices.append(max_idx)

gaussian_indices = np.array([np.argmax(em.hidden_probabilities[i]) for i in range(0, len(X))])

print(np.unique(gaussian_indices, return_counts=True))

res['GaussianIndex'] = gaussian_indices

display(res.head(5))

plt.figure(figsize=(10, 8))
sns_plot_2 = sns.scatterplot(x=res['PC1'], y=res['PC2'], hue=res['GaussianIndex'], size=[0]*len(res), s=50, palette='dark', legend=False)

# sns_plot_2.figure.savefig("/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-3-PCA-EM/PCA-Clustered.png")

"""## Diagrams and matrices saving for report"""

def appendAndReturn_DataframeRes(df_old, metrics_list):
    a_series = pd.Series(metrics_list, index=df_old.columns)
    df_old = df_old.append(a_series, ignore_index=True)
    return df_old

column_names =["k", "Mean_k", "Covar_k", "MixCoeff_k"]

df_gaussians = pd.DataFrame(columns = column_names)
print(df_gaussians.head(5))

em.covariances[0].ravel()

df_gaussians['k'] = np.arange(len(em.means)) + 1 ## 1, 2, 3, ..., K
df_gaussians['Mean_k'] = em.means.tolist()
df_gaussians['Covar_k'] = em.covariances
df_gaussians['MixCoeff_k'] = em.weights

display(df_gaussians)

# df_gaussians.to_csv("/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-3-PCA-EM/DF-Gaussians.csv", index=False)