import numpy as np
from sklearn.decomposition import PCA

def mean_vector(data_matrix):
    """
      Compute the mean of data matrix using libraries
    """
    # Compute the mean along the columns
    mean_vector = np.mean(data_matrix, axis=0)

    # Convert the mean vector to a column vector
    mean_column_vector = np.reshape(mean_vector, (mean_vector.shape[0], 1))

    return mean_column_vector

def custom_mean_vector(data_matrix):
    """
      Compute the mean of data matrix without libraries
    """
    num_rows, num_cols = len(data_matrix), len(data_matrix[0])
    mean_X = [0] * num_cols

    for j in range(num_cols):
        col_sum = 0
        for i in range(num_rows):
            col_sum += data_matrix[i][j]
        mean_X[j] = col_sum / num_rows

    return mean_X

def center_matrix(data_matrix):
    """
      Center a matrix using libraries
    """

    # Compute the mean of matrix X
    mean_X = np.mean(data_matrix, axis=0)

    # Center matrix X
    return data_matrix - mean_X

def custom_center_matrix(data_matrix):
    """
      Center a Matrix without libraries
    """
    num_rows, num_cols = len(data_matrix), len(data_matrix[0])
    mean_X = [0] * num_cols

    for j in range(num_cols):
        col_sum = 0
        for i in range(num_rows):
            col_sum += data_matrix[i][j]
        mean_X[j] = col_sum / num_rows

    # Center matrix X
    return [[data_matrix[i][j] - mean_X[j] for j in range(num_cols)] for i in range(num_rows)]

def compute_unnormalized_covariance(data_matrix):
    """
      Compute Unnormalized Covariance
    """
    return np.dot(data_matrix.T, data_matrix)

def custom_compute_unnormalized_covariance(data_matrix):
    """
      Custom Compute Unnormalized Covariance
    """
    num_rows, num_cols = len(data_matrix), len(data_matrix[0])

    # Compute the covariance matrix of X1
    C = [[0] * num_cols for _ in range(num_cols)]

    for i in range(num_cols):
        for j in range(num_cols):
            col_i, col_j = [data_matrix[k][i] for k in range(num_rows)], [data_matrix[k][j] for k in range(num_rows)]
            cov_ij = sum([col_i[k] * col_j[k] for k in range(num_rows)]) / (num_rows - 1)
            C[i][j] = 2 * cov_ij

    return C

def compute_first_principal_component(C_value):
    """
      Compute The First principal Component
    """
    # Compute the eigenvalues and eigenvectors of C
    eigenvalues, eigenvectors = np.linalg.eig(C_value)

    # Find the index of the largest eigenvalue
    max_eigenvalue_idx = np.argmax(eigenvalues)

    # Extract the corresponding eigenvector
    return eigenvectors[:, max_eigenvalue_idx]

def custom_compute_first_principal_component(C_value):
    """
      Custom Compute The First principal Component
    """
    # Compute the eigenvectors and eigenvalues of C
    eigenvalues, eigenvectors = np.linalg.eig(C_value)

    # Find the index of the maximum eigenvalue
    max_eigenvalue_idx = np.argmax(eigenvalues)

    # Extract the corresponding eigenvector (i.e., the first principal component)
    first_principal_component = eigenvectors[:, max_eigenvalue_idx]

    # Compute the corresponding principal value
    first_principal_value = eigenvalues[max_eigenvalue_idx]

    # Print the first principal component and its principal value
    return [first_principal_component, first_principal_value]

def D1_representation(data_matrix):
    mean = np.mean(data_matrix, axis=1)

    # subtract mean from data matrix to center the data
    X_centered = data_matrix - mean.reshape(-1, 1)

    # compute covariance matrix of centered data
    cov = np.cov(X_centered)

    # compute eigenvector corresponding to largest eigenvalue
    eigvals, eigvecs = np.linalg.eig(cov)
    max_eigval_idx = np.argmax(eigvals)
    principal_component = eigvecs[:, max_eigval_idx]

    # project centered data matrix onto principal component to obtain 1D representation
    return np.dot(principal_component.T, X_centered) + mean[max_eigval_idx]

    # Project the data onto the first principal component
    X_projected = np.dot(first_principal_component.reshape(-1,1).T, X1.T)

    # Add back the data mean to get the best 1D representation of X
    X_1D = X_mean + np.outer(first_principal_component, X_projected)

    return X_1D

def custom_D1_representation(X):
    # compute mean vector
    mean = [0]*X.shape[0]
    for i in range(X.shape[0]):
        mean[i] = sum(X[i])/X.shape[1]

    # subtract mean from data matrix to center the data
    X_centered = [[0]*X.shape[1] for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_centered[i][j] = X[i][j] - mean[i]

    # compute covariance matrix of centered data
    cov = [[0]*X.shape[0] for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                cov[i][j] += X_centered[i][k]*X_centered[j][k]
            cov[i][j] /= X.shape[1]

    # compute eigenvector corresponding to largest eigenvalue
    eigvals = [0]*X.shape[0]
    eigvecs = [[0]*X.shape[0] for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i == j:
                eigvecs[i][j] = 1
            else:
                eigvecs[i][j] = 0

    for iter in range(100):
        max_eigval_idx = 0
        max_eigval = eigvals[0]
        for i in range(1, X.shape[0]):
            if eigvals[i] > max_eigval:
                max_eigval_idx = i
                max_eigval = eigvals[i]
        v = [0]*X.shape[0]
        for i in range(X.shape[0]):
            v[i] = eigvecs[i][max_eigval_idx]
        w = [0]*X.shape[0]
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                w[i] += cov[i][j] * v[j]
        eigvals[max_eigval_idx] = sum([w[i] * v[i] for i in range(X.shape[0])])
        for i in range(X.shape[0]):
            eigvecs[i][max_eigval_idx] = w[i] / eigvals[max_eigval_idx]

    # project centered data matrix onto principal component to obtain 1D representation
    best_1d_representation = [0]*X.shape[1]
    for i in range(X.shape[1]):
        best_1d_representation[i] = sum([X_centered[j][i] * eigvecs[j][0] for j in range(X.shape[0])]) + mean[0]

    return best_1d_representation

def mypca(X, k):
    """
    Perform PCA on the input data matrix X and return the optimal k-dimensional representation.

    Args:
        X (ndarray): Input data matrix with shape (n_features, n_samples)
        k (int): Number of dimensions to keep for the output representation

    Returns:
        rep (ndarray): Optimal k-dimensional representation of X with shape (k, n_samples)
        pc (ndarray): Top k principal components with shape (n_features, k)
        pv (ndarray): Top k principal values with shape (k,)
    """

    # Step 1: Center the data by subtracting the mean of each feature
    mean_vec = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_vec

    # Step 2: Compute the covariance matrix of the centered data
    cov_mat = np.cov(X_centered, rowvar=True)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Step 5: Select the top k eigenvectors and eigenvalues
    pc = eig_vecs[:, :k]
    pv = eig_vals[:k]

    # Step 6: Compute the k-dimensional representation of the centered data
    rep_centered = np.dot(pc.T, X_centered)

    return rep_centered, pc, pv

def mypca2(X, k):
    # Subtract the mean from the data matrix
    X_centered = X - np.mean(X, axis=1, keepdims=True)

    # Compute the covariance matrix
    cov = np.cov(X_centered)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov)

    # Sort the eigenvectors based on their eigenvalues in decreasing order
    sorted_indices = np.argsort(eig_vals)[::-1]
    sorted_eig_vals = eig_vals[sorted_indices]
    sorted_eig_vecs = eig_vecs[:, sorted_indices]

    # Extract the top k eigenvectors and eigenvalues
    top_k_eig_vals = sorted_eig_vals[:k]
    top_k_eig_vecs = sorted_eig_vecs[:, :k]

    # Compute the principal components
    pc = np.dot(X_centered.T, top_k_eig_vecs)

    # Compute the optimal k-dimensional representation of the data
    rep = np.dot(top_k_eig_vecs, pc.T) #+ np.mean(X, axis=1, keepdims=True)

    # Return the results
    return rep, pc, top_k_eig_vals

def pca_comparison(data_matrix):
    rep, pc, pv = mypca(data_matrix, 3)

    # Compute PCA using scipy library
    pca = PCA(n_components=3)
    pca.fit(data_matrix.T)
    rep_scipy = pca.transform(data_matrix.T).T
    pc_scipy = pca.components_.T
    pv_scipy = pca.explained_variance_

    # Verify that the results are identical
    print('Rep: {}'.format(np.allclose(rep, rep_scipy)))
    print('PC: {}'.format(np.allclose(pc, pc_scipy)))
    print('PV: {}'.format(np.allclose(pv, pv_scipy)))


def main():
    X = np.array([
        [-2, 1, 4, 6, 5, 3, 6, 2],
        [9, 3, 2, -1, -4, -2, -4, 5],
        [0, 7, 5, 3, 2, -3, 4, 6]
    ])

    print('principal Component Analysis in Python')

    print('----------------------------------------')
    print('Given the following matrix: ')
    print(X)
    print('\n')

    print('1. Deriving Mean Column Vector of X')
    X_mean = mean_vector(X)
    print(X_mean)
    print('\n')

    print('2. Custom Function for Deriving Mean Column Vector of X')
    custom_X_mean = custom_mean_vector(X)
    print(custom_X_mean)
    print('\n')

    print('3. Center data matrix X as X1')
    X1 = center_matrix(X)
    print(X1)
    print('\n')

    print('4. Custom Function for Center data matrix X as X1')
    custom_X1 = center_matrix(X)
    print(custom_X1)
    print('\n')

    print('5. Compute Unnormalized Covariance Matrix of the Centered Data Matrix X1')
    C = compute_unnormalized_covariance(X1)
    print(C)
    print('\n')

    print('6. Custom Compute Unnormalized Covariance Matrix of the Centered Data Matrix X1')
    custom_C = custom_compute_unnormalized_covariance(custom_X1)
    print(custom_C)
    print('\n')

    print('7. Compute the First principal Component')
    first_principal_component = compute_first_principal_component(C)
    print(first_principal_component)
    print('\n')

    print('8. Custom Compute the First principal Component')
    custom_first_principal_component = custom_compute_first_principal_component(C)
    print(custom_first_principal_component)
    print('\n')

    print('9. 1D Representation of Centered X')
    D1_representation_X = D1_representation(X)
    print(D1_representation_X)
    print('\n')

    print('10. Custom 1D Representation of Centered X')
    custom_D1_representation_X = custom_D1_representation(X)
    print(custom_D1_representation_X)
    print('\n')

    print('11. My PCA Test')
    res = mypca(X,4)
    print(res)
    print('\n')
    
    print('12. PCA Comparison')
    pca_comparison(X)


if __name__ == '__main__':
    main()