import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def mean_shift_data(data, mean):
    return (data.T + mean).T


def generate_data(mean, cov, num_instances):
    data = [np.random.multivariate_normal(mean, cov) for _ in range(num_instances)]
    return np.asarray(data).T


def get_random_positive_semidefinite_matrix():
     A= np.random.rand(2, 2)
     return np.dot(A,A.transpose())


def get_covariance_matrix(data):
    return (1/len(data))*np.matmul(data, data.transpose())


def get_eigenvector_matrix(data):
    eig_val, eig_vec = np.linalg.eig(data)
    return eig_val, eig_vec


def get_transformed_data(data, transformation):
    return np.matmul(transformation, data)


def pca_using_cov_matrix(data):
    data = mean_shift_data(data, -data.mean(axis=1))
    cov = get_covariance_matrix(data)
    eig_val, eig_vec_matrix = get_eigenvector_matrix(cov)
    new_data = get_transformed_data(data, eig_vec_matrix)
    return new_data, eig_val, eig_vec_matrix


def pca_using_svd(data):
    data = mean_shift_data(data, -data.mean(axis=1))
    eig_vecs, eig_vals, vh = np.linalg.svd(data)
    new_data = get_transformed_data(data, eig_vecs)
    return new_data, eig_vals, eig_vecs


def plot_data_after_pca(data, new_data, eig_vecs, title=""):
    """
    A part of the code is taken from https://stackoverflow.com/questions/18299523/basic-example-for-pca-with-matplotlib
    """
    x = data[0, :]
    y = data[1, :]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for temp_axis in eig_vecs:
        start, end = data.mean(axis=1), data.mean(axis=1) + new_data.std(axis=1).mean()*temp_axis
        ax.annotate('', end, start, arrowprops=dict(facecolor='red', width=2.0))
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    title = title + "\nCovariance Matrix is " + str(cov)
    plt.suptitle(title)
    plt.axis('equal');
    plt.show()


if __name__=='__main__':
    mean = [0, 0]
    cov = get_random_positive_semidefinite_matrix()
    data = generate_data(mean, cov, 1000)
    x = data[0, :]
    y = data[1, :]
    plt.scatter(x, y)
    plt.suptitle("Original Data")
    plt.show()
    new_data, eig_vals, eig_vecs = pca_using_svd(data)
    plot_data_after_pca(data, new_data, eig_vecs, "eigenvectors using Singular Value Decomposition")
    new_data, eig_vals, eig_vecs = pca_using_cov_matrix(data)
    plot_data_after_pca(data, new_data, eig_vecs, "eigenvectors using covariance matrix method")

