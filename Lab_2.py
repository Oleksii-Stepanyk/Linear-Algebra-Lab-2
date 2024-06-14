import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from sklearn.decomposition import PCA

matrix = np.array([[-1, 0, 0], [1, 4, 4], [-1, -2, -2]])


def compress_image(image, components):
    pca = PCA(n_components=components)
    image_compressed = pca.fit_transform(image)
    new_image = pca.inverse_transform(image_compressed)
    plt.imshow(new_image, cmap="gray")
    plt.show()


def eigen(matrix):
    (eigenvalues, eigenvectors) = np.linalg.eig(matrix)
    for index in range(len(eigenvectors)):
        matrix_dot = np.dot(matrix, eigenvectors[:, index])
        eigen_dot = np.dot(eigenvalues[index], eigenvectors[:, index])
        if not np.allclose(matrix_dot, eigen_dot):
            print("Error")
            return
    return eigenvalues, eigenvectors


def get_sorted_eigenvalues(matrix):
    std_matrix = standartization(image_bw)
    cov_matrix = covariance(std_matrix)
    eigenvalues, eigenvectors = eigen(cov_matrix)
    return sort_eigenvalues(eigenvalues)


def standartization(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    return (matrix - mean) / std


def covariance(matrix):
    return np.cov(matrix)


def sort_eigenvalues(eigenvalues):
    index = np.argsort(eigenvalues)[::-1]
    return eigenvalues[index]


def cumulative_variance(eigenvalues):
    return np.cumsum(eigenvalues) / np.sum(eigenvalues)


eigenvalues, eigenvectors = eigen(matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

image_raw = imread("image.jpg")
print(image_raw.shape)
plt.imshow(image_raw)
plt.show()

image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum / image_sum.max()
print(image_bw.max())
plt.imshow(image_bw, cmap="gray")
plt.show()

eigenvalues = get_sorted_eigenvalues(image_bw)
cumulative = cumulative_variance(eigenvalues)
print("Cumulative variance: ", cumulative)

components = np.argmax(cumulative > 0.95) + 1
print("Components(95%): ", components)

compress_image(image_bw, components)

compress_image(image_bw, 15)
compress_image(image_bw, 35)
compress_image(image_bw, 75)
compress_image(image_bw, 150)
compress_image(image_bw, 200)
compress_image(image_bw, 250)