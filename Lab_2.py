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
    mean = np.mean(matrix)
    std = np.std(matrix)
    std_matrix = (matrix - mean) / std
    cov_matrix = np.cov(std_matrix)
    eigenvalues, _ = eigen(cov_matrix)
    index = np.argsort(eigenvalues)[::-1]
    return eigenvalues[index]


def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(
        np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors)
    )
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(
        np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors)
    )
    decrypted_vector = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector)
    decrypted_message = "".join([chr(int(np.round(ch))) for ch in decrypted_vector])
    return decrypted_message


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
cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
print("Cumulative variance: ", cumulative)
cumulative = cumulative * 100

plt.plot(cumulative)
plt.plot([-15, 750], [95, 95], "r--")
plt.plot([140, 139], [-15, 105], "g--")
plt.xlim(-15, 750)
plt.ylim(-15, 105)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance by components")

components = np.argmax(cumulative > 95) + 1
print("Components(95%): ", components)

compress_image(image_bw, components)

compress_image(image_bw, 15)
compress_image(image_bw, 35)
compress_image(image_bw, 75)
compress_image(image_bw, 150)
compress_image(image_bw, 200)
compress_image(image_bw, 250)

message = "Hello, World!"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted_message = encrypt_message(message, key_matrix)
print("Encrypted message:", encrypted_message)
decrypted_message = decrypt_message(encrypted_message, key_matrix)
print("Decrypted message:", decrypted_message)