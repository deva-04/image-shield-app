import numpy as np

def reversible_xor_diffusion(image):
    flat_image = image.flatten()
    diffused = np.copy(flat_image)

    for j in range(1, len(flat_image)):
        diffused[j] ^= diffused[j - 1]  

    return diffused.reshape(image.shape)

def reversible_xor_inverse_diffusion(image):
    flat_image = image.flatten()
    recovered = np.copy(flat_image)

    for j in range(len(flat_image) - 1, 0, -1):  
        recovered[j] ^= recovered[j - 1]  

    return recovered.reshape(image.shape)

def apply_permutation(image, chaotic_sequence, key=0.5):
    return image.ravel()[np.argsort(chaotic_sequence)].reshape(image.shape)

def inverse_permutation(image, chaotic_sequence):
    sorted_indices = np.argsort(chaotic_sequence)
    return image.ravel()[np.argsort(sorted_indices)].reshape(image.shape)
