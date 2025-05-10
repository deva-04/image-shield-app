import numpy as np

def logistic_map(x, mu=3.99):
    x = np.clip(x, 0.001, 0.999) 
    return mu * x * (1 - x)

def compute_initial_x0(image):
    return (np.sum(image) % 10) / 10

def generate_chaotic_sequence(image, size, mu=3.99, x0=0.5):
    x0 = (compute_initial_x0(image) + x0) % 1
    sequence = np.empty(size)
    sequence[0] = np.clip(x0, 0.001, 0.999)

    for i in range(1, size):
        sequence[i] = logistic_map(sequence[i - 1], mu)

    return np.clip(sequence, 0.001, 0.999)

def spatiotemporal_chaos(image, image_size, epsilon=0.3, p=0.5):
    chaos_matrix = np.zeros((image_size, image_size))
    chaos_matrix[:, 0] = generate_chaotic_sequence(image, image_size, x0=0.5)

    for i in range(1, image_size):
        chaos_matrix[:, i] = (1 - epsilon) * logistic_map(chaos_matrix[:, i - 1]) + \
                             (epsilon / 2) * (logistic_map(chaos_matrix[:, max(i-1,0)]) + 
                                              logistic_map(chaos_matrix[:, min(i+1, image_size-1)]))
        chaos_matrix[:, i] = np.clip(chaos_matrix[:, i], 0.001, 0.999)

    return chaos_matrix

def binary_chaotic_mask(chaos_matrix):
    chaos_matrix = np.nan_to_num(chaos_matrix, nan=0.5)
    return ((chaos_matrix * 1e17) % 2).astype(int)
