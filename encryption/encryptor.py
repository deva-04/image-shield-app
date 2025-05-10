import numpy as np
import time
import cv2

from encryption.diffusion_and_permutation import reversible_xor_diffusion, reversible_xor_inverse_diffusion, apply_permutation, inverse_permutation
from encryption.chaotic_maps import generate_chaotic_sequence, spatiotemporal_chaos, binary_chaotic_mask
from encryption.rca import apply_rca 

def resize_image(image):
    if len(image.shape) == 3:  # If the image is RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    if (height, width) == (256, 256):
        print(f"Image is already 256x256. Proceeding with encryption...")
        processed_image = image  # No resizing needed
    elif (height, width) == (512, 512):
        print(f"Image is already 512x512. Proceeding with encryption...")
        processed_image = image
    else:
        print(f"Image is {height}x{width}, resizing to 512x512 for consistency.")
        processed_image = cv2.resize(image, (512, 512))
    print(f"Processing image of size: {processed_image.shape}")
    return processed_image

def encrypt_image(image, mu=3.99, iterations=1, key=0.5):
    start_time = time.perf_counter() 
    image = resize_image(image)
    height, width = image.shape
    diffused_image = reversible_xor_diffusion(image)
    chaotic_sequence = generate_chaotic_sequence(image, image.size, mu=mu, x0=key)
    permuted_image = apply_permutation(diffused_image, chaotic_sequence)
    chaos_matrix = spatiotemporal_chaos(image, height, p=key)
    rca_mask = binary_chaotic_mask(chaos_matrix).reshape(image.shape)  
    encrypted_image = apply_rca(permuted_image, rca_mask, iterations=iterations)
    encryption_time = time.perf_counter() - start_time
    return encrypted_image, chaotic_sequence, rca_mask, encryption_time

def decrypt_image(encrypted_image, chaotic_sequence, rca_mask, iterations=1, key=0.5):
    start_time = time.perf_counter() 
    decrypted_rca = apply_rca(encrypted_image, rca_mask, iterations=iterations, reverse=True)
    inverse_permuted = inverse_permutation(decrypted_rca, chaotic_sequence)
    recovered_image = reversible_xor_inverse_diffusion(inverse_permuted)
    decryption_time = time.perf_counter() - start_time 
    return recovered_image, decryption_time
