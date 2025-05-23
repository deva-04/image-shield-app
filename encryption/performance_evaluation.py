import numpy as np 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from skimage.metrics import structural_similarity as ssim
import random
import cv2
from .encryptor import encrypt_image, decrypt_image
import gc

def correlation_coefficient(image1, image2):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return np.corrcoef(image1.flatten(), image2.flatten())[0, 1]

def npcr_uaci(image1, image2):
    M, N = image1.shape
    D = (image1 != image2).astype(int)
    NPCR = 100 * np.sum(D) / (M * N)
    UACI = 100 * np.sum(np.abs(image1 - image2)) / (255 * M * N)
    return NPCR, UACI

def plot_correlation(image1, image2, save_path, title="Image Correlation", label1="Image 1", label2="Image 2"): 
    try:
        r = correlation_coefficient(image1, image2)

        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        plt.figure(figsize=(5, 5))
        plt.scatter(image1.flatten(), image2.flatten(), s=1, alpha=0.5)
        plt.xlabel(f"Pixel values of {label1}")
        plt.ylabel(f"Pixel values of {label2}")
        plt.title(f"{title}\nCorrelation Coefficient: {r:.4f}")
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=72)
        plt.close('all')
        gc.collect()

    except Exception as e:
        print(f"[ERROR] plot_correlation failed: {e}")

def compute_histogram(image):
    hist = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    return hist

def plot_histograms(original_image, encrypted_image, save_path):
    try:
        plt.figure(figsize=(10, 4))

        # Histogram for original image
        plt.subplot(1, 2, 1)
        plt.hist(original_image.ravel(), bins=256, color='blue', alpha=0.7)
        plt.title("Original Image Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # Histogram for encrypted image
        plt.subplot(1, 2, 2)
        plt.hist(encrypted_image.ravel(), bins=256, color='red', alpha=0.7)
        plt.title("Encrypted Image Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=80, bbox_inches='tight')  # Reduce DPI to save memory
        plt.close()  # Free memory after saving plot

    except Exception as e:
        print(f"[ERROR] plot_histograms failed: {e}")

def entropy(image):
    hist = compute_histogram(image)
    hist = hist[hist > 0] 
    return -np.sum(hist * np.log2(hist))

def chi_square_test(original_img, encrypted_img):
    original_hist = cv2.calcHist([original_img], [0], None, [256], [0, 256]).flatten()
    encrypted_hist = cv2.calcHist([encrypted_img], [0], None, [256], [0, 256]).flatten()
    chi2_stat, p_value, _, _ = chi2_contingency([original_hist, encrypted_hist])
    return chi2_stat, p_value

def key_sensitivity_test(image, mu=3.99, iterations=1, key=0.5, delta=1e-5):
    perturbed_key = np.sin((key - delta) * np.pi / 2) ** 2
    perturbed_key = np.clip(perturbed_key, 0, 1)
    encrypted_image_1, _, _, _ = encrypt_image(image, mu, iterations, key)
    encrypted_image_2, _, _, _ = encrypt_image(image, mu, iterations, perturbed_key)
    difference = np.sum(encrypted_image_1 != encrypted_image_2)
    return difference

def avalanche_effect(original_image, encrypted_image, key=0.5, mu=3.99, iterations=1, trials=3):
    avg_npcr, avg_uaci = 0, 0
    for _ in range(trials):
        flipped_image = original_image.copy()
        x, y = random.randint(0, original_image.shape[0] - 1), random.randint(0, original_image.shape[1] - 1)
        bit_to_flip = random.randint(0, 7)
        flipped_image[x, y] ^= (1 << bit_to_flip)
        flipped_encrypted_image, _, _, _ = encrypt_image(flipped_image, mu, iterations, key)
        npcr, uaci = npcr_uaci(encrypted_image, flipped_encrypted_image)
        avg_npcr += npcr
        avg_uaci += uaci
    return avg_npcr / trials, avg_uaci / trials

def psnr(original_image, decrypted_image):
    original_image = original_image.astype(np.float32)
    decrypted_image = decrypted_image.astype(np.float32)
    return cv2.PSNR(original_image, decrypted_image)

def ssim_index(original_image, decrypted_image):
    return ssim(original_image, decrypted_image, data_range=decrypted_image.max() - decrypted_image.min())
