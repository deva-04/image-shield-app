from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask_cors import CORS
import matplotlib.pyplot as plt

from encryption.encryptor import encrypt_image, decrypt_image, resize_image
from encryption.performance_evaluation import (
    correlation_coefficient, npcr_uaci, plot_correlation, plot_histograms,
    entropy, chi_square_test, key_sensitivity_test, avalanche_effect, ssim_index
)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get form parameters
    mu = float(request.form['mu'])
    iterations = int(request.form['iterations'])
    key = float(request.form['key'])
    delta = float(request.form['delta'])

    # Timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{timestamp}_{file.filename.rsplit('.', 1)[0]}"

    # Save and load original image
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_original.png")
    file.save(original_path)
    image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    image = resize_image(image)  # Ensure all inputs are 512x512

    # --- Encryption ---
    encrypted_image, chaotic_sequence, rca_mask, encryption_time = encrypt_image(
        image, mu=mu, iterations=iterations, key=key
    )
    encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_encrypted.png")
    cv2.imwrite(encrypted_path, encrypted_image)

    # --- Decryption ---
    decrypted_image, decryption_time = decrypt_image(
        encrypted_image, chaotic_sequence, rca_mask, iterations=iterations, key=key
    )
    decrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_decrypted.png")
    cv2.imwrite(decrypted_path, decrypted_image)

    # --- Metric Calculations ---
    correlation_enc = correlation_coefficient(image, encrypted_image)
    correlation_dec = correlation_coefficient(image, decrypted_image)
    npcr, uaci = npcr_uaci(image, encrypted_image)
    entropy_enc = entropy(encrypted_image)
    chi2, p_val = chi_square_test(image, encrypted_image)
    ssim_val = ssim_index(image, decrypted_image)

    mse = np.mean((image - decrypted_image) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse != 0 else float('inf')

    key_diff = key_sensitivity_test(image, mu, iterations, key, delta)
    avg_npcr, avg_uaci = avalanche_effect(image, encrypted_image, key=key, mu=mu, iterations=iterations)

    # --- Save Excel Report ---
    results_dict = {
        "Image Name": file.filename,
        "Mu Value (Chaos Function)": mu,
        "Iterations (Cellular Automata)": iterations,
        "Encryption Key": key,
        "Delta (Key Change)": delta,
        "Encryption Time (s)": f"{encryption_time:.6f}",
        "Decryption Time (s)": f"{decryption_time:.6f}",
        "Correlation Coefficient (Original - Encrypted)": f"{correlation_enc:.4f}",
        "Correlation Coefficient (Original - Decrypted)": f"{correlation_dec:.4f}",
        "NPCR (%)": f"{npcr:.4f}",
        "UACI (%)": f"{uaci:.4f}",
        "Entropy (Encrypted Image)": f"{entropy_enc:.4f}",
        "Mean Squared Error (MSE)": mse,
        "Peak Signal-to-Noise Ratio (PSNR, dB)": psnr,
        "SSIM (Structural Similarity Index)": ssim_val,
        "Chi-Square Statistic": chi2,
        "Chi-Square P-value": p_val,
        "Key Sensitivity (Pixel Changes)": key_diff,
        "Avalanche Effect - Avg NPCR (%)": f"{avg_npcr:.4f}",
        "Avalanche Effect - Avg UACI (%)": f"{avg_uaci:.4f}"
    }

    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_metrics.xlsx")
    pd.DataFrame([results_dict]).to_excel(excel_path, index=False)

    # --- Save plots ---
    correlation_enc_plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_correlation_enc.png")
    plot_correlation(image, encrypted_image, correlation_enc_plot_path, title="Original vs Encrypted")

    correlation_dec_plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_correlation_dec.png")
    plot_correlation(image, decrypted_image, correlation_dec_plot_path, title="Original vs Decrypted")

    histogram_plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_histograms.png")
    plot_histograms(image, encrypted_image, histogram_plot_path)

    # --- Response ---
    return jsonify({
        'encrypted_image': f"/{encrypted_path}",
        'decrypted_image': f"/{decrypted_path}",
        'excel_file': f"/{excel_path}",
        'correlation_plot_encrypted': f"/{correlation_enc_plot_path}",
        'correlation_plot_decrypted': f"/{correlation_dec_plot_path}",
        'histogram_plot': f"/{histogram_plot_path}"
    })

@app.route('/static/results/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
