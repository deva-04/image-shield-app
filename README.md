# 🛡️ Image Shield: Secure Image Encryption Web App

**Image Shield** is a Flask-based web application that performs advanced image encryption and decryption using chaotic maps, permutation-diffusion mechanisms, and RCA logic. It’s designed for both performance analysis and practical usage in secure image processing.

🚀 Live Demo: [https://image-shield-app.onrender.com](https://image-shield-app.onrender.com)

---

## 🔒 Features

- ✅ Upload and encrypt grayscale images
- 🔁 Reversible decryption with high fidelity
- 📊 Performance metrics: PSNR, SSIM, Entropy, NPCR, UACI, Chi-square, Key Sensitivity
- 📉 Histogram and correlation plots
- 📦 Downloadable results (Excel + images)
- 🖼️ Auto image resizing (256x256) to ensure speed and reliability

---

## 📥 Installation & Local Run

# Clone the repo

```bash
git clone https://github.com/deva-04/image-shield-app.git

cd image-shield-app

```

# Create virtual environment

<pre> python -m venv .venv  
  
 source .venv/bin/activate
  
 # On Windows: .venv\Scripts\activate  </pre>

# Install dependencies
```bash
pip install -r requirements.txt
```

# Run the app
```bash
python app.py
```  

- Then open [localhost](http://127.0.0.1:5000/) in your browser.

## 🌐 Deployment

This app is deployed on Render using Gunicorn:

- `render.yaml` or deploy setting: `gunicorn app:app`

---
**Made with 🤍**



