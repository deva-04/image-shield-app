// Show preview of uploaded image
document.getElementById('image').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            const img = document.getElementById('uploadedPreview');
            img.src = event.target.result;
            img.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('encryptionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('image');
    const mu = document.getElementById('mu').value;
    const key = document.getElementById('key').value;
    const iterations = document.getElementById('iterations').value;
    const delta = document.getElementById('delta').value;

    if (fileInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    formData.append('mu', mu);
    formData.append('key', key);
    formData.append('iterations', iterations);
    formData.append('delta', delta);

    const response = await fetch('/process', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        alert(error.error || "Error during encryption/decryption.");
        return;
    }

    const data = await response.json();

    // Show image previews
    document.getElementById('previewEncrypted').src = data.encrypted_image;
    document.getElementById('previewDecrypted').src = data.decrypted_image;
    document.getElementById('previewCorr1').src = data.correlation_plot_encrypted;
    document.getElementById('previewCorr2').src = data.correlation_plot_decrypted;
    document.getElementById('previewHist').src = data.histogram_plot;

    // Set download links
    document.getElementById('downloadEncrypted').href = data.encrypted_image;
    document.getElementById('downloadDecrypted').href = data.decrypted_image;
    document.getElementById('downloadCorrelation1').href = data.correlation_plot_encrypted;
    document.getElementById('downloadCorrelation2').href = data.correlation_plot_decrypted;
    document.getElementById('downloadHistogram').href = data.histogram_plot;
    document.getElementById('downloadExcel').href = data.excel_file;

    // Show result containers
    document.getElementById('resultsContainer').classList.remove('hidden');
    document.getElementById('previewContainer').classList.remove('hidden');
    document.getElementById('excelDownloadLink').classList.remove('hidden');
});

// Function to open modal on image click
function openModal(imgSrc) {
    const modal = document.getElementById('imageModal');
    const modalImg = modal.querySelector('img');
    modalImg.src = imgSrc;
    modal.classList.add('show');
}

// Function to close modal
function closeModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('show');
}

// Event listener for closing the modal
document.querySelector('.modal .close').addEventListener('click', closeModal);

// Add click event to images for opening in full screen
document.querySelectorAll('.preview-grid img').forEach(img => {
    img.addEventListener('click', () => {
        openModal(img.src);
    });
});
