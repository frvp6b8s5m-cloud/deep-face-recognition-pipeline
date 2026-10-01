// Face Recognition Web App

const STATE = {
    currentImage: null,
    currentEmbedding: null,
    galleryEmbeddings: [],
    similarityThreshold: 0.6,
    metric: 'cosine',
    faceDetectorLoaded: false,
};

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing Face Recognition Pipeline...');
    await loadFaceDetector();
    setupEventListeners();
    showStatus('Face detector loaded successfully', 'success');
});

// Load Face Detection Model
async function loadFaceDetector() {
    try {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights/';
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
            faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);
        STATE.faceDetectorLoaded = true;
    } catch (error) {
        console.error('Failed to load face detector:', error);
        showStatus('Error loading face detector: ' + error.message, 'error');
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Main image upload
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');

    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            imageInput.files = e.dataTransfer.files;
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageUpload(e.target.files[0]);
        }
    });

    // Gallery upload
    const galleryUploadArea = document.getElementById('galleryUploadArea');
    const galleryInput = document.getElementById('galleryInput');

    galleryUploadArea.addEventListener('click', () => galleryInput.click());
    galleryUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        galleryUploadArea.classList.add('dragover');
    });
    galleryUploadArea.addEventListener('dragleave', () => {
        galleryUploadArea.classList.remove('dragover');
    });
    galleryUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        galleryUploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleGalleryUpload(e.dataTransfer.files);
        }
    });

    galleryInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleGalleryUpload(e.target.files);
        }
    });

    // Settings
    document.getElementById('similarityThreshold').addEventListener('input', (e) => {
        STATE.similarityThreshold = parseFloat(e.target.value);
        document.getElementById('thresholdValue').textContent = e.target.value;
    });

    document.getElementById('metric').addEventListener('change', (e) => {
        STATE.metric = e.target.value;
    });
}

// Handle Image Upload
async function handleImageUpload(file) {
    if (!STATE.faceDetectorLoaded) {
        showStatus('Face detector not loaded yet', 'error');
        return;
    }

    try {
        const reader = new FileReader();
        reader.onload = async (e) => {
            const img = new Image();
            img.onload = async () => {
                STATE.currentImage = img;
                displayImagePreview(img);
                await detectAndExtractEmbedding(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('Error uploading image:', error);
        showStatus('Error uploading image: ' + error.message, 'error');
    }
}

// Display Image Preview
function displayImagePreview(img) {
    const previewContainer = document.getElementById('imagePreview');
    previewContainer.innerHTML = `
        <div class="preview-item">
            <img src="${img.src}" alt="Preview">
        </div>
    `;
}

// Detect Faces and Extract Embeddings
async function detectAndExtractEmbedding(img) {
    try {
        showStatus('Detecting faces...', 'info');

        const detections = await faceapi
            .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptors();

        if (detections.length === 0) {
            showStatus('No faces detected in the image', 'warning');
            document.getElementById('results').innerHTML = '<p>No faces detected</p>';
            return;
        }

        // Use first detected face
        const detection = detections[0];
        STATE.currentEmbedding = detection.descriptor;

        // Display results
        displayDetectionResults(detections, img);

        // Draw probe face
        drawProbeCanvas(img, detection);

        showStatus(`Detected ${detections.length} face(s)`, 'success');
    } catch (error) {
        console.error('Error detecting faces:', error);
        showStatus('Error detecting faces: ' + error.message, 'error');
    }
}

// Display Detection Results
function displayDetectionResults(detections, img) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = detections.map((d, idx) => `
        <div class="result-item">
            <strong>Face ${idx + 1}</strong>
            <p>Position: (${Math.round(d.detection.box.x)}, ${Math.round(d.detection.box.y)})</p>
            <p>Size: ${Math.round(d.detection.box.width)}x${Math.round(d.detection.box.height)}</p>
            <p>Confidence: ${(d.detection.score * 100).toFixed(2)}%</p>
        </div>
    `).join('');
}

// Draw Probe Canvas
function drawProbeCanvas(img, detection) {
    const canvas = document.getElementById('probeCanvas');
    const ctx = canvas.getContext('2d');
    const box = detection.detection.box;

    // Draw the detected face region
    ctx.drawImage(
        img,
        box.x,
        box.y,
        box.width,
        box.height,
        0,
        0,
        canvas.width,
        canvas.height
    );

    // Display embedding info
    const embeddingText = Array.from(detection.descriptor)
        .slice(0, 5)
        .map(v => v.toFixed(4))
        .join(', ') + '...';
    document.getElementById('probeEmbedding').textContent = `Embedding (first 5 dims): [${embeddingText}]`;
}

// Handle Gallery Upload
async function handleGalleryUpload(files) {
    if (!STATE.faceDetectorLoaded) {
        showStatus('Face detector not loaded yet', 'error');
        return;
    }

    if (!STATE.currentEmbedding) {
        showStatus('Please upload a probe image first', 'warning');
        return;
    }

    try {
        showStatus(`Processing ${files.length} gallery image(s)...`, 'info');
        STATE.galleryEmbeddings = [];

        for (const file of files) {
            const reader = new FileReader();
            reader.onload = async (e) => {
                const img = new Image();
                img.onload = async () => {
                    try {
                        const detections = await faceapi
                            .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
                            .withFaceLandmarks()
                            .withFaceDescriptors();

                        if (detections.length > 0) {
                            STATE.galleryEmbeddings.push({
                                name: file.name,
                                descriptor: detections[0].descriptor,
                                image: img,
                            });
                            displayGalleryResults();
                        }
                    } catch (error) {
                        console.error('Error processing gallery image:', error);
                    }
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    } catch (error) {
        console.error('Error uploading gallery:', error);
        showStatus('Error uploading gallery: ' + error.message, 'error');
    }
}

// Display Gallery Results
function displayGalleryResults() {
    const resultsContainer = document.getElementById('galleryResults');

    const results = STATE.galleryEmbeddings.map((item) => {
        const similarity = computeSimilarity(STATE.currentEmbedding, item.descriptor);
        const isMatch = similarity >= STATE.similarityThreshold;

        return {
            ...item,
            similarity,
            isMatch,
        };
    }).sort((a, b) => b.similarity - a.similarity);

    resultsContainer.innerHTML = results.map((r) => `
        <div class="gallery-item ${r.isMatch ? 'match' : ''}">
            <div class="name">${r.name}</div>
            <div class="score">${(r.similarity * 100).toFixed(2)}%</div>
        </div>
    `).join('');
}

// Compute Similarity
function computeSimilarity(embedding1, embedding2) {
    const e1 = Array.from(embedding1);
    const e2 = Array.from(embedding2);

    if (STATE.metric === 'cosine') {
        return cosineSimilarity(e1, e2);
    } else if (STATE.metric === 'euclidean') {
        return 1 / (1 + euclideanDistance(e1, e2));
    }

    return 0;
}

// Cosine Similarity
function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}

// Euclidean Distance
function euclideanDistance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

// Show Status Message
function showStatus(message, type = 'info') {
    const resultsContainer = document.getElementById('results');
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-message ${type}`;
    statusDiv.innerHTML = `<span>${message}</span>`;

    // Insert at top of results or create new container
    if (resultsContainer.firstChild) {
        resultsContainer.insertBefore(statusDiv, resultsContainer.firstChild);
    } else {
        resultsContainer.appendChild(statusDiv);
    }

    // Auto-remove after 5 seconds for info messages
    if (type === 'info' || type === 'success') {
        setTimeout(() => statusDiv.remove(), 5000);
    }
}
