const STATE = {
  currentProbe: null,
  gallery: [],
  threshold: 0.6,
  metric: 'cosine',
  initialized: false,
  stream: null,
  selectedPanel: 'upload-panel',
};

const navButtons = document.querySelectorAll('.nav-btn');
const panels = document.querySelectorAll('.panel');
const imageInput = document.getElementById('imageInput');
const galleryInput = document.getElementById('galleryInput');
const dropZone = document.getElementById('dropZone');
const galleryDropZone = document.getElementById('galleryDropZone');
const thresholdSlider = document.getElementById('thresholdSlider');
const metricSelect = document.getElementById('metricSelect');
const matchSummary = document.getElementById('matchSummary');
const matchResults = document.getElementById('matchResults');

navButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    navButtons.forEach(b => b.classList.toggle('active', b === btn));
    const target = btn.dataset.panel;
    STATE.selectedPanel = target;
    panels.forEach(p => p.classList.toggle('active', p.id === target));
  });
});

thresholdSlider.addEventListener('input', (e) => {
  STATE.threshold = Number(e.target.value);
  runMatching();
});

metricSelect.addEventListener('change', (e) => {
  STATE.metric = e.target.value;
  runMatching();
});

setupDropZone(dropZone, imageInput, handleProbeFile);
setupDropZone(galleryDropZone, galleryInput, handleGalleryFiles);

document.getElementById('startCameraBtn').addEventListener('click', startCamera);
document.getElementById('captureBtn').addEventListener('click', captureProbeFromCamera);

async function init() {
  await loadModels();
  STATE.initialized = true;
  console.log('Face API loaded');
}

async function loadModels() {
  const MODEL_URL = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights';
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
}

function setupDropZone(zone, input, fileHandler) {
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const files = [...e.dataTransfer.files];
    if (files.length) fileHandler(files);
  });
  input.addEventListener('change', (e) => {
    if (e.target.files?.length) fileHandler([...e.target.files]);
  });
}

async function handleProbeFile(files) {
  const file = files[0];
  if (!file) return;
  const img = await loadImageFromFile(file);
  const detected = await detectFacesInImage(img);
  if (!detected.length) {
    alert('No face detected. Please choose another image.');
    return;
  }
  STATE.currentProbe = { fileName: file.name, image: img, descriptor: detected[0].descriptor };
  renderProbePreview();
  runMatching();
}

async function handleGalleryFiles(files) {
  const processed = [];
  for (const file of files) {
    const img = await loadImageFromFile(file);
    const detections = await detectFacesInImage(img);
    if (!detections.length) continue;
    processed.push({
      fileName: file.name,
      image: img,
      descriptor: detections[0].descriptor,
    });
  }
  STATE.gallery = processed;
  renderGalleryPreview();
  runMatching();
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

async function detectFacesInImage(img) {
  return await faceapi
    .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptors();
}

function renderProbePreview() {
  const preview = document.getElementById('probePreview');
  if (!STATE.currentProbe) {
    preview.innerHTML = '';
    return;
  }
  preview.innerHTML = `
    <div class="preview-box">
      <img src="${STATE.currentProbe.image.src}" alt="Probe preview" />
      <div class="preview-meta"><strong>${STATE.currentProbe.fileName}</strong></div>
    </div>
  `;
}

function renderGalleryPreview() {
  const galleryPreview = document.getElementById('galleryPreview');
  galleryPreview.innerHTML = '';
  STATE.gallery.forEach(item => {
    const card = document.createElement('div');
    card.className = 'gallery-item';
    card.innerHTML = `
      <img src="${item.image.src}" alt="${item.fileName}" />
      <div class="gallery-meta">
        <strong>${item.fileName}</strong>
      </div>
    `;
    galleryPreview.appendChild(card);
  });
}

function runMatching() {
  if (!STATE.currentProbe || !STATE.gallery.length) {
    matchSummary.textContent = 'Upload a probe image and gallery images to begin matching.';
    matchResults.innerHTML = '';
    return;
  }

  const matches = STATE.gallery.map(item => {
    const score = computeSimilarity(STATE.currentProbe.descriptor, item.descriptor, STATE.metric);
    return { ...item, score, matched: score >= STATE.threshold };
  }).sort((a, b) => b.score - a.score);

  const topMatch = matches[0];
  matchSummary.textContent = topMatch
    ? `Best match: ${topMatch.fileName} at ${(topMatch.score * 100).toFixed(2)}% using ${STATE.metric}.`
    : 'No matches available.';

  const cards = matches.map(item => `
    <div class="match-result ${item.matched ? 'matched' : 'no-match'}">
      <strong>${item.fileName}</strong>
      <div class="score">${(item.score * 100).toFixed(2)}% similarity</div>
      <div>${item.matched ? 'Match' : 'Not a match'} · Threshold ${STATE.threshold.toFixed(2)}</div>
    </div>
  `).join('');

  matchResults.innerHTML = cards;

  const galleryItems = document.querySelectorAll('.gallery-item');
  galleryItems.forEach((card, idx) => {
    card.classList.toggle('matched', matches[idx]?.matched);
  });
}

function computeSimilarity(embeddingA, embeddingB, metric) {
  const a = Array.from(embeddingA);
  const b = Array.from(embeddingB);

  if (metric === 'cosine') {
    return cosineSimilarity(a, b);
  }

  return 1 / (1 + euclideanDistance(a, b));
}

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

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

async function startCamera() {
  const video = document.getElementById('cameraVideo');
  const canvas = document.getElementById('cameraCanvas');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Camera API not supported in this browser.');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    STATE.stream = stream;
    video.srcObject = stream;
    video.play();
  } catch (error) {
    alert('Camera access denied or unavailable.');
    console.error(error);
  }
}

async function captureProbeFromCamera() {
  const video = document.getElementById('cameraVideo');
  const canvas = document.getElementById('cameraCanvas');

  if (!STATE.stream || video.readyState < 2) {
    alert('Start the camera before capturing.');
    return;
  }

  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataURL = canvas.toDataURL('image/png');
  const img = new Image();
  img.onload = async () => {
    const detected = await detectFacesInImage(img);
    if (!detected.length) {
      alert('No face detected in the camera frame.');
      return;
    }
    STATE.currentProbe = {
      fileName: 'camera_capture.png',
      image: img,
      descriptor: detected[0].descriptor,
    };
    renderProbePreview();
    runMatching();
  };
  img.src = dataURL;
}

init();

window.addEventListener('beforeunload', () => {
  if (STATE.stream) {
    STATE.stream.getTracks().forEach(track => track.stop());
  }
});























































































































































































































$n




















































































































































































































































































































































n











































































