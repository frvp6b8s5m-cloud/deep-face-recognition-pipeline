# Web-Based Face Recognition Pipeline

Browser-compatible facial recognition application built with TensorFlow.js and face-api.js.

## Features

- Real-time face detection in the browser
- Face embedding extraction using pre-trained models
- Similarity matching with configurable thresholds
- Drag-and-drop image upload
- Support for both cosine similarity and Euclidean distance metrics
- Responsive design for mobile and desktop

## Quick Start

### Option 1: Local Development

1. Serve the files with a local HTTP server:
   ```bash
   cd web
   python -m http.server 8000
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Option 2: Direct File Open

Simply open `index.html` directly in your browser (some features may be limited due to CORS restrictions).

## Usage

1. **Upload Probe Image**: Drag and drop or click to upload an image containing a face.
2. **View Detection**: The app detects faces and displays the first face as the probe.
3. **Upload Gallery**: Upload multiple images to match against the probe.
4. **View Matches**: See similarity scores and identify matches.
5. **Adjust Settings**: Change the similarity threshold and distance metric as needed.

## Models Used

- **Face Detection**: TensorFlow.js (face-api.js MTCNN)
- **Face Landmarks**: Face-api.js landmark detection
- **Face Embeddings**: Face-api.js face recognition net (based on ResNet)

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Dependencies

- TensorFlow.js 4.11.0
- face-api.js 0.22.2
- coco-ssd 2.2.3 (optional)

## Performance Notes

- First load takes ~2-3 seconds to load models
- Face detection: ~200-500ms per image
- Embedding extraction: included in detection
- Similarity matching: <1ms

## Limitations

- Models run entirely in the browser (no server required)
- Performance depends on device hardware
- Best results with clear, frontal face images
- Embedding model is smaller than production models (for performance)

## Advanced Configuration

Edit `app.js` to modify:

- `STATE.similarityThreshold`: Default similarity threshold (0.0-1.0)
- `STATE.metric`: Default distance metric ('cosine' or 'euclidean')
- Model URLs and options in `loadFaceDetector()`

## Troubleshooting

### Models fail to load
- Check internet connection (models are loaded from CDN)
- Clear browser cache
- Try a different browser

### No faces detected
- Ensure image has a clear frontal face
- Try with a different image
- Check browser console for errors

### Performance is slow
- Close other browser tabs
- Use a more powerful device or GPU-enabled browser
- Reduce image resolution before upload

## License

MIT
