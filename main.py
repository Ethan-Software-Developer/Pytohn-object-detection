import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

def detect_objects(image_bytes):
    """Detect objects in the uploaded image using the YOLOv5 model."""
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    
    # Perform object detection
    results = model(img_tensor)
    detections = results.pandas().xyxy[0]  # Bounding box data as pandas DataFrame

    # Extract relevant information
    detected_objects = []
    for _, row in detections.iterrows():
        detected_objects.append({
            'label': row['name'],
            'confidence': float(row['confidence']),
            'xmin': int(row['xmin']),
            'ymin': int(row['ymin']),
            'xmax': int(row['xmax']),
            'ymax': int(row['ymax'])
        })
    return detected_objects

@app.route('/detect', methods=['POST'])
def detect():
    """API endpoint to handle image uploads and return detection results."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image_bytes = file.read()

    try:
        results = detect_objects(image_bytes)
        return jsonify({'detections': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Object Detection API. Use the /detect endpoint with an image."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
