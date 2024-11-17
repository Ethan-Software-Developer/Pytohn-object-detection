Object Detection API
Overview
A Python-based Flask API that performs object detection on uploaded images using a pre-trained YOLOv5 model. The API accepts image uploads and returns detected objects with labels, confidence scores, and bounding box coordinates.

Features
Accepts image uploads via a POST request.
Uses YOLOv5 for object detection.
Returns detection results as JSON.
Lightweight and easy to integrate with other applications.
Requirements
Install dependencies:

bash
Copy code
pip install torch torchvision flask pillow requests
Running the Application
Clone this repo and run:

bash
Copy code
python object_detection_app.py
The server will run on http://localhost:5000.

Usage
Endpoint: /detect
Method: POST
Parameter: image (required)
Example using curl:

bash
Copy code
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/detect
Sample Response
json
Copy code
{
  "detections": [
    {"label": "person", "confidence": 0.97, "xmin": 34, "ymin": 45, "xmax": 230, "ymax": 300}
  ]
}
License
MIT License

