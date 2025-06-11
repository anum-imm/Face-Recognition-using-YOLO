from flask import Flask, request, render_template, send_file
import torch
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
from flask import Flask, request, render_template, send_file
import torch
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load YOLOv5 model (make sure the path is correct)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/apple/Desktop/dnn_proj/best-4.pt', force_reload=True)
model.eval()

def process_image(image):
    # Inference
    results = model(image)
    print(results)
    # Render results
    rendered_img = results.render()[0]  # returns list of images
    return Image.fromarray(rendered_img)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'

        # Read and process the image
        img = Image.open(file.stream)
        
        # Process image through YOLO
        processed_img = process_image(img)
        
        # Save to bytes
        img_io = io.BytesIO()
        processed_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')

    return '''
    <!doctype html>
    <html>
    <head>
        <title>Upload Image for Object Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                text-align: center;
            }
            #result-img {
                max-width: 100%;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YOLO Object Detection</h1>
            <form id="upload-form">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Detect Objects">
            </form>
            <img id="result-img" style="display: none;">
        </div>
        <script>
            document.getElementById('upload-form').onsubmit = function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                fetch('/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.blob())
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    const img = document.getElementById('result-img');
                    img.src = url;
                    img.style.display = 'block';
                })
                .catch(error => console.error('Error:', error));
            };
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)