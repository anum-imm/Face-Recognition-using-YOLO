import os
import psycopg2
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import torch
from PIL import Image
import io
import numpy as np
from datetime import datetime
from deepface import DeepFace
import os
import cv2

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Make sure to keep this secure!

# Ensure the uploads directory exists
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads directory if it doesn't exist

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best-6.pt')  # Update the path to your model
model.eval()

def get_conn():
    """Establish connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname="attendance_system",  # Update your database name
            user="postgres",  # Update your username
            password="123456789",  # Update your password
            host="localhost"  # Ensure your database is running locally or update the host if needed
        )
        return conn
    except Exception as e:
        flash(f"Error connecting to the database: {e}", "error")
        return None

def process_image(image):
    """Process the uploaded image with YOLO model."""
    results = model(image)
    # Get unique predictions and their counts
    # Initialize detections with all classes as 'Absent'
    detections = {name: 'Absent' for name in model.names.values()}
    
     # Create a directory to store cropped images
    cropped_faces_folder = './cropped_faces'
    os.makedirs(cropped_faces_folder, exist_ok=True)
    
    # Update detections for present students
    for pred in results.pred[0]:
        class_id = int(pred[5])
        if class_id in model.names:
            label = model.names[class_id]
            detections[label] = 'Present'

                        # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, pred[:4])

            # Crop the face
            cropped_face = image.crop((x1, y1, x2, y2))

            # Save the cropped face with a unique filename
            cropped_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{label}.png"
            cropped_face_path = os.path.join(cropped_faces_folder, cropped_filename)
            cropped_face.save(cropped_face_path)
        
    # Render results
    rendered_img = results.render()[0]  # returns list of images
    return Image.fromarray(rendered_img), detections

# @app.route('/')
# def index():
#     """Render the home page."""
#     return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle student registration and insert data into the database."""
    if request.method == 'POST':
        # Retrieve form data
        student_name = request.form['student_name']
        student_email = request.form['student_email']
        student_roll_number = request.form['student_roll_number']
        student_department = request.form['student_department']

        # Retrieve the image file
        student_image = request.files['student_image']
        if student_image.filename == '':
            flash('No image selected', 'error')
            return redirect(url_for('register'))

        # Save the image file
        image_filename = datetime.now().strftime('%Y%m%d%H%M%S') + "_" + student_image.filename
        save_path = os.path.join(UPLOAD_FOLDER, image_filename)
        student_image.save(save_path)

        # Establish database connection
        conn = get_conn()
        if not conn:
            flash("Error connecting to the database", "error")
            return redirect(url_for('index'))

        try:
            # Insert the data into the database along with the image filename
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO students (name, email, roll_number, department, image_filename)
                VALUES (%s, %s, %s, %s, %s)
            """, (student_name, student_email, student_roll_number, student_department, image_filename))

            conn.commit()
            cur.close()
            conn.close()

            # Render the success page
            #flash("Student registered successfully!", "success")
            return render_template('success.html')
        except psycopg2.Error as e:
            conn.rollback()
            flash(f"Database error: {e}", "error")
            return redirect(url_for('register'))  # Redirect back to registration form on error
        finally:
            conn.close()  # Ensure connection is closed even if an error occurs

    # If request method is GET, render the registration form
    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page with upload functionality."""
    global img_io
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'

        img = Image.open(file.stream)
        processed_img, detections = process_image(img)
        
        img_io = io.BytesIO()
        processed_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Update attendance in database
        conn = get_conn()
        if conn:
            cur = conn.cursor()
            current_date = datetime.now().date()
            
            # Insert attendance records
            for student_name, status in detections.items():
                cur.execute("""
                    INSERT INTO attendance (student_name, status, date)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (student_name, date) 
                    DO UPDATE SET status = EXCLUDED.status
                """, (student_name, status, current_date))
            
            conn.commit()
              # Fetch the most recent attendance records
            cur.execute("""
                SELECT student_name, status, date, created_at
                FROM attendance
                ORDER BY created_at DESC
                
            """)
            recent_attendance = cur.fetchall()  # Get recent attendance records
            
         
            conn.close()

        # Store processed image in session or temp storage
        img_io.seek(0)
            
        # Return detections data as JSON and image URL
        response = {
            'detections': detections,
            'image_url': url_for('get_processed_image', _external=True)
        }
        return response
    

    return render_template('index.html')  # Move your HTML to a template file

@app.route('/processed_image')
def get_processed_image():
    """
    Serve the processed image to the client.

    This function retrieves the processed image stored in memory (global variable `img_io`)
    and sends it as a response. The image is in PNG format.

    Returns:
    - The processed image if available (status 200).
    - A 404 error if the image is not found.
    """
    if 'img_io' in globals():  # Check if the global variable `img_io` exists
        return send_file(img_io, mimetype='image/png')  # Serve the image file with PNG mimetype
    return 'No image found', 404  # Return a 404 error if no image is found in memory


def load_known_embeddings():
    """
    Load known face embeddings from a file for comparison.

    This function attempts to load a dictionary of known face embeddings from the file `encodings.pkl`.
    Each entry in the dictionary typically maps a name to a corresponding face embedding (numpy array).

    Returns:
    - A dictionary of known embeddings if the file exists.
    - An empty dictionary if the file does not exist or cannot be loaded.
    """
    try:
        with open("encodings.pkl", "rb") as f:  # Attempt to open the file in binary read mode
            known_embeddings = pickle.load(f)  # Load the dictionary of embeddings using pickle
        return known_embeddings  # Return the loaded embeddings
    except FileNotFoundError:  # Handle the case where the file is not found
        print("No known embeddings found. Please ensure encodings.pkl exists.")  # Print an error message
        return {}  # Return an empty dictionary if the file is not found


def compare_embeddings(new_embedding, saved_embeddings, threshold=0.6):
    """
    Compare a new embedding with saved embeddings to find the closest match.

    Parameters:
    - new_embedding: numpy array, the embedding of the new face to compare.
    - saved_embeddings: dict, a dictionary where keys are names and values are embeddings (numpy arrays).
    - threshold: float, the maximum allowable distance for a match.

    Returns:
    - match_name: str, the name of the closest match (or 'Unknown' if no match is within the threshold).
    - min_distance: float, the distance to the closest match.
    """
    min_distance = float('inf')
    match_name = 'Unknown'

    for name, saved_embedding in saved_embeddings.items():
        # Calculate the Euclidean distance (or use another metric like cosine similarity)
        distance = np.linalg.norm(new_embedding - saved_embedding)

        if distance < min_distance:
            min_distance = distance
            match_name = name

    # Check if the closest match is within the threshold
    if min_distance > threshold:
        match_name = 'Unknown'

    return match_name, min_distance

if __name__ == '__main__':
    app.run(debug=True)