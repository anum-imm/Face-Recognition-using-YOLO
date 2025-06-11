from deepface import DeepFace
import os
import pickle

def prepare_and_save_encodings(known_faces_folder, output_file):
    """Encode faces and save the encodings and names to a file using DeepFace."""
    known_face_encodings = []
    known_face_names = []
    
    # Loop through images in the folder
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_folder, filename)
            
            # Use DeepFace to analyze the image and get face embeddings
            try:
                result = DeepFace.represent(image_path, model_name="VGG-Face", enforce_detection=False)
                if result:
                    # DeepFace returns a list of dictionaries; we take the first (and typically only) entry
                    encoding = result[0]["embedding"]
                    known_face_encodings.append(encoding)
                    # Use filename (without extension) as the person's name
                    known_face_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Save encodings and names to a file
    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Encodings saved to {output_file}")

# Run the function
prepare_and_save_encodings('./known_faces', 'encodings.pkl')
