import face_recognition
import cv2
import sqlite3
import numpy as np

def initialize_face_recognition():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, encoding FROM faces")
    known_faces = cursor.fetchall()
    
    known_names = []
    known_encodings = []
    
    for name, encoding_bytes in known_faces:
        try:
            # Convert BLOB to numpy array with explicit error checking
            encoding_array = np.frombuffer(encoding_bytes, dtype=np.float64)
            
            # Verify encoding length
            if len(encoding_array) != 128:
                print(f"Warning: Skipping invalid encoding for {name} (length={len(encoding_array)})")
                continue
                
            known_names.append(name)
            known_encodings.append(encoding_array)
            
        except Exception as e:
            print(f"Error loading encoding for {name}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(known_names)} faces from database")
    return conn, known_names, known_encodings

def process_frame(frame):
    # Ensure frame is in correct format
    if frame is None:
        return None, None
        
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Verify image dimensions
    if len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
        print("Error: Invalid image format")
        return None, None
        
    return rgb_frame, frame

def run_face_recognition():
    conn, known_names, known_encodings = initialize_face_recognition()
    
    if not known_names:
        print("No valid faces found in database!")
        return
        
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Cannot open webcam!")
        return
    
    print("Starting face recognition... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
                
            # Process frame
            rgb_frame, display_frame = process_frame(frame)
            if rgb_frame is None:
                continue
            
            # Detect faces
            try:
                face_locations = face_recognition.face_locations(rgb_frame)
                if not face_locations:
                    continue
                    
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Process each face
                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    try:
                        # Verify encoding format
                        if len(face_encoding) != 128:
                            print(f"Warning: Invalid face encoding length: {len(face_encoding)}")
                            continue
                            
                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                        name = "Tidak Dikenali"
                        
                        if True in matches:
                            match_index = matches.index(True)
                            name = known_names[match_index]
                        
                        # Draw rectangle
                        color = (0, 255, 0) if name != "Tidak Dikenali" else (0, 0, 255)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(display_frame, (left, top - 35), (right, top), color, cv2.FILLED)
                        cv2.putText(display_frame, name, (left + 6, top - 6),
                                  cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        
                    except Exception as e:
                        print(f"Error processing face: {str(e)}")
                        continue
                
                cv2.imshow('Face Recognition', display_frame)
                
            except Exception as e:
                print(f"Error in face detection: {str(e)}")
                continue
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        conn.close()

if __name__ == "__main__":
    run_face_recognition()