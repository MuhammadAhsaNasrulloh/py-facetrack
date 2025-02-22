import sqlite3
import face_recognition
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

def create_database():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    # Explicitly specify BLOB type and add a size column for verification
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL,
        encoding_size INTEGER NOT NULL,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    return conn, cursor

def save_face_from_camera(name):
    conn, cursor = create_database()
    
    try:
        video_capture = cv2.VideoCapture(0)
        print("Arahkan wajah ke kamera dan tekan 's' untuk menyimpan...")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Gagal mengakses kamera.")
                break
                
            cv2.imshow("Capture Face", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Save image
                cv2.imwrite(f"{name}.jpg", frame)
                print(f"Foto {name}.jpg berhasil disimpan.")
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Load and process image
        image = face_recognition.load_image_file(f"{name}.jpg")
        encodings = face_recognition.face_encodings(image)
        
        if not encodings:
            print("❌ Tidak ada wajah yang terdeteksi! Coba lagi.")
            return
            
        encoding_array = encodings[0]
        encoding_bytes = encoding_array.tobytes()
        encoding_size = len(encoding_bytes)
        
        # Debug prints
        print(f"Encoding shape: {encoding_array.shape}")
        print(f"Encoding size in bytes: {encoding_size}")
        
        # Save to database with size verification
        cursor.execute(
            "INSERT INTO faces (name, encoding, encoding_size) VALUES (?, ?, ?)",
            (name, encoding_bytes, encoding_size)
        )
        conn.commit()
        
        # Verify the save
        cursor.execute("SELECT encoding, encoding_size FROM faces WHERE name = ?", (name,))
        saved_data = cursor.fetchone()
        
        if saved_data:
            saved_encoding, saved_size = saved_data
            if len(saved_encoding) == saved_size:
                print(f"✅ Wajah {name} berhasil disimpan ke database!")
                print(f"Ukuran data tersimpan: {saved_size} bytes")
            else:
                print("⚠️ Data tersimpan tidak sesuai dengan ukuran yang diharapkan!")
        
        # Display the image
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Wajah {name} yang disimpan")
        plt.show()
        
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
        
    finally:
        conn.close()

# Test the function
if __name__ == "__main__":
    save_face_from_camera("ahsan")