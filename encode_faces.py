import face_recognition
import os
import pickle

known_encodings = []
known_names = []

for file in os.listdir("faces"):
    name = file.split(" ")[0]    # موقعی که عکسو میدید باید اسم رو با شماره جدا کنید

    img = face_recognition.load_image_file(f"faces/{file}")
    enc = face_recognition.face_encodings(img)

    if len(enc) > 0:
        known_encodings.append(enc[0])
        known_names.append(name)
    

with open("faces.db", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Database is ready")
