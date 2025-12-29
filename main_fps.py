import cv2
import face_recognition
import pickle
import numpy as np
import time

with open("faces.db", "rb") as f:
    known_encodings, known_names = pickle.load(f)

cap = cv2.VideoCapture(0) # اگه بیشتر از یک دوربین دارید میتونید اینو عوض کنید 0 1 

process_this_frame = True
last_face_locations = []
last_face_names = []

fps_avg_frame_count = 30  
frame_times = []
display_fps = 0

while True:
    start_time = time.time() 
    
    ret, frame = cap.read()
    if not ret:
        break

    if process_this_frame:
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        last_face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, last_face_locations)

        last_face_names = []
        for face_enc in face_encodings:
            distances = face_recognition.face_distance(known_encodings, face_enc)
            name = "Unknown"
            if len(distances) > 0:
                best_match = np.argmin(distances)
                if distances[best_match] < 0.45:    # هر چی کوچیک تر بشه دقیق تر میشه ولی از اون طرف unknown زیاد میزنه
                    name = known_names[best_match]
            last_face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    end_time = time.time()
    frame_times.append(end_time - start_time)
   
    if len(frame_times) > fps_avg_frame_count:
        frame_times.pop(0)
    
    if len(frame_times) > 0:
        avg_time_per_frame = sum(frame_times) / len(frame_times)
        display_fps = int(1 / avg_time_per_frame)

    cv2.putText(frame, f"Avg FPS: {display_fps}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27: # با کلید esc میاد بیرون
        break

cap.release()
cv2.destroyAllWindows()