import cv2
import face_recognition
import os
import numpy as np
import time
import base64

# chống giả mạo
from challenge_response import challenge_action

# Thư mục ảnh tham chiếu
reference_dir = "img/"
success_image_path = "Modes/3.png"

# Tạo biến đọc và mã hóa khuôn mặt từ thư mục tham chiếu
known_encodings = []
known_names = []

# Load ảnh từ thư viện /img để mã hóa và so sánh
for filename in os.listdir(reference_dir):
    filepath = os.path.join(reference_dir, filename)
    image = face_recognition.load_image_file(filepath)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # Giữ tỷ lệ khung hình
    face_encodings = face_recognition.face_encodings(image)
    
    if face_encodings:  # Nếu tìm thấy khuôn mặt
        known_encodings.append(face_encodings[0])
        known_names.append(os.path.splitext(filename)[0])

print(known_names)  # Lấy dữ liệu ID qua tên ảnh

# IP Webcam
http = "https://192.168.1.203:3030/video"
cap = cv2.VideoCapture(http)
start_time = time.time()

while True:
    ret, frame = cap.read()
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm số frame lưu trong buffer

    if not ret:
        break
     
    # Chuyển khuôn mặt sang RGB và nhận diện khuôn mặt
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Chuyển danh sách thành mảng NumPy
    face_encoding_array = np.array(face_encodings, dtype=np.float32)
    # Mã hóa thành base64
    face_encoding_base64 = base64.b64encode(face_encoding_array.tobytes()).decode('utf-8')
    
    print(f"Số lượng khuôn mặt phát hiện: {len(face_locations)}")
    match_found = False  
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) 
        print(face_distances, best_match_index)
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if best_match_index is not None and face_distances[best_match_index] <= 0.43999999: ## tỉ lệ để được tính là khuôn mặt chuẩn
            name = known_names[best_match_index]  # Lấy tên file trùng khớp
            print(f"✅ Phát hiện khuôn mặt giống với {name}")
            
            #đọc và lấy dữ liệu ảnh khớp sang Base64 
            with open(reference_dir+name+".png", 'rb') as image_file:
                image_data = image_file.read()
            # Chuyển byte sang Base64
            base64_str = base64.b64encode(image_data).decode('utf-8')
            match_found = True
            # In tọa độ khuôn mặt và encoding
            print(f"Tọa độ khuôn mặt: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
            print(f"Encoding khuôn mặt (vector đặc trưng 128-d):\n {face_encoding}\n")
            print(f"Mã Base64 của dữ liệu gốc : \n{face_encoding_base64}\n")
            print(f"Mã hóa b64 của ảnh tham chiếu : \n{base64_str}\n")
            break  # Thoát vòng lặp khi tìm thấy khuôn mặt trùng khớp
        else : 
            print("⛔ Không phát hiện khuôn mặt trùng khớp.")
                #   {name}
            match_found = False

    if match_found:
        if not challenge_action(cap):
            print("⚠️ Không phát hiện nháy mắt. Có thể là ảnh/video.")
            match_found = False
            continue
        import subprocess
        subprocess.run(["python", "call_api.py", name])
        break  # Thoát khỏi vòng lặp chính

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Hiển thị ảnh thông báo nếu có khuôn mặt trùng khớp
success_image = cv2.imread("img/"+name+".png")
if success_image is not None and match_found:
    cv2.imshow("Match Found!", success_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
