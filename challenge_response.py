# challenge_response.py
import time
import os
import cv2
import mediapipe as mp
import random

mp_face_mesh = mp.solutions.face_mesh

######################################
# Load ảnh nền
imgBackground = cv2.imread('Modes/background.png')
folderModePath = "Modes"

#Vòng for để được thư mục laatys từng ảnh của thư mục : lấy ảnh background 
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
##################################### show treen lap

def detect_spoofing_by_edges(frame, threshold_ratio=0.015):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = frame.shape[:2]
    suspicious_rects = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)

        if len(approx) == 4 and area > (width * height * threshold_ratio):
            # Là hình chữ nhật đủ lớn
            suspicious_rects += 1

    return suspicious_rects >= 1  # Phát hiện ít nhất 1 hình chữ nhật


def get_head_pose_direction(landmarks):
    nose_tip = landmarks[1]
    chin = landmarks[152]
    diff = nose_tip.x - chin.x
    if diff > 0.07:
        return "left"
    elif diff < -0.06:
        return "right"
    else:
        return "center"

def challenge_action(cap, timeout=10):
    challenges = ["turn_left", "turn_right"]
    random.shuffle(challenges)  # 🔀 Xáo trộn thử thách

    print(f"🔒 Bắt đầu thử thách. Vui lòng hoàn thành tất cả ({', '.join(challenges)}) trong {timeout} giây...")

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        start_time = time.time()

        for challenge in challenges:
            print(f"👉 Checking: {challenge.replace('_', ' ')}")

            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                if not ret:
                    break
                
                  # 🛡️ Phát hiện giả mạo bằng viền ảnh
                if detect_spoofing_by_edges(frame):
                    print("🚨 Phát hiện nghi ngờ ảnh 2D hoặc thiết bị giả mạo! Thử thách bị hủy.")
                    return False
                
                frame = cv2.resize(frame, (640, 480))
                imgBackground[162:162+480,55:55+640] = frame # tọa độ của ảnh nền chính 
                # imgBackground[44:44+633,808:808+414] = imgModeList[0] # tọa độ của các ảnh con : lấy ảnh số 4
    
                    
                text = ""
                if challenge == "turn_left":
                    text = "turn_left verify"
                elif challenge == "turn_right":
                    text = "turn_right verify"
                cv2.putText(imgBackground, f"Thử thách: {text}", (50, 180), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 6, 45), 2)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    if challenge == "turn_left":
                        if get_head_pose_direction(landmarks) == "left":
                            print("✅ Đã quay đầu trái.")
                            break

                    elif challenge == "turn_right":
                        if get_head_pose_direction(landmarks) == "right":
                            print("✅ Đã quay đầu phải.")
                            break

                cv2.imshow("Challenge Verification", imgBackground) # treen lap
                # cv2.imshow("Challenge Verification", frame) # tren cam


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
            else:
                # Nếu vòng while kết thúc mà không break (tức là chưa hoàn thành)
                print(f"⛔ Thất bại: Không hoàn thành thử thách '{challenge}' đúng hạn.")
                return False

    print("🎉 Hoàn thành tất cả thử thách!")
    return True
