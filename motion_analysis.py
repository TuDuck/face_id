# motion_analysis.py
# def detect_blink(cap, threshold, timeout):
#     import time
#     import cv2
#     import mediapipe as mp

#     mp_face_mesh = mp.solutions.face_mesh
#     LEFT_EYE = [362, 385, 387, 263, 373, 380]
#     RIGHT_EYE = [33, 160, 158, 133, 153, 144]

#     def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
#         eye = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
#         ver = ((eye[1][1] + eye[2][1]) / 2) - ((eye[5][1] + eye[4][1]) / 2)
#         hor = eye[0][0] - eye[3][0]
#         ratio = abs(ver / hor) if hor != 0 else 0
#         return ratio

#     with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
#         start_time = time.time()
#         while time.time() - start_time < timeout:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(image)

#             if results.multi_face_landmarks:
#                 h, w, _ = frame.shape
#                 landmarks = results.multi_face_landmarks[0].landmark
#                 left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
#                 right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
#                 avg_ear = (left_ear + right_ear) / 2.0

#                 if avg_ear < threshold:
#                     return True
#             cv2.waitKey(1)
#         return False

def detect_blink(cap, threshold_scale, timeout, calibration_time):
    import time
    import cv2
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
        eye = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
        ver = ((eye[1][1] + eye[2][1]) / 2) - ((eye[5][1] + eye[4][1]) / 2)
        hor = eye[0][0] - eye[3][0]
        ratio = abs(ver / hor) if hor != 0 else 0
        return ratio

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        # Calibration phase: collect EAR while eyes are open
        baseline_ears = []
        start_time = time.time()
        while time.time() - start_time < calibration_time:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                baseline_ears.append(avg_ear)
            cv2.waitKey(1)
        
        # Nếu không thu được dữ liệu để tính EAR mở
        if not baseline_ears:
            return False
        
        open_ear = sum(baseline_ears) / len(baseline_ears)
        threshold = open_ear * threshold_scale

        # Blink detection phase
        start_time = time.time()
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < threshold:
                    return True
            cv2.waitKey(1)
        return False
