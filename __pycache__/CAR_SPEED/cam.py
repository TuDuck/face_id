import cv2
import numpy as np
import time

# Khởi tạo video capture
cap = cv2.VideoCapture('Lane.mp4')

# Lấy tốc độ khung hình của video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1 / fps

# Biến để tính toán tốc độ
distance_between_lines = 10  # Khoảng cách thực tế giữa hai vạch đứt (mét)
line_cross_times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển đổi khung hình sang không gian màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng bộ lọc Gaussian để làm mịn ảnh
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sử dụng Canny Edge Detection để phát hiện cạnh
    edges = cv2.Canny(blur, 50, 150)
    
    # Định nghĩa vùng quan tâm (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), int(height * 0.6)),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    
    # Sử dụng Hough Transform để phát hiện các đường thẳng
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if 0.5 < abs(slope) < 2:  # Chỉ giữ lại các đường có độ dốc trong khoảng này
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                
                # Kiểm tra nếu xe đã qua vạch kẻ đứt
                if len(line_cross_times) == 0 or (time.time() - line_cross_times[-1]) > 1:  # Đảm bảo không ghi nhận nhiều lần cho cùng một vạch
                    line_cross_times.append(time.time())
    
    # Tính toán tốc độ khi xe qua hai vạch đứt
    if len(line_cross_times) >= 2:
        time_diff = line_cross_times[-1] - line_cross_times[-2]
        speed = distance_between_lines / time_diff  # Tính tốc độ m/s
        cv2.putText(frame, f"Speed: {speed:.2f} m/s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Speed', frame)
    
    # Thêm thời gian chờ giữa các khung hình
    time.sleep(frame_delay)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
