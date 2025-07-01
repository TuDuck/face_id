import sys
import requests

if len(sys.argv) < 2:
    print("Thiếu đối số studentId")
    sys.exit(1)

student_id = sys.argv[1]

url = f"http://localhost:8080/api/attendance/update?studentId={student_id}"

try:
    response = requests.put(url)
    if response.status_code == 200:
        print(f"✅ Gọi API thành công cho {student_id}")
    else:
        print(f"⚠️ Gọi API thất bại: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ Lỗi khi gọi API: {e}")
