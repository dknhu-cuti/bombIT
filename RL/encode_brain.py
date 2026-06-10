import base64
from pathlib import Path

# Sử dụng pathlib thay vì os.path để quản lý đường dẫn
# Script chạy từ thư mục RL hiện tại

# Đọc file não bộ nén dạng .zip mà vừa train xong
zip_path = Path("bomber_final_agent.zip")
with open(zip_path, "rb") as f:
    encoded_string = base64.b64encode(f.read()).decode('utf-8')

# Tự động mở file agent.py ra và nhét chuỗi này vào biến BRAIN_BASE64
agent_file_path = Path("agent.py")
with open(agent_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip().startswith("BRAIN_BASE64 ="):
        lines[i] = f'BRAIN_BASE64 = "{encoded_string}"\n'
        break

with open(agent_file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Đã nhúng não thành công vào file agent.py đem đi nộp!")