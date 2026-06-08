import base64
import os

# Resolve all paths relative to this script's own directory so the script
# works correctly no matter which folder it is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Đọc file não bộ nén dạng .zip mà m vừa train xong
zip_path = os.path.join(_HERE, "bomber_final_agent.zip")
with open(zip_path, "rb") as f:
    encoded_string = base64.b64encode(f.read()).decode('utf-8')

# Tự động mở file agent.py ra và nhét chuỗi này vào biến BRAIN_BASE64
agent_file_path = os.path.join(_HERE, "agent.py")
with open(agent_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip().startswith("BRAIN_BASE64 ="):
        lines[i] = f'BRAIN_BASE64 = "{encoded_string}"\n'
        break

with open(agent_file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Đã nhúng não thành công vào file agent.py đem đi nộp!")