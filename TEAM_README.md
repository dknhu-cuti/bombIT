# 🚀 Nhóm Glitch - Các lệnh thường dùng

Đây là cẩm nang tóm tắt các lệnh Terminal (chạy trong WSL) mà team sẽ sử dụng thường xuyên nhất trong suốt quá trình phát triển thuật toán cho giải Bomberland.

## 1. Khởi động môi trường 
**(Luôn chạy đầu tiên khi mở Terminal WSL mới trong VS Code)**
```bash
conda activate aic_gdgoc
```
*(Nếu báo lỗi `conda: command not found` thì dùng lệnh ép kích hoạt: `source ~/miniconda3/bin/activate aic_gdgoc`)*

## 2. Chạy thử Bot (Xem đồ hoạ trực quan)
Quan sát bot `my_submission` của team chiến đấu với 3 con bot baseline để xem nó "khôn" hay "ngu" ở đâu mà sửa:
```bash
PYTHONPATH=. python scripts/participant/run_local_match.py --agent_paths my_submission RandomAgent SmarterRuleAgent TacticalRuleAgent --visualize True
```
*(Chữ `True` cuối cùng sẽ mở giao diện. Chỉnh thành `False` nếu muốn chạy ngầm thật nhanh chỉ lấy kết quả)*

## 3. Đo tốc độ phản xạ của Bot (CỰC KỲ QUAN TRỌNG)
Luật thi bắt buộc hàm `act()` phải trả kết quả **dưới 100ms** (0.1 giây). Lệnh sau đây cho bot chạy 5 trận ẩn để đo số millisecond (ms) trung bình mỗi lượt:
```bash
PYTHONPATH=. python scripts/participant/estimate_agent_time.py my_submission --opponents None None None --num_matches 5
```
*(Lưu ý: Luôn đảm bảo nó nằm ở khoảng 30-50ms ở nhà là an toàn nhất)*

## 4. Ước lượng sức mạnh (Rank) của Bot
Sau khi Code xong một thuật toán mới, hãy cho nó cày tự động 50-100 trận đấu không giao diện với các Baseline Agents để lấy ước lượng điểm TrueSkill:
```bash
PYTHONPATH=. python -m scripts.participant.estimate_rankings --agent_path my_submission --num_matches 50
```

## 5. Nén file để đem nộp bài
Theo luật, file `agent.py` phải nằm ở gốc của file `.zip`. Đứng ở thư mục lớn, bạn chạy lệnh sau để tự đóng gói file `submission.zip`:
```bash
cd my_submission
zip submission.zip agent.py 
cd ..
```
*(Nếu team có file đồ thị Weights của Deep Learning như `.pth` hoặc `.onnx` thì chỉ cần cách thêm tên file đó ra sau chữ `agent.py`)*

## 6. Lưu trữ và đồng bộ với đồng đội qua Github
Cứ mỗi khi code thành công thuật toán ngon, hãy lưu lại kẻo mất:
```bash
git add .
git commit -m "Thêm thuật toán A-Star tìm đồ nhanh hơn"
git push
```