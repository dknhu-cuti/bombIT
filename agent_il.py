import os
import torch
import numpy as np
from nhukei.agent import Agent as RuleBasedAgent
from train_il import BombItCNN

class Agent(RuleBasedAgent):
    def __init__(self, agent_id: int):
        # Gọi __init__ của lớp cha để khởi tạo các biến cần thiết (ví dụ: team_id, escape_mode...)
        super().__init__(agent_id)
        self.team_id = "Nhukei_IL_Bot"
        
        # Thiết lập thiết bị chạy (CPU hoặc GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo mạng Neural
        self.model = BombItCNN().to(self.device)
        
        # Load file trọng số tốt nhất đã train
        # Sử dụng os.path để tự động tìm đúng đường dẫn thư mục hiện tại
        model_path = os.path.join(os.path.dirname(__file__), "best_il_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[IL Agent] Đã load thành công model từ: {model_path}")
        else:
            print(f"[IL Agent] CẢNH BÁO: Không tìm thấy file trọng số '{model_path}'. AI sẽ chạy bằng trọng số ngẫu nhiên!")
            
        self.model.eval() # Chuyển sang chế độ Inference

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        # Nếu agent đã chết hoặc id không hợp lệ, trả về 0 (Idle)
        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        # Lấy thông tin tọa độ cần thiết
        my_x, my_y, _, _, _ = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]
        
        # Tính toán danger_tiles (Sử dụng hàm đã tối ưu có sẵn từ lớp cha RuleBasedAgent)
        danger_soon, danger_now = self._danger_tiles(grid, bombs, players)
        
        # 1. Chuyển đổi trạng thái map thành Tensor 3D (6, 13, 13)
        # (Sử dụng hàm _preprocess_obs kế thừa trực tiếp từ RuleBasedAgent)
        state_np = self._preprocess_obs(grid, danger_soon, danger_now, my_pos, enemies)
        
        # 2. Convert sang PyTorch Tensor, cast sang Float và thêm chiều Batch size = 1 -> (1, 6, 13, 13)
        state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Chạy Forward qua mạng nơ-ron
        with torch.no_grad():
            outputs = self.model(state_tensor)
            
            # 4. Lấy index của nút bấm có xác suất/score cao nhất làm action tiếp theo
            predicted_action = torch.argmax(outputs, dim=1).item()
            
        return predicted_action
