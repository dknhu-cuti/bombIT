import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================
# 1. XÂY DỰNG DATASET & DATALOADER
# ==========================================
class BombItDataset(Dataset):
    def __init__(self, data_dir="il_dataset"):
        self.states = []
        self.actions = []
        
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not npz_files:
            raise ValueError(f"Không tìm thấy file .npz nào trong thư mục {data_dir}. Hãy chạy Giai đoạn 1 trước!")
            
        print(f"Đang load {len(npz_files)} files dataset từ thư mục '{data_dir}'...")
        for f in npz_files:
            data = np.load(f)
            self.states.append(data['states'])
            self.actions.append(data['actions'])
            
        # Nối tất cả các trận đấu lại thành 1 mảng lớn duy nhất
        self.states = np.concatenate(self.states, axis=0).astype(np.float32)
        self.actions = np.concatenate(self.actions, axis=0).astype(np.int64)
        
        print(f"Tổng số sample (steps) thu thập được: {len(self.states)}")
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ==========================================
# 2. THIẾT KẾ MẠNG NEURAL (CNN)
# ==========================================
class BombItCNN(nn.Module):
    def __init__(self):
        super(BombItCNN, self).__init__()
        
        # Mạng nhẹ dành cho Real-time inference
        # Input size: (Batch, 6, 13, 13)
        self.conv_block = nn.Sequential(
            # Layer 1: Giữ nguyên spatial dimensions (13x13) do padding=1
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Layer 2: Trích xuất thêm đặc trưng
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Duỗi dữ liệu: 64 channels * 13 width * 13 height = 10816 features
        self.flatten = nn.Flatten()
        
        # Mạng Fully Connected
        self.fc_block = nn.Sequential(
            nn.Linear(10816, 256),
            nn.ReLU(),
            nn.Linear(256, 6) # 6 outputs tương ứng với 6 nút bấm (0 -> 5)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    # Tự động nhận diện thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang huấn luyện trên thiết bị: {device}")
    
    # Chuẩn bị Data
    full_dataset = BombItDataset("il_dataset")
    
    # Chia tỉ lệ Train 80%, Validation 20%
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Chia dataset thành công: {train_size} Train | {val_size} Val")
    
    # Khởi tạo mô hình và Optimizer
    model = BombItCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 15
    best_val_loss = float('inf')
    
    print("\nBẮT ĐẦU HUẤN LUYỆN...")
    for epoch in range(num_epochs):
        # ----------------- TRAIN -----------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            
            # Forward
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Tính toán Metric
            train_loss += loss.item() * states.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += actions.size(0)
            train_correct += (predicted == actions).sum().item()
            
        train_loss /= train_total
        train_acc = 100 * train_correct / train_total
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                outputs = model(states)
                loss = criterion(outputs, actions)
                
                val_loss += loss.item() * states.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += actions.size(0)
                val_correct += (predicted == actions).sum().item()
                
        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        
        # In log
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
        # Lưu best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_il_model.pth")
            print(f" -> Đã lưu Model tốt nhất (Val Loss cải thiện xuống {val_loss:.4f})")

if __name__ == "__main__":
    train()

"""
=====================================================
HƯỚNG DẪN INFERENCE (Sử dụng model đã train vào Game)
=====================================================

Sau khi chạy xong script này và có file `best_il_model.pth`, 
bạn hãy tạo một file `agent_il.py` (Deep Learning Agent) có cấu trúc như sau:

import torch
import numpy as np
from train_il import BombItCNN # Import class mạng nơ-ron

class ILAgent:
    team_id = "Nhukei_IL_Bot"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        
        # Khởi tạo model và đưa lên thiết bị (CUDA/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BombItCNN().to(self.device)
        
        # Load trọng số đã học được
        self.model.load_state_dict(torch.load("best_il_model.pth", map_location=self.device))
        self.model.eval() # Bật chế độ đánh giá (tắt Dropout/BatchNorm nếu có)

    # COPY HÀM _preprocess_obs TỪ GIAI ĐOẠN 1 SANG ĐÂY
    def _preprocess_obs(self, grid, danger_soon, danger_now, my_pos, enemies):
        # ... mã nguồn y hệt agent cũ ...
        return state

    def act(self, obs):
        # ... Tính toán my_pos, enemies, danger_soon, danger_now ...
        
        # 1. Chuyển đổi bản đồ sang Tensor numpy
        state_np = self._preprocess_obs(grid, danger_soon, danger_now, my_pos, enemies)
        
        # 2. Convert sang PyTorch Tensor, cast sang Float và thêm chiều Batch size = 1
        # Kích thước trở thành: (1, 6, 13, 13)
        state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Chạy qua mạng nơ-ron
        with torch.no_grad():
            outputs = self.model(state_tensor)
            
            # 4. Lấy action có xác suất cao nhất
            predicted_action = torch.argmax(outputs, dim=1).item()
            
        return predicted_action
"""
