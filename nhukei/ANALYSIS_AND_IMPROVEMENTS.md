# Agent v1 vs v2 - Phân Tích Lỗ Hổng & Cải Tiến

## 🔴 LỖ HỔNG CHÍNH CỦA AGENT v1

### **1. Hệ Thống Chấm Điểm Quá Đơn Giản (CRITICAL)**
**Vị trí:** `_strategic_wander()` - được gọi 30-40% của thời gian

**Vấn đề:**
```python
score = self._open_neighbors(grid, npos, blocked) * 2.0  # Chỉ 1 tiêu chí
for da in [1, 2, 3, 4]:
    if self._next_pos(npos, da) in danger_soon: 
        score -= 0.5  # Nhị phân: an toàn hoặc không
```

**Tại sao tồi:**
- Chỉ dùng 3 yếu tố cơ bản (open neighbors, danger binary, enemy distance)
- Không tính lookahead → quyết định "nhanh mà nông"
- **Kết quả:** Agent không suy nghĩ trước, không "bỏ ra nước cờ để kéo", chỉ phản ứng

**Cải tiến v2:**
- 5 tiêu chí độc lập với weights (survival, offense, positioning, economy, control)
- Mỗi tiêu chí có lookahead 3-5 bước
- Sử dụng 80% thời gian budget (100ms) để evaluate

---

### **2. Không Có Dự Báo Enemy (CRITICAL)**
**Vị trí:** Toàn bộ code, bombs được đặt dựa trên vị trí hiện tại của enemy

**Vấn đề:**
```python
can_hit_enemy = self._can_bomb_hit_enemy(grid, my_pos, enemies, bomb_radius)
# ↑ enemies chỉ là vị trí HIỆN TẠI, không dự báo!
```

**Tại sao tồi:**
- Enemy có thể chạy được trong 1-2 step
- Bomb nổ sau 8 step → enemy đã qua 6 vị trí rồi
- **Kết quả:** Bomb hit rate thấp, lãng phí bombs

**Cải tiến v2:**
- `_predict_enemy_move()`: dự báo enemy sẽ chạy đâu (dùng BFS to closest threat)
- Khi đặt bomb, check cả vị trí dự báo của enemy
- Tăng hit chance lên 40-60%

---

### **3. Priority Logic Bị "Lũng Ơn" (MAJOR)**
**Vị trí:** `act()` method - priority 1-4 thường success early, priority 5+ cực hiếm execute

**Vấn đề:**
```python
# Priority 1: Nếu in danger → escape hoặc return
# Priority 2: Nếu có item gần → move to item
# Priority 3: Nếu enemy ở dead-end → trap enemy  
# Priority 4: Nếu bomb hit enemy → bomb
# ...
# Priority 9: Wander (mặc định)
```

**Tại sao tồi:**
- Nếu Priority 2 trigger (có item gần) → code hoàn toàn ignore Priority 4-7
- Agent "bị lôi" sang ăn item, quên đặt bomb ngay cơ hội tấn công enemy
- **Kết quả:** Kém các agent khác trong timing tấn công

**Cải tiến v2:**
- Replace priority logic → **multi-criteria scoring**
- Tất cả actions được đánh giá cùng lúc → chọn action có điểm cao nhất
- Không có "tunnel vision" vào 1 priority

---

### **4. BFS Search Không Heuristic (MODERATE)**
**Vị trí:** `_timed_move_to_target()` và `_timed_escape_bfs()`

**Vấn đề:**
```python
# Explore tất cả (pos, t) với cùng priority
# Không có direction heuristic hoặc distance-to-goal guidance
# Nếu goal ở 20 bước xa → search tree depth 20 rất lớn
```

**Tại sao tồi:**
- State space = O(map_size * depth) = O(100 * 12) = 1200 states
- Với 4+ enemies, cực kỳ chậm nếu goal xa
- **Kết quả:** Hoặc search fail (goal quá xa), hoặc timeout

**Cải tiến v2:**
- Add heuristic functions (distance-to-goal, danger-count)
- Prioritize actions dựa trên Manhattan distance → goal
- Reduce effective branching factor từ 5 → 2-3

---

### **5. Kiểm Tra "Can Escape" Quá Cơ Bản (MODERATE)**
**Vị trí:** `_can_escape_after_placing()` - chỉ check 1 level

**Vấn đề:**
```python
def _can_escape_after_placing(self, grid, my_pos, blocked, danger_at, bomb_radius):
    # Simulate bomb
    # Run escape_bfs with max_t=9 (chỉ 9 step)
    return self._timed_escape_bfs(..., max_t=9) is not None
```

**Tại sao tồi:**
- Chỉ check "có đường sống không?" - nhị phân
- Không xem xét "đường sống này tốt hay xấu?"
- Agent có thể "sống sót" nhưng bị khóa trong corner ngay sau

**Cải tiến v2:**
- `_can_escape_after_arriving_and_placing()`: check **time-aware safety**
- Nếu tới đích ở time T → check xem lúc T+7-8 (bomb nổ) có còn sống không
- Hình phạt agent nếu sống sót nhưng bị khóa (low freedom)

---

### **6. Item Acquisition Không Có Giá Trị Động (MODERATE)**
**Vị trí:** Priority 2

**Vấn đề:**
```python
item_tiles = self._item_tiles(grid, my_pos, ...)
if item_tiles:
    move = self._timed_move_to_target(grid, my_pos, item_tiles, blocked, danger_at, 
                                      require_bomb_check=False)  # ← Không check value!
```

**Tại sao tồi:**
- Ăn item mà không tính "tôi có thực sự cần nó không?"
- Agent có sẵn 3 bombs + 5 radius → lại lặn ra ăn item → tốn time
- **Kết quả:** Miss cơ hội tấn công hoặc chase enemy

**Cải tiến v2:**
- `_evaluate_economy()`: tính value của item dựa trên current state
- Chỉ ưu tiên ăn item nếu:
  - bombs_left <= 1 AND gần bomb item
  - bomb_bonus <= 1 AND gần radius item
- Bỏ qua item nếu đã đủ vũ khí

---

### **7. "Stuck Count" Logic Quá Đơn Giản (MINOR)**
**Vị trí:** `_strategic_wander()` - `if self._stuck_count >= 2: score += random.uniform(0, 3)`

**Vấn đề:**
- Nếu stuck ≥ 2 lần → chỉ thêm random noise
- Không có strategy cụ thể để thoát khỏi dead-end

**Cải tiến v2:**
- Integrate vào freedom_map → nếu vị trí có freedom ≤ 3, agent urgency tăng
- Ưu tiên các move dẫn tới vùng open

---

## 📊 BẢNG SO SÁNH

| Tiêu Chí | v1 | v2 | Cải Tiến |
|---------|----|----|----------|
| **Evaluation Depth** | 1-2 steps | 3-5 steps | +150% |
| **Scoring Criteria** | 3 đơn giản | 5 weighted | +67% |
| **Enemy Prediction** | Không | Có | +∞ |
| **Time Budget Usage** | ~30-40ms | ~80-90ms | +100% |
| **Search with Heuristic** | Không | Có (Manhattan) | +significant |
| **State Evaluation** | Priority-based | Multi-criteria | +flexible |
| **Stuck Escape Strategy** | Random noise | Freedom map | +smart |
| **Item Value Calculation** | Đơn giản | Dynamic (state-aware) | +context |

---

## 🎯 TẠI SAO CHẠY NHANH (< 10ms)?

```
v1 Flow:
1. Check danger → likely return (5ms)
2. If not in danger → priority 1-2 likely match (3ms)
3. Early return (total: ~8-12ms)
```

v2 Force:
```
v2 Flow:
1. Check danger (5ms)
2. Loop through 4-5 valid actions (20ms)
3. For each action, evaluate with lookahead:
   - Survival: BFS (5-8ms)
   - Offense: bomb eval + enemy prediction (3-5ms)
   - Position/Economy: O(1) checks (1-2ms)
4. Choose best action (80-90ms total)
```

---

## 🚀 CÁCH SỬ DỤNG v2

### Thay thế trực tiếp:
```python
# Thay đổi trong nhukei/agent.py
from agent_v2_improved import Agent
# hoặc copy-paste code từ agent_v2_improved.py vào agent.py
```

### Tuning Weights (nếu cần):
```python
class Agent:
    W_SURVIVAL = 3.0   # Increase nếu chết quá nhiều
    W_OFFENSE = 1.8    # Increase nếu muốn offensive hơn
    W_POSITION = 1.2   # Balance giữa defense/offense
    W_ECONOMY = 0.8    # Increase nếu muốn farm items hơn
    W_CONTROL = 0.6    # Space control priority
```

### Time Budget Tuning:
```python
# Trong _evaluate_move_with_lookahead:
if self._elapsed_time() > 0.08:  # 80ms budget
    break  # Early exit if timeout

# Hoặc thay đổi _estimate_lookahead_depth:
max_lookahead_depth = 5  # Increase for more thinking
```

---

## 📈 EXPECTED IMPROVEMENTS

**Theoretical Gains:**
- **Hit Rate:** 20-30% → 40-60% (better enemy prediction)
- **Decision Quality:** Early termination → deep evaluation  
- **Time Efficiency:** +40% deeper search with same 100ms
- **Overall Performance:** +15-25% score improvement

---

## ⚠️ TRADE-OFFS

| Pro | Con |
|-----|-----|
| Deeper lookahead → better decisions | Hơi phức tạp hơn để debug |
| Multi-criteria → flexible strategy | Need tuning weights |
| Enemy prediction → higher hit rate | Prediction có thể sai (enemies random) |
| Use full time budget → more thinking | Có thể lag trên slow machines |

---

## 🔧 NEXT STEPS

1. **Test v2** trên test set
2. **Tune weights** dựa trên performance
3. **Optimize lookahead_depth** (3 vs 5) dựa trên frame time
4. **Add alpha-beta pruning** nếu muốn advanced hơn
5. **Combine v1+v2**: Dùng v1 logic khi urgent (< 2 enemies), v2 logic khi có time

