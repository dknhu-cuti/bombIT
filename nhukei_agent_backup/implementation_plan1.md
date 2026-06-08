# Phân tích sâu NhukeiAgent vs Đối thủ — Kế hoạch cải tiến V5

## Phát hiện từ Engine (Cơ chế game thực tế)

### Thông tin chủ chốt từ engine:
1. **Bomb timer = 7 ticks** ([bomb.py:L2](file:///g:/CuocThi/aic/bombIT/engine/bomb.py#L2)). Agent có 6 bước di chuyển để thoát.
2. **Chain reaction**: Bom nổ gần bom khác sẽ kích nổ dây chuyền ([game.py:L110-L123](file:///g:/CuocThi/aic/bombIT/engine/game.py#L110-L123)).
3. **Item collection**: Nếu 2+ players đứng cùng ô item, item bị hủy, KHÔNG AI được nhặt ([game.py:L87](file:///g:/CuocThi/aic/bombIT/engine/game.py#L87)).
4. **Bom phá hộp → 30% drop item radius, 30% drop item capacity** ([game.py:L176-L180](file:///g:/CuocThi/aic/bombIT/engine/game.py#L176-L180)).
5. **Players CAN overlap** (đi xuyên qua nhau) ([player.py:L38-L40](file:///g:/CuocThi/aic/bombIT/engine/player.py#L38-L40)).
6. **blocked** trong code = chỉ cấm đi qua ô có BOM, không cấm đi qua ô có PLAYER khác (đúng theo engine).
7. **Start stats**: `max_bombs=1`, `bombs_left=1`, `bomb_radius_bonus=0` → radius nổ ban đầu = 1.
8. **Random item spawn** tăng dần theo thời gian: xác suất = `0.0003 * current_step / 165` ([game.py:L192](file:///g:/CuocThi/aic/bombIT/engine/game.py#L192)).

### Hệ thống chấm điểm:
- Đánh giá ranking dùng `estimate_rankings.py`: đấu **1 vs 3 opponents** (mix TacticalRule, SmarterRule, GeniusRule)
- Survivors ở step 500 đều nhận rank 0 (theo script chấm hiện tại)
- Theo tiêu chí mới: survivors được tiebreak bởi Kills > Boxes > Items > Bombs

---

## Phân tích chi tiết 6 BẪY / ĐIỂM YẾU trong code hiện tại

### BẪY 1: `blocked` chứa cả `occupied` (vị trí kẻ địch) → SAI ENGINE

**Vấn đề**: Code hiện tại:
```python
blocked = set(occupied) | bomb_positions  # dòng 45
```
Trong engine thực tế, **players CAN overlap** ([player.py:L38](file:///g:/CuocThi/aic/bombIT/engine/player.py#L38)). Chỉ có bom mới chặn di chuyển.

Hệ quả:
- Agent coi ô có kẻ địch đứng là "không đi được" → BFS tìm đường đi vòng rất xa
- Khi kẻ địch đứng chặn hành lang hẹp, agent bị tắc hoàn toàn
- TacticalRuleAgent & GeniusRuleAgent đều chỉ block bom, KHÔNG block players:
  - Tactical: `occupied = set(enemies)` nhưng `blocked = set(occupied) | bomb_positions` rồi truyền `blocked` vào valid_actions → cũng sai?
  
  **QUAN TRỌNG**: Đọc kỹ lại — Tactical truyền `blocked` vào `_valid_actions` nhưng `_move_to_targets` lại cho phép đi qua target positions ngay cả khi chúng nằm trong blocked (`npos in occupied and npos not in targets`). Còn GeniusRuleAgent `blocked = set(bomb_positions)` — KHÔNG chứa enemies.

→ **GeniusRuleAgent là đối thủ mạnh nhất** vì nó KHÔNG block enemies, cho phép đi xuyên qua player khác.

→ **Fix**: `blocked` chỉ nên chứa `bomb_positions`, KHÔNG chứa `occupied`.

---

### BẪY 2: `_best_escape_action` chỉ nhìn 1 bước → dễ chui vào ngõ cụt

**Vấn đề**: Hàm `_best_escape_action` (dòng 175-191) chỉ đánh giá ô LIỀN KỀ:
```python
for a in self._valid_actions(grid, my_pos, blocked):
    npos = self._next_pos(my_pos, a)
    score += self._open_neighbors(grid, npos, blocked)
```

Nó chỉ đếm `_open_neighbors` ở depth=1. Nếu ô liền kề có 3 lối mở nhưng cả 3 lối đều là ngõ cụt depth=2, agent vẫn chọn.

**Giải pháp**: Escape nên ưu tiên dùng BFS `_move_to_safe_tile` trước (đã tìm đường xa). `_best_escape_action` chỉ nên là fallback.

→ **Đã OK trong code hiện tại** — `_best_escape_action` được gọi trước, nhưng nếu nó fail thì `_move_to_targets(safe_tiles)` được gọi. Tuy nhiên thứ tự nên ĐẢO: gọi BFS trước (tìm đường dài chắc chắn thoát), heuristic 1-bước sau.

---

### BẪY 3: Không xử lý BOMB CHAIN REACTION

**Vấn đề**: Khi đặt bom, `_can_escape_after_placing` chỉ tính blast của bom MỚI + bom CŨ. Nhưng nếu bom mới nổ → kích nổ bom cũ gần đó → blast area mở rộng bất ngờ. Agent không tính chain → nghĩ mình escape được nhưng thực tế bị chain blast giết.

**Giải pháp**: Mô phỏng chain reaction trong `_danger_tiles`.

---

### BẪY 4: `danger_now` chỉ dùng `timer <= 1` → CHẬM 1 LƯỢT

**Vấn đề**: Bom có timer=7. Trong `_danger_tiles`:
```python
if timer <= 1:
    danger_now |= blast
```
Timer giảm 1 mỗi step. Khi agent nhận obs, timer đã giảm xong. Nếu timer=1, bom sẽ nổ ở step TIẾP THEO.

Nhưng `_best_escape_action` từ chối bước vào `danger_now`. Vấn đề: nếu timer=2, bom sẽ nổ sau 2 steps. Nếu agent bước vào ô blast ở step này và ở yên đó, nó sẽ chết ở step+2. Nhưng code cho phép bước vào (vì không phải `danger_now`). Nếu escape logic quyết định "tạm bước qua danger_soon để thoát ra phía bên kia", thì OK. Nhưng nếu agent kẹt lại trong danger_soon 2 steps, nó chết.

→ Đây là trường hợp edge nhưng không phải nguyên nhân chính.

---

### BẪY 5: `enemy_dist <= 2` đặt bom vô điều kiện → TỰ SÁT

**Vấn đề** (dòng 89-91):
```python
elif enemy_dist <= 2:
    should_bomb = True
```
Khi kẻ địch ở khoảng cách Manhattan = 2 (ví dụ: chéo 1 ô), bom đặt tại `my_pos` có thể KHÔNG trúng kẻ địch (vì blast chỉ đi theo 4 hướng thẳng). Nhưng agent vẫn đặt bom → phải chạy trốn → mất vị trí chiến thuật → lãng phí bom capacity. Đặc biệt nguy hiểm khi bomb capacity chỉ có 1.

**Giải pháp**: Bỏ điều kiện này. Chỉ đặt bom khi `can_hit_enemy` hoặc `boxes_hit >= 1`.

---

### BẪY 6: `_move_to_targets` tránh `danger_soon` trên đường đi → bị tắc

**Vấn đề** (dòng 236):
```python
if npos in danger_soon: continue
```
Khi bản đồ có nhiều bom, `danger_soon` bao phủ diện tích rất lớn. BFS tránh hoàn toàn `danger_soon` → không tìm được đường đi → agent đứng yên → mất cơ hội farm.

**Giải pháp cho navigation**: Chỉ tránh `danger_soon` ở bước ĐẦU TIÊN (immediate step). Cho phép đi qua `danger_soon` ở bước 2+ (vì khi đến đó bom có thể đã nổ xong).

---

## Đề xuất thay đổi cụ thể

### 1. Fix blocked (CRITICAL)
```python
# CHỈ block bom, KHÔNG block enemies (giống GeniusRuleAgent)
blocked = set(bomb_positions)
blocked.discard(my_pos)
```

### 2. Fix escape priority order
Gọi BFS `_move_to_targets(safe_tiles)` TRƯỚC, `_best_escape_action` (heuristic 1-bước) SAU.

### 3. Bỏ `enemy_dist <= 2` bomb
Chỉ đặt bom khi: `can_hit_enemy` hoặc `boxes_hit >= 1`.

### 4. Escape mode (học từ GeniusRuleAgent)
Thêm biến `self.escape_mode`. Sau khi đặt bom, bật escape_mode = True. Khi escape_mode, luôn ưu tiên chạy trốn cho đến khi ra khỏi danger_soon.

### 5. Nới lỏng BFS navigation
Cho phép BFS đi qua `danger_soon` ở depth >= 2 (bom timer >= 2 sẽ nổ trước khi agent đến).

### 6. Tối ưu endgame (tie-breaking stats)
Cuối game (step > 400), nếu không có hộp/item/enemy gần, cứ đặt bom bừa (nếu có đường thoát) để cày `bombs_placed` stat.

## Verification Plan
- Chạy `estimate_rankings.py` với 50-100 matches
- So sánh Win Rate, Average Rank, TrueSkill score với bản hiện tại
