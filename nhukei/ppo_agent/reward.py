"""
Hàm Thưởng (Reward Function) cho PPO Agent trong môi trường 4 người chơi BombIt.
Tái sử dụng và mở rộng từ agent/dqn_agent/reward.py.

Thay đổi so với bản DQN mẫu:
- win: 2.0 -> 15.0 (thắng 4 người khó hơn 1v1 rất nhiều)
- enemy_death: 1.0 -> 10.0 (giết địch là cốt lõi, cần ưu tiên cao)
- agent_death: -2.0 -> -10.0 (đối xứng với reward giết địch)
- item_collection: 0.1 -> 1.0 (đủ hấp dẫn để bot đi nhặt đồ)
- plant_near_box: 0.05 -> 0.5 (đủ hấp dẫn để bot phá hộp)
"""

import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine import Map

_DEFAULT_BOMB_TIMER = 7
_DEFAULT_BOMB_OWNER = 0

REWARD_DICT = {
    "win":              15.0,   # Trở thành người sống sót cuối cùng
    "enemy_death":      10.0,   # Giết được 1 địch
    "agent_death":     -10.0,   # Bị chết (bom, tự sát)
    "item_collection":   1.0,   # Ăn được vật phẩm tăng sức mạnh
    "plant_near_box":    0.5,   # Đặt bom cạnh hộp gỗ (khuyến khích farm)
    "danger_evasion":    0.12,  # Thoát ra khỏi vùng bom thành công
    "danger_enter":     -0.06,  # Bước vào vùng bom
    "own_blast_loiter": -0.04,  # Đứng trong vùng nổ của bom mình đặt
    "approach_enemy":    0.02,  # Tiếp cận địch (khuyến khích tấn công)
    "standing_still":   -0.01,  # Đứng yên (phạt để tránh AFK)
    "time_penalty":     -0.005, # Phạt theo thời gian (khuyến khích đánh nhanh)
}


def _parse_bomb_row(b):
    arr = np.asarray(b, dtype=np.float64).ravel()
    if arr.size < 2:
        return None
    bx, by = int(arr[0]), int(arr[1])
    timer    = int(arr[2]) if arr.size > 2 else _DEFAULT_BOMB_TIMER
    owner_id = int(arr[3]) if arr.size > 3 else _DEFAULT_BOMB_OWNER
    return bx, by, timer, owner_id


def _bomb_radius_from_obs(players, owner_id):
    return 1 + int(players[int(owner_id)][4])


def _explosion_tiles_for_bomb(grid, bx, by, radius):
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == Map.WALL:
                break
            tiles.add((tx, ty))
            if cell == Map.BOX:
                break
    return tiles


def _blast_status_at(obs, x, y):
    bombs = obs["bombs"]
    if bombs is None:
        return False, None
    arr = np.asarray(bombs)
    if arr.size == 0:
        return False, None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ix, iy   = int(x), int(y)
    players  = obs["players"]
    grid     = obs["map"]
    in_blast = False
    min_timer = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        radius = _bomb_radius_from_obs(players, owner_id)
        tiles  = _explosion_tiles_for_bomb(grid, bx, by, radius)
        if (ix, iy) in tiles:
            in_blast = True
            t = int(timer)
            min_timer = t if min_timer is None else min(min_timer, t)
    return in_blast, min_timer


def _any_bombs(obs):
    b = obs["bombs"]
    if b is None:
        return False
    return np.asarray(b).size > 0


def _enemy_alive_count(players, agent_id):
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return sum(1 for pid in range(arr.shape[0]) if pid != agent_id and int(arr[pid][2]) == 1)


def _manhattan_to_nearest_alive_enemy(players, agent_id, x, y):
    best = None
    ix, iy = int(x), int(y)
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for pid in range(arr.shape[0]):
        if pid == agent_id or int(arr[pid][2]) != 1:
            continue
        d = abs(ix - int(arr[pid][0])) + abs(iy - int(arr[pid][1]))
        best = d if best is None else min(best, d)
    return best


def _min_own_blast_timer_at(obs, agent_id, x, y):
    bombs = obs["bombs"]
    if bombs is None:
        return None
    arr = np.asarray(bombs)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    players = obs["players"]
    grid    = obs["map"]
    ix, iy  = int(x), int(y)
    aid     = int(agent_id)
    best    = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        if int(owner_id) != aid:
            continue
        radius = _bomb_radius_from_obs(players, owner_id)
        tiles  = _explosion_tiles_for_bomb(grid, bx, by, radius)
        if (ix, iy) in tiles:
            t = int(timer)
            best = t if best is None else min(best, t)
    return best


def compute_reward(prev_obs, curr_obs, agent_id):
    """Tính reward từ cặp (prev_obs, curr_obs) cho agent_id."""
    if prev_obs is None:
        return 0.0

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]

    prev_alive = int(prev_players[agent_id][2])
    curr_alive  = int(curr_players[agent_id][2])

    reward = 0.0

    # 1. CHẾT / THẮNG
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARD_DICT["agent_death"])

    prev_enemies = _enemy_alive_count(prev_players, agent_id)
    curr_enemies  = _enemy_alive_count(curr_players, agent_id)

    if curr_enemies < prev_enemies:
        reward += REWARD_DICT["enemy_death"] * (prev_enemies - curr_enemies)
    if curr_enemies == 0 and prev_enemies > 0:
        reward += REWARD_DICT["win"]

    # 2. DI CHUYỂN & TIME PENALTY
    prev_x, prev_y = prev_players[agent_id][0], prev_players[agent_id][1]
    curr_x, curr_y  = curr_players[agent_id][0], curr_players[agent_id][1]

    if prev_x == curr_x and prev_y == curr_y:
        reward += REWARD_DICT["standing_still"]
    else:
        reward -= REWARD_DICT["standing_still"]  # Thưởng nhỏ khi di chuyển
    reward += REWARD_DICT["time_penalty"]

    # 3. NÉ BOM / BƯỚC VÀO BOM
    if _any_bombs(prev_obs) or _any_bombs(curr_obs):
        prev_in_blast, prev_timer = _blast_status_at(prev_obs, prev_x, prev_y)
        curr_in_blast, _          = _blast_status_at(curr_obs, curr_x, curr_y)
        if prev_in_blast and not curr_in_blast:
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 3) else 1.0
            reward += REWARD_DICT["danger_evasion"] * urgency
        elif not prev_in_blast and curr_in_blast and (prev_x != curr_x or prev_y != curr_y):
            reward += REWARD_DICT["danger_enter"]

    mt_own = _min_own_blast_timer_at(curr_obs, agent_id, curr_x, curr_y)
    if curr_alive == 1 and mt_own is not None:
        urgency = max(1, 8 - int(mt_own))
        reward += REWARD_DICT["own_blast_loiter"] * float(urgency)

    # 4. TIẾP CẬN ĐỊCH
    if curr_alive == 1 and prev_enemies > 0 and curr_enemies > 0:
        prev_d = _manhattan_to_nearest_alive_enemy(prev_players, agent_id, prev_x, prev_y)
        curr_d  = _manhattan_to_nearest_alive_enemy(curr_players, agent_id, curr_x, curr_y)
        if prev_d is not None and curr_d is not None:
            reward += REWARD_DICT["approach_enemy"] * (prev_d - curr_d)

    # 5. NHẶT ITEM
    stepped_on = prev_obs["map"][int(curr_x), int(curr_y)]
    if stepped_on in [Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
        reward += REWARD_DICT["item_collection"]
    else:
        prev_radius = int(prev_players[agent_id][4])
        curr_radius  = int(curr_players[agent_id][4])
        if curr_radius > prev_radius:
            reward += REWARD_DICT["item_collection"]

    # 6. ĐẶT BOM CẠNH HỘP
    prev_bombs_left = int(prev_players[agent_id][3])
    curr_bombs_left  = int(curr_players[agent_id][3])
    if curr_bombs_left < prev_bombs_left:
        grid = prev_obs["map"]
        H, W = grid.shape
        cx, cy = int(curr_x), int(curr_y)
        adjacent = [
            grid[max(0, cx - 1), cy],
            grid[min(H - 1, cx + 1), cy],
            grid[cx, max(0, cy - 1)],
            grid[cx, min(W - 1, cy + 1)],
        ]
        if Map.BOX in adjacent:
            reward += REWARD_DICT["plant_near_box"]

    return float(reward)
