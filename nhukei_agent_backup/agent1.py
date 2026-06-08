import random
from collections import deque

class Agent:
    team_id = "NhukeiAgent"
    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.escape_mode = False

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]
        
        # BẪY 1 FIX: Không chứa occupied vào blocked. 
        # Engine cho phép player đi xuyên qua nhau, nên chỉ bomb chặn đường.
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        blocked = set(bomb_positions)
        blocked.discard(my_pos)

        # BẪY 3 FIX: Tính toán chain reaction trong danger_tiles
        danger_soon, danger_now = self._danger_tiles(grid, bombs, players)
        
        if self.escape_mode and my_pos not in danger_soon:
            self.escape_mode = False

        valid_actions = self._valid_actions(grid, my_pos, blocked)

        # ── PRIORITY 1: SURVIVAL & ESCAPE MODE ──
        if self.escape_mode or my_pos in danger_now or my_pos in danger_soon:
            escape = self._move_to_safe_tile(grid, my_pos, blocked, danger_soon, danger_now, search_depth=12)
            
            if escape is not None:
                return escape

            # Fallback heuristic (nhìn 1 bước, nhưng đếm neighbor)
            escape_fb = self._best_escape_action(grid, my_pos, blocked, danger_now, danger_soon)
            if escape_fb is not None:
                return escape_fb
                
            # Fallback 2: Tránh immediate death
            loose = [
                a for a in valid_actions
                if a != 0 and self._next_pos(my_pos, a) not in danger_now
            ]
            if loose:
                return random.choice(loose)
                
            return 0

        # ── PRIORITY 2: ITEMS ──
        item_tiles = self._item_tiles(
            grid,
            my_pos,
            prefer_capacity=int(bombs_left) <= 1,
            prefer_radius=int(bomb_bonus) <= 1,
        )
        if item_tiles:
            move = self._move_to_targets(grid, my_pos, item_tiles, blocked, danger_soon)
            if move is not None:
                return move

        # ── PRIORITY 3: BOMB PLACEMENT ──
        if bombs_left > 0 and my_pos not in bomb_positions:
            can_hit_enemy = self._can_bomb_hit_enemy(grid, my_pos, enemies, bomb_radius)
            boxes_hit = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            
            should_bomb = False
            if can_hit_enemy:
                should_bomb = True
            elif boxes_hit >= 1:
                should_bomb = True
            # BẪY 5 FIX: Đã XÓA điều kiện enemy_dist <= 2 vô tội vạ. 
            # Giờ chỉ nổ khi có line of sight (can_hit_enemy) hoặc có hộp.

            if should_bomb and self._can_escape_after_placing(grid, my_pos, blocked, danger_soon, danger_now, bomb_radius):
                self.escape_mode = True # Đặt bom xong thì bật mode chạy trốn
                return 5

        # ── PRIORITY 4: FARM BOXES ──
        box_spots = self._box_bomb_spots(grid, my_pos, blocked)
        if box_spots:
            move = self._move_to_targets(grid, my_pos, box_spots, blocked, danger_soon)
            if move is not None:
                return move

        # ── PRIORITY 5: HUNT ENEMIES ──
        if enemies and bombs_left > 0:
            enemy_spots = self._enemy_bomb_spots(grid, enemies, blocked, bomb_radius)
            if enemy_spots:
                move = self._move_to_targets(grid, my_pos, enemy_spots, blocked, danger_soon)
                if move is not None:
                    return move

        # ── PRIORITY 6: IDLE WANDER ──
        safe_moves = [a for a in valid_actions if self._next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0


    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

    def _valid_actions(self, grid, my_pos, blocked):
        actions = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                actions.append(a)
        return actions

    def _open_neighbors(self, grid, pos, blocked):
        cnt = 0
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                cnt += 1
        return cnt

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y): break
                cell = grid[x, y]
                if cell == 1: break
                tiles.add((x, y))
                if cell == 2: break
        return tiles

    def _danger_tiles(self, grid, bombs, players, default_radius=2):
        danger_soon = set()
        danger_now = set()
        
        bomb_map = {}
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            if timer <= 0: continue
            
            radius = default_radius
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
                
            bomb_map[(bx, by)] = {'timer': timer, 'radius': radius}
            
        # Mô phỏng CHAIN REACTION
        changed = True
        while changed:
            changed = False
            for (bx, by), info in bomb_map.items():
                timer = info['timer']
                radius = info['radius']
                blast = self._blast_tiles(grid, bx, by, radius)
                for (ox, oy) in blast:
                    if (ox, oy) in bomb_map:
                        if timer < bomb_map[(ox, oy)]['timer']:
                            bomb_map[(ox, oy)]['timer'] = timer
                            changed = True

        for (bx, by), info in bomb_map.items():
            timer = info['timer']
            radius = info['radius']
            blast = self._blast_tiles(grid, bx, by, radius)
            danger_soon |= blast
            if timer <= 1:
                danger_now |= blast
                
        return danger_soon, danger_now

    def _best_escape_action(self, grid, my_pos, blocked, danger_now, danger_soon):
        best_action = None
        best_score = -10**9
        for a in self._valid_actions(grid, my_pos, blocked):
            if a == 0: continue
            npos = self._next_pos(my_pos, a)
            if npos in danger_now: continue
            
            score = 0
            if npos not in danger_soon:
                score += 10
            score += self._open_neighbors(grid, npos, blocked)
            
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    def _move_to_safe_tile(self, grid, start, blocked, danger_soon, danger_now, search_depth=12):
        q = deque([(start, None, 0)])
        seen = {start}
        while q:
            pos, first_action, depth = q.popleft()
            if pos not in danger_soon and depth > 0:
                return first_action
            if depth >= search_depth: continue
            
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen or not self._passable(grid, nx, ny) or npos in blocked:
                    continue
                # Tuyệt đối không nhảy vào danger_now ở ngay next step
                if depth == 0 and npos in danger_now:
                    continue
                
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action, depth + 1))
        return None

    def _can_escape_after_placing(self, grid, my_pos, blocked, danger_soon, danger_now, bomb_radius):
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        combined_soon = set(danger_soon) | my_blast
        combined_now = set(danger_now)
        
        # Nếu đặt bom ở ô đã sẵn nằm trong danger_now, bom mới sẽ nổ theo dây chuyền NGAY LẬP TỨC
        if my_pos in danger_now:
            combined_now |= my_blast
            
        return self._move_to_safe_tile(grid, my_pos, blocked, combined_soon, combined_now, search_depth=8) is not None

    def _move_to_targets(self, grid, start, targets, blocked, danger_soon):
        if not targets:
            return None
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in targets and first_action is not None:
                return first_action
                
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen: continue
                if not self._passable(grid, nx, ny): continue
                if npos in blocked and npos not in targets: continue
                if npos in danger_soon: continue
                
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None

    def _item_tiles(self, grid, my_pos, prefer_capacity=False, prefer_radius=False, radius=5):
        preferred_values = set()
        if prefer_radius: preferred_values.add(3)
        if prefer_capacity: preferred_values.add(4)

        mx, my = my_pos
        x_min = max(0, mx - radius)
        x_max = min(grid.shape[0], mx + radius + 1)
        y_min = max(0, my - radius)
        y_max = min(grid.shape[1], my + radius + 1)

        preferred_tiles = set()
        any_items = set()
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                val = grid[x, y]
                if val in preferred_values:
                    preferred_tiles.add((x, y))
                if val in [3, 4]:
                    any_items.add((x, y))

        if preferred_tiles:
            return preferred_tiles

        return any_items

    def _line_clear(self, grid, a, b):
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if grid[ax, y] in [1, 2]: return False
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if grid[x, ay] in [1, 2]: return False
            return True
        return False

    def _can_bomb_hit_enemy(self, grid, my_pos, enemies, radius):
        mx, my = my_pos
        for ex, ey in enemies:
            if mx == ex and abs(ey - my) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
            if my == ey and abs(ex - mx) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
        return False

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x, y in self._blast_tiles(grid, pos[0], pos[1], radius) if grid[x, y] == 2)

    def _box_bomb_spots(self, grid, my_pos, blocked, radius=5):
        spots = set()
        mx, my = my_pos
        x_min = max(0, mx - radius)
        x_max = min(grid.shape[0], mx + radius + 1)
        y_min = max(0, my - radius)
        y_max = min(grid.shape[1], my + radius + 1)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if grid[x, y] == 2:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                            spots.add((nx, ny))
        return spots

    def _enemy_bomb_spots(self, grid, enemies, blocked, bomb_radius):
        spots = set()
        enemy_positions = set(enemies)
        for ex, ey in enemies:
            for dx in range(-bomb_radius, bomb_radius + 1):
                nx = ex + dx
                if self._passable(grid, nx, ey) and (nx, ey) not in blocked and (nx, ey) not in enemy_positions:
                    if self._line_clear(grid, (nx, ey), (ex, ey)):
                        spots.add((nx, ey))
            for dy in range(-bomb_radius, bomb_radius + 1):
                ny = ey + dy
                if self._passable(grid, ex, ny) and (ex, ny) not in blocked and (ex, ny) not in enemy_positions:
                    if self._line_clear(grid, (ex, ny), (ex, ey)):
                        spots.add((ex, ny))
        return spots
