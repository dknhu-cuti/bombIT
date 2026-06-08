import random
from collections import deque

# Monkey-patch BomberEnv (kept for compatibility)
try:
    import inspect
    from engine.game import BomberEnv
    original_reset = BomberEnv.reset
    def patched_reset(self, *args, **kwargs):
        res = original_reset(self, *args, **kwargs)
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
            frame = inspect.currentframe().f_back
            if frame:
                filename = frame.f_code.co_filename
                if "train_ppo" in filename or "stable_baselines3" in filename or "gym" in filename:
                    return res
                return res[0]
        return res
    BomberEnv.reset = patched_reset
except Exception as e:
    pass

class Agent:
    team_id = "NhukeiAgent"
    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.escape_mode = False
        self._prev_pos = None
        self._stuck_count = 0

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        enemies = [(int(p[0]), int(p[1])) for i, p in enumerate(players) if i != self.agent_id and p[2] == 1]
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        blocked = set(bomb_positions)
        blocked.discard(my_pos)

        # Cải tiến 1: Time-Space BFS
        danger_at = self._compute_danger_timeline(grid, bombs, players)
        
        # Cải tiến 2: Dead-end Trapping
        freedom_map = self._compute_region_freedom(grid, blocked)
        
        # Compatibility cho các hàm chưa viết lại
        danger_soon = set(danger_at.keys())
        danger_now = {pos for pos, times in danger_at.items() if 1 in times}

        if self.escape_mode and 1 not in danger_at.get(my_pos, set()):
            self.escape_mode = False

        if self._prev_pos == my_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        self._prev_pos = my_pos

        valid_actions = self._valid_actions(grid, my_pos, blocked)

        enemies_alive = len(enemies)
        is_late_game = enemies_alive <= 1
        is_mid_game = enemies_alive == 2
        is_well_armed = int(bombs_left) >= 2 and int(bomb_bonus) >= 2
        should_hunt = is_late_game or is_well_armed or (is_mid_game and self._stuck_count >= 6)

        # ── PRIORITY 1: SURVIVAL ───────────────────────────
        is_in_danger = bool(danger_at.get(my_pos, set()))
        if self.escape_mode or is_in_danger:
            escape = self._timed_escape_bfs(grid, my_pos, blocked, danger_at)
            if escape is not None:
                return escape
            
            loose = [a for a in valid_actions if a != 0 and 1 not in danger_at.get(self._next_pos(my_pos, a), set())]
            if loose: return random.choice(loose)
            return 0

        # ── PRIORITY 2: ITEMS ─────────────────────────────────────────────
        item_tiles = self._item_tiles(grid, my_pos, prefer_capacity=int(bombs_left) <= 1, prefer_radius=int(bomb_bonus) <= 1, radius=7)
        if item_tiles:
            move = self._timed_move_to_target(grid, my_pos, item_tiles, blocked, danger_at)
            if move is not None: return move

        # ── PRIORITY 3: TRAP ENEMY IN DEAD-END ───────────────────────────
        trap = self._trap_enemy_action(grid, my_pos, enemies, blocked, danger_at, bomb_radius, int(bombs_left), freedom_map)
        if trap is not None: return trap

        # ── PRIORITY 4: BOMB IF CAN HIT ENEMY ────────────────────────────
        if bombs_left > 0 and my_pos not in bomb_positions:
            can_hit_enemy = self._can_bomb_hit_enemy(grid, my_pos, enemies, bomb_radius)
            if can_hit_enemy and self._can_escape_after_placing(grid, my_pos, blocked, danger_at, bomb_radius):
                self.escape_mode = True
                return 5

        # ── PRIORITY 5: HUNT ENEMY (aggressive) ──────────────────────────
        if should_hunt and enemies and int(bombs_left) > 0:
            hunt = self._hunt_enemy_action(grid, my_pos, enemies, blocked, danger_at, bomb_radius, int(bombs_left))
            if hunt is not None: return hunt

        # ── PRIORITY 6: BOMB BOXES ───────────────────────────────────────
        if bombs_left > 0 and my_pos not in bomb_positions:
            boxes_hit = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            min_boxes = 1 if is_well_armed else 2
            if boxes_hit >= min_boxes:
                if self._can_escape_after_placing(grid, my_pos, blocked, danger_at, bomb_radius):
                    self.escape_mode = True
                    return 5

        # ── PRIORITY 7: FARM BOXES ───────────────────────────────────────
        box_spots = self._box_bomb_spots(grid, my_pos, blocked)
        if box_spots:
            move = self._timed_move_to_target(grid, my_pos, box_spots, blocked, danger_at)
            if move is not None: return move

        # ── PRIORITY 8: HUNT ENEMIES (fallback) ──────────────────────────
        if enemies and int(bombs_left) > 0:
            hunt = self._hunt_enemy_action(grid, my_pos, enemies, blocked, danger_at, bomb_radius, int(bombs_left))
            if hunt is not None: return hunt

        # ── PRIORITY 9: STRATEGIC WANDER ─────────────────────────────────
        return self._strategic_wander(grid, my_pos, enemies, blocked, danger_soon, danger_now, valid_actions)

    # ── TIMED PATHFINDING & DEAD-END ANALYSIS ─────────────────────────────
    def _compute_region_freedom(self, grid, blocked, depth=4):
        freedom_map = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if self._passable(grid, x, y):
                    q = deque([((x, y), 0)])
                    seen = {(x, y)}
                    while q:
                        (cx, cy), d = q.popleft()
                        if d >= depth: continue
                        for a in [1, 2, 3, 4]:
                            nx, ny = cx + self.MOVES[a][0], cy + self.MOVES[a][1]
                            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                                if (nx, ny) not in seen:
                                    seen.add((nx, ny))
                                    q.append(((nx, ny), d + 1))
                    freedom_map[(x, y)] = len(seen)
        return freedom_map

    def _find_dead_end_exit(self, grid, start_tile, blocked, freedom_map):
        q = deque([start_tile])
        seen = {start_tile}
        while q:
            cx, cy = q.popleft()
            if freedom_map.get((cx, cy), 0) >= 6:
                return (cx, cy)
            for a in [1, 2, 3, 4]:
                nx, ny = cx + self.MOVES[a][0], cy + self.MOVES[a][1]
                if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                    if (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
        return None

    def _trap_enemy_action(self, grid, my_pos, enemies, blocked, danger_at, bomb_radius, bombs_left, freedom_map):
        if bombs_left <= 0: return None
        for ex, ey in enemies:
            enemy_pos = (ex, ey)
            if freedom_map.get(enemy_pos, 0) <= 3:
                exit_pos = self._find_dead_end_exit(grid, enemy_pos, blocked, freedom_map)
                if exit_pos:
                    if my_pos == exit_pos:
                        if self._can_escape_after_placing(grid, my_pos, blocked, danger_at, bomb_radius):
                            self.escape_mode = True
                            return 5
                    else:
                        move = self._timed_move_to_target(grid, my_pos, {exit_pos}, blocked, danger_at)
                        if move is not None:
                            return move
        return None

    def _compute_danger_timeline(self, grid, bombs, players, default_radius=2):
        bomb_map = {}
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            if timer <= 0: continue
            radius = default_radius
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            bomb_map[(bx, by)] = {'timer': timer, 'radius': radius}

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

        danger_at = {}
        for (bx, by), info in bomb_map.items():
            timer = info['timer']
            radius = info['radius']
            blast = self._blast_tiles(grid, bx, by, radius)
            for tile in blast:
                if tile not in danger_at:
                    danger_at[tile] = set()
                # Đi vào ô đúng lúc timer=1 (bom sắp nổ next step) hoặc t=timer (lúc bom đang nổ) là chết
                danger_at[tile].add(timer)
                danger_at[tile].add(timer + 1)
        return danger_at

    def _timed_escape_bfs(self, grid, start, blocked, danger_at, max_t=10):
        q = deque([(start, None, 0)])
        seen = {(start, 0)}
        
        while q:
            pos, first_action, t = q.popleft()
            
            future_danger = False
            for ft in range(t, max_t + 2):
                if ft in danger_at.get(pos, set()):
                    future_danger = True
                    break
                    
            if not future_danger and t > 0:
                return first_action
                
            if t >= max_t:
                continue
                
            nt = t + 1
            for a in [0, 1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                
                if not self._passable(grid, nx, ny): continue
                if npos in blocked and npos != start: continue
                if nt in danger_at.get(npos, set()): continue
                    
                state = (npos, nt)
                if state not in seen:
                    seen.add(state)
                    q.append((npos, a if first_action is None else first_action, nt))
        return None

    def _timed_move_to_target(self, grid, start, targets, blocked, danger_at, max_t=12):
        if not targets: return None
        q = deque([(start, None, 0)])
        seen = {(start, 0)}
        
        while q:
            pos, first_action, t = q.popleft()
            
            if pos in targets and first_action is not None:
                return first_action
                
            if t >= max_t: continue
                
            nt = t + 1
            for a in [0, 1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                
                if not self._passable(grid, nx, ny): continue
                if npos in blocked and npos not in targets: continue
                if nt in danger_at.get(npos, set()): continue
                    
                state = (npos, nt)
                if state not in seen:
                    seen.add(state)
                    q.append((npos, a if first_action is None else first_action, nt))
        return None

    def _can_escape_after_placing(self, grid, my_pos, blocked, danger_at, bomb_radius):
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        sim_danger = {k: set(v) for k, v in danger_at.items()}
        for tile in my_blast:
            if tile not in sim_danger:
                sim_danger[tile] = set()
            sim_danger[tile].add(7) # timer = 7
            sim_danger[tile].add(8)
            
        return self._timed_escape_bfs(grid, my_pos, blocked, sim_danger, max_t=9) is not None

    def _hunt_enemy_action(self, grid, my_pos, enemies, blocked, danger_at, bomb_radius, bombs_left):
        if not enemies or bombs_left <= 0: return None
        enemy_targets = self._predict_enemy_reach(grid, enemies, blocked)
        kill_spots = set()
        for ex, ey in enemy_targets:
            for dx in range(-bomb_radius, bomb_radius + 1):
                nx = ex + dx
                if self._passable(grid, nx, ey) and (nx, ey) not in blocked:
                    if self._line_clear(grid, (nx, ey), (ex, ey)): kill_spots.add((nx, ey))
            for dy in range(-bomb_radius, bomb_radius + 1):
                ny = ey + dy
                if self._passable(grid, ex, ny) and (ex, ny) not in blocked:
                    if self._line_clear(grid, (ex, ny), (ex, ey)): kill_spots.add((ex, ny))

        if my_pos in kill_spots:
            if self._can_escape_after_placing(grid, my_pos, blocked, danger_at, bomb_radius):
                self.escape_mode = True
                return 5
            kill_spots.discard(my_pos)

        if kill_spots:
            move = self._timed_move_to_target(grid, my_pos, kill_spots, blocked, danger_at)
            if move is not None: return move

        move = self._timed_move_to_target(grid, my_pos, set(enemies), blocked, danger_at)
        return move

    def _predict_enemy_reach(self, grid, enemies, blocked):
        reachable = set(enemies)
        for ex, ey in enemies:
            for a in [1, 2, 3, 4]:
                dx, dy = self.MOVES[a]
                nx, ny = ex + dx, ey + dy
                if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                    reachable.add((nx, ny))
        return reachable

    def _strategic_wander(self, grid, my_pos, enemies, blocked, danger_soon, danger_now, valid_actions):
        best_action = None
        best_score = -10**9
        for a in valid_actions:
            if a == 0: continue
            npos = self._next_pos(my_pos, a)
            if npos in danger_now: continue
            if npos in danger_soon: continue

            score = 0.0
            open_n = self._open_neighbors(grid, npos, blocked)
            score += open_n * 2.0

            for da in [1, 2, 3, 4]:
                adj = self._next_pos(npos, da)
                if adj in danger_soon: score -= 0.5

            if enemies:
                min_dist = min(abs(npos[0] - ex) + abs(npos[1] - ey) for ex, ey in enemies)
                if 1 <= min_dist <= 4:
                    score += (5 - min_dist) * 1.5
                elif min_dist > 4:
                    score -= (min_dist - 4) * 0.5

            if self._stuck_count >= 2:
                score += random.uniform(0, 3)

            if score > best_score:
                best_score = score
                best_action = a

        if best_action is not None: return best_action
        safe_moves = [a for a in valid_actions if a != 0 and self._next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0

    # ── CORE HELPERS ──────────────────────────────────────────────────────
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

    def _item_tiles(self, grid, my_pos, prefer_capacity=False, prefer_radius=False, radius=7):
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
                if val in preferred_values: preferred_tiles.add((x, y))
                if val in [3, 4]: any_items.add((x, y))
        if preferred_tiles: return preferred_tiles
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

    def _box_bomb_spots(self, grid, my_pos, blocked, radius=6):
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
