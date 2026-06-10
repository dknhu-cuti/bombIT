"""
Agent v2 - Intelligent Scoring + Deep Lookahead
Focus: Multi-criteria move evaluation, enemy prediction, 3-5 level lookahead
"""
import random
import time
from collections import deque

class Agent:
    team_id = "Nhukei_Grandmaster_v2"
    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    
    # Scoring weights - có thể tune
    W_SURVIVAL = 3.0      # Tránh chết là ưu tiên số 1
    W_OFFENSE = 1.8       # Tấn công enemies
    W_POSITION = 1.2      # Vị trí tốt cho tương lai
    W_ECONOMY = 0.8       # Kinh tế bom/items
    W_CONTROL = 0.6       # Kiểm soát không gian

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.escape_mode = False
        self._prev_pos = None
        self._stuck_count = 0
        self._enemy_prediction_cache = {}
        self._time_budget_start = None

    def act(self, obs):
        self._time_budget_start = time.time()
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

        # ── PRECOMPUTE STATIC ANALYSIS ───────────────────────────────────────
        danger_at = self._compute_danger_timeline(grid, bombs, players)
        freedom_map = self._compute_region_freedom(grid, blocked)
        
        # Update tracking
        if self._prev_pos == my_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        self._prev_pos = my_pos

        # ── PHASE 1: SURVIVAL CHECK (If in immediate danger, escape only) ─────
        is_in_danger = bool(danger_at.get(my_pos, set()))
        if is_in_danger:
            escape = self._timed_escape_bfs(grid, my_pos, blocked, danger_at)
            if escape is not None:
                self.escape_mode = True
                return escape
            loose = [a for a in self._valid_actions(grid, my_pos, blocked) 
                    if a != 0 and 1 not in danger_at.get(self._next_pos(my_pos, a), set())]
            if loose:
                return random.choice(loose)
            return 0

        # ── PHASE 2: INTELLIGENT ACTION EVALUATION ──────────────────────────────
        # Sử dụng thời gian để đánh giá sâu vào các moves
        valid_actions = self._valid_actions(grid, my_pos, blocked)
        
        # Tính lookahead depth dựa trên time budget
        max_lookahead_depth = self._estimate_lookahead_depth(enemies, int(bombs_left), int(bomb_bonus))
        
        best_action = None
        best_score = -float('inf')
        
        for action in valid_actions:
            # Evaluate move với deep lookahead
            score = self._evaluate_move_with_lookahead(
                grid, my_pos, action, enemies, int(bombs_left), bomb_radius,
                blocked, danger_at, freedom_map, max_lookahead_depth
            )
            
            if score > best_score:
                best_score = score
                best_action = action
                
            # Check time budget
            if self._elapsed_time() > 0.08:  # 80ms budget for evaluation
                break
        
        if best_action is not None:
            return best_action
        
        # ── FALLBACK: Safe wander ───────────────────────────────────────────
        return self._safe_wander(grid, my_pos, enemies, blocked, danger_at, valid_actions)

    # ── CORE EVALUATION SYSTEM ───────────────────────────────────────────────
    
    def _evaluate_move_with_lookahead(self, grid, pos, action, enemies, bombs_left, bomb_radius,
                                      blocked, danger_at, freedom_map, depth=3):
        """
        Multi-criteria evaluation với lookahead sâu
        Tính toán: survival + offense + positioning + economy
        """
        next_pos = self._next_pos(pos, action)
        
        # ▼ Tier 1: Immediate Danger Check
        if next_pos in danger_at.get(1, set()):
            return -1000  # Chết ngay lập tức
        
        # ▼ Tier 2: Base Move Scores
        score = 0
        
        # 2.1 - Survival (30%)
        survival = self._evaluate_survival(grid, pos, next_pos, action, bombs_left, bomb_radius,
                                           blocked, danger_at, depth)
        score += self.W_SURVIVAL * survival
        
        # 2.2 - Offensive Value (25%)
        if action == 5:  # Place bomb
            offense = self._evaluate_bomb_placement(grid, next_pos, enemies, bombs_left,
                                                   bomb_radius, blocked, danger_at, depth)
        else:
            offense = self._evaluate_positioning_for_offense(grid, next_pos, enemies, bombs_left,
                                                             bomb_radius, blocked, depth)
        score += self.W_OFFENSE * offense
        
        # 2.3 - Positioning (20%)
        positioning = self._evaluate_positioning(grid, next_pos, enemies, freedom_map, blocked)
        score += self.W_POSITION * positioning
        
        # 2.4 - Economy (15%)
        economy = self._evaluate_economy(grid, next_pos, enemies, bombs_left, bomb_bonus=bomb_radius-1)
        score += self.W_ECONOMY * economy
        
        # 2.5 - Space Control (10%)
        control = self._evaluate_control(grid, next_pos, blocked, freedom_map)
        score += self.W_CONTROL * control
        
        return score

    def _evaluate_survival(self, grid, pos, next_pos, action, bombs_left, bomb_radius,
                          blocked, danger_at, depth):
        """
        Đánh giá khả năng sống sót (0-1)
        - Kiểm tra escape route 3-5 bước phía trước
        """
        # Nếu đặt bom → check xem có đường sống không
        if action == 5:
            my_blast = self._blast_tiles(grid, next_pos[0], next_pos[1], bomb_radius)
            sim_danger = {k: set(v) for k, v in danger_at.items()}
            for tile in my_blast:
                if tile not in sim_danger:
                    sim_danger[tile] = set()
                sim_danger[tile].update({7, 8})
            
            sim_blocked = set(blocked)
            sim_blocked.add(next_pos)
            
            # Lookahead BFS từ next_pos
            can_escape = self._lookahead_escape(grid, next_pos, sim_blocked, sim_danger, depth)
            return 1.0 if can_escape else -2.0
        
        # Nếu move thường → kiểm tra danger timeline
        future_danger_count = sum(1 for t in range(1, depth + 1) if t in danger_at.get(next_pos, set()))
        if future_danger_count > 0:
            return -1.0 - future_danger_count * 0.3
        
        return 0.5 + random.uniform(-0.1, 0.1)

    def _evaluate_bomb_placement(self, grid, bomb_pos, enemies, bombs_left, bomb_radius,
                                blocked, danger_at, depth):
        """
        Đánh giá value của việc đặt bom
        - Hit enemy? (gây sát thương trực tiếp)
        - Clear path? (mở đường thoát)
        - Future control? (kiểm soát không gian)
        """
        if bombs_left <= 0:
            return -10.0
        
        my_blast = self._blast_tiles(grid, bomb_pos[0], bomb_pos[1], bomb_radius)
        score = 0.0
        
        # Hit detection (trực tiếp + dự báo)
        enemy_hit_count = 0
        for ex, ey in enemies:
            enemy_pos = (ex, ey)
            if enemy_pos in my_blast:
                enemy_hit_count += 1
            # Predicted position
            elif self._line_clear(grid, bomb_pos, enemy_pos):
                pred_enemy_pos = self._predict_enemy_move(grid, enemy_pos, bomb_pos, blocked, depth)
                if pred_enemy_pos in my_blast:
                    enemy_hit_count += 0.5
        
        score += enemy_hit_count * 5.0
        
        # Box clearing
        box_count = sum(1 for x, y in my_blast if grid[x, y] == 2)
        score += box_count * 1.5
        
        # Escape ability after placing
        can_escape = self._can_escape_after_placing(grid, bomb_pos, blocked, danger_at, bomb_radius, depth)
        if not can_escape:
            score -= 3.0
        
        return score

    def _evaluate_positioning_for_offense(self, grid, next_pos, enemies, bombs_left, bomb_radius,
                                         blocked, depth):
        """
        Di chuyển để có vị trí tấn công tốt hơn
        """
        if bombs_left <= 0:
            return 0.0
        
        score = 0.0
        
        # Khoảng cách đến kill spots
        kill_spots = self._get_kill_spots(grid, enemies, bomb_radius, blocked)
        if kill_spots:
            min_dist_to_kill = min(abs(next_pos[0] - kx) + abs(next_pos[1] - ky) 
                                  for kx, ky in kill_spots)
            score += max(0, 5.0 - min_dist_to_kill * 0.5)
        
        # Khoảng cách đến enemies
        if enemies:
            min_dist_to_enemy = min(abs(next_pos[0] - ex) + abs(next_pos[1] - ey) 
                                   for ex, ey in enemies)
            if 1 <= min_dist_to_enemy <= 4:
                score += (5 - min_dist_to_enemy) * 1.0
        
        return score

    def _evaluate_positioning(self, grid, pos, enemies, freedom_map, blocked):
        """
        Đánh giá vị trí tốt cho tương lai
        - Không bị nhốt trong dead-end
        - Có nhiều escape route
        - Gần items/kill spots
        """
        freedom = freedom_map.get(pos, 0)
        score = 0.0
        
        # Freedom is good
        if freedom <= 3:
            score -= 1.0
        elif freedom >= 8:
            score += 0.5
        
        # Adjacent openness
        open_neighbors = self._open_neighbors(grid, pos, blocked)
        score += open_neighbors * 0.2
        
        return score

    def _evaluate_economy(self, grid, pos, enemies, bombs_left, bomb_bonus):
        """
        Đánh giá tài chính - nên lấy item hay đặt bom?
        """
        # Item locations
        mx, my = pos
        score = 0.0
        
        # Gần item → prioritize
        for x in range(max(0, mx - 5), min(grid.shape[0], mx + 6)):
            for y in range(max(0, my - 5), min(grid.shape[1], my + 6)):
                if grid[x, y] in [3, 4]:  # Bomb item or radius item
                    dist = abs(x - mx) + abs(y - my)
                    if grid[x, y] == 3 and bombs_left <= 1:  # Need bomb capacity
                        score += (6 - dist) * 0.3
                    elif grid[x, y] == 4 and bomb_bonus <= 1:  # Need radius
                        score += (6 - dist) * 0.2
        
        return score

    def _evaluate_control(self, grid, pos, blocked, freedom_map):
        """
        Đánh giá kiểm soát không gian
        """
        neighbors_freedom = []
        for a in [1, 2, 3, 4]:
            npos = self._next_pos(pos, a)
            neighbors_freedom.append(freedom_map.get(npos, 0))
        
        avg_neighbor_freedom = sum(neighbors_freedom) / len(neighbors_freedom) if neighbors_freedom else 0
        return min(1.0, avg_neighbor_freedom / 6.0)

    # ── ENEMY PREDICTION & LOOKAHEAD ────────────────────────────────────────
    
    def _predict_enemy_move(self, grid, enemy_pos, threat_pos, blocked, depth=2):
        """
        Dự báo nước đi tiếp theo của enemy (simple: closest threat)
        """
        ex, ey = enemy_pos
        # BFS closest to threat
        q = deque([(enemy_pos, 0)])
        seen = {enemy_pos}
        best_pos = enemy_pos
        
        while q:
            pos, d = q.popleft()
            if d > depth:
                break
            
            if abs(pos[0] - threat_pos[0]) + abs(pos[1] - threat_pos[1]) < \
               abs(best_pos[0] - threat_pos[0]) + abs(best_pos[1] - threat_pos[1]):
                best_pos = pos
            
            for a in [1, 2, 3, 4]:
                npos = self._next_pos(pos, a)
                if self._passable(grid, npos[0], npos[1]) and npos not in blocked and npos not in seen:
                    seen.add(npos)
                    q.append((npos, d + 1))
        
        return best_pos

    def _lookahead_escape(self, grid, start, blocked, danger_at, depth):
        """
        Kiểm tra xem có đường thoát sau depth bước không
        """
        q = deque([(start, 0)])
        seen = {(start, 0)}
        
        while q:
            pos, t = q.popleft()
            future_danger = any(ft in danger_at.get(pos, set()) for ft in range(t, depth + 2))
            
            if not future_danger and t > 0:
                return True
            
            if t >= depth:
                continue
            
            for a in [1, 2, 3, 4, 0]:
                npos = self._next_pos(pos, a)
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                
                if npos in blocked and not (t == 0 and npos == start):
                    continue
                
                if (t + 1) in danger_at.get(npos, set()):
                    continue
                
                state = (npos, t + 1)
                if state not in seen:
                    seen.add(state)
                    q.append((npos, t + 1))
        
        return False

    def _estimate_lookahead_depth(self, enemies, bombs_left, bomb_bonus):
        """
        Tính lookahead depth dựa trên game state
        """
        if len(enemies) <= 1:
            return 5  # Late game → sâu hơn
        elif bombs_left >= 2 and bomb_bonus >= 2:
            return 4  # Well armed
        else:
            return 3  # Early game

    # ── EXISTING HELPERS (PRESERVED & OPTIMIZED) ──────────────────────────
    
    def _time_elapsed(self):
        """Milliseconds elapsed since act() start"""
        return (time.time() - self._time_budget_start) * 1000 if self._time_budget_start else 0
    
    def _elapsed_time(self):
        return time.time() - self._time_budget_start if self._time_budget_start else 0

    def _can_escape_after_placing(self, grid, my_pos, blocked, danger_at, bomb_radius, depth):
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        sim_danger = {k: set(v) for k, v in danger_at.items()}
        for tile in my_blast:
            if tile not in sim_danger:
                sim_danger[tile] = set()
            sim_danger[tile].update({7, 8})
        
        sim_blocked = set(blocked)
        sim_blocked.add(my_pos)
        return self._lookahead_escape(grid, my_pos, sim_blocked, sim_danger, depth)

    def _timed_escape_bfs(self, grid, start, blocked, danger_at, max_t=10):
        q = deque([(start, None, 0)])
        seen = {(start, 0)}
        
        while q:
            pos, first_action, t = q.popleft()
            future_danger = any(ft in danger_at.get(pos, set()) for ft in range(t, max_t + 2))
            if not future_danger and t > 0:
                return first_action
            
            if t >= max_t:
                continue
            
            nt = t + 1
            for a in [1, 2, 3, 4, 0]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if not self._passable(grid, nx, ny):
                    continue
                
                if npos in blocked:
                    if t == 0 and npos == start and a != 0:
                        pass
                    else:
                        continue
                
                if nt in danger_at.get(npos, set()):
                    continue
                
                state = (npos, nt)
                if state not in seen:
                    seen.add(state)
                    q.append((npos, a if first_action is None else first_action, nt))
        
        return None

    def _compute_region_freedom(self, grid, blocked, depth=4):
        freedom_map = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if self._passable(grid, x, y):
                    q = deque([((x, y), 0)])
                    seen = {(x, y)}
                    while q:
                        (cx, cy), d = q.popleft()
                        if d >= depth:
                            continue
                        for a in [1, 2, 3, 4]:
                            nx, ny = cx + self.MOVES[a][0], cy + self.MOVES[a][1]
                            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                                if (nx, ny) not in seen:
                                    seen.add((nx, ny))
                                    q.append(((nx, ny), d + 1))
                    freedom_map[(x, y)] = len(seen)
        return freedom_map

    def _compute_danger_timeline(self, grid, bombs, players, default_radius=2):
        bomb_map = {}
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            if timer <= 0:
                continue
            radius = default_radius
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            bomb_map[(bx, by)] = {'timer': timer, 'radius': radius}

        changed = True
        while changed:
            changed = False
            for (bx, by), info in bomb_map.items():
                timer, radius = info['timer'], info['radius']
                blast = self._blast_tiles(grid, bx, by, radius)
                for (ox, oy) in blast:
                    if (ox, oy) in bomb_map and timer < bomb_map[(ox, oy)]['timer']:
                        bomb_map[(ox, oy)]['timer'] = timer
                        changed = True

        danger_at = {}
        for (bx, by), info in bomb_map.items():
            timer, radius = info['timer'], info['radius']
            for tile in self._blast_tiles(grid, bx, by, radius):
                if tile not in danger_at:
                    danger_at[tile] = set()
                danger_at[tile].update({timer, timer + 1})
        return danger_at

    def _get_kill_spots(self, grid, enemies, bomb_radius, blocked):
        kill_spots = set()
        for ex, ey in enemies:
            for dx in range(-bomb_radius, bomb_radius + 1):
                nx = ex + dx
                if self._passable(grid, nx, ey) and (nx, ey) not in blocked:
                    if self._line_clear(grid, (nx, ey), (ex, ey)):
                        kill_spots.add((nx, ey))
            for dy in range(-bomb_radius, bomb_radius + 1):
                ny = ey + dy
                if self._passable(grid, ex, ny) and (ex, ny) not in blocked:
                    if self._line_clear(grid, (ex, ny), (ex, ey)):
                        kill_spots.add((ex, ny))
        return kill_spots

    def _safe_wander(self, grid, my_pos, enemies, blocked, danger_at, valid_actions):
        """Fallback: Đi lại an toàn nếu không có strategy nào"""
        danger_soon = set(danger_at.keys())
        best_action = 0
        best_score = -float('inf')
        
        for a in valid_actions:
            if a == 0:
                continue
            npos = self._next_pos(my_pos, a)
            if npos in danger_soon:
                continue
            
            # Score: open neighbors + distance to enemies
            score = self._open_neighbors(grid, npos, blocked) * 1.0
            if enemies:
                min_dist = min(abs(npos[0] - ex) + abs(npos[1] - ey) for ex, ey in enemies)
                if 2 <= min_dist <= 5:
                    score += (6 - min_dist) * 0.5
            
            if score > best_score:
                best_score = score
                best_action = a
        
        return best_action

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES.get(action, (0, 0))
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
        return sum(1 for a in [1, 2, 3, 4] 
                  if self._passable(grid, *self._next_pos(pos, a)) 
                  and self._next_pos(pos, a) not in blocked)

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = grid[x, y]
                if cell == 1:
                    break
                tiles.add((x, y))
                if cell == 2:
                    break
        return tiles

    def _line_clear(self, grid, a, b):
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if grid[ax, y] in [1, 2]:
                    return False
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if grid[x, ay] in [1, 2]:
                    return False
            return True
        return False
