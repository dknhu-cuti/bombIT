"""
Agent v3 - HYBRID (Fast Heuristics + Deep Lookahead)
- Nhanh chóng decide nước cơ bản (escape, hit nearby enemy)
- Sâu suy nghĩ cho nước chiến thuật (farm, hunt, trap)
- Sử dụng time budget một cách cân bằng
"""
import random
import time
from collections import deque


class Agent:
    team_id = "Nhukei_Grandmaster_Hybrid"
    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    # Scoring weights
    W_SURVIVAL = 3.0
    W_OFFENSE = 1.8
    W_POSITION = 1.2
    W_ECONOMY = 0.8
    W_CONTROL = 0.6

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.escape_mode = False
        self._prev_pos = None
        self._stuck_count = 0
        self._time_start = None

    def act(self, obs):
        self._time_start = time.time()
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        enemies = [(int(p[0]), int(p[1])) for i, p in enumerate(players) 
                  if i != self.agent_id and p[2] == 1]
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        blocked = set(bomb_positions)
        blocked.discard(my_pos)

        # Precompute
        danger_at = self._compute_danger_timeline(grid, bombs, players)
        freedom_map = self._compute_region_freedom(grid, blocked)

        # Update tracking
        if self._prev_pos == my_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        self._prev_pos = my_pos

        # ═══ HYBRID DECISION TREE ═══════════════════════════════════════════

        # PHASE 1: SURVIVAL (IMMEDIATE DANGER)
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

        # PHASE 2A: QUICK HEURISTIC CHECKS (< 5ms)
        # ─────────────────────────────────────────
        
        # Q1: Can I bomb-hit an enemy right now with high confidence?
        if int(bombs_left) > 0 and my_pos not in bomb_positions:
            quick_bomb_move = self._quick_bomb_check(grid, my_pos, enemies, bomb_radius, 
                                                     blocked, danger_at)
            if quick_bomb_move is not None:
                return quick_bomb_move
        
        # Q2: Should I eat an item NOW? (Close + valuable)
        quick_item_move = self._quick_item_grab(grid, my_pos, blocked, danger_at, 
                                               int(bombs_left), int(bomb_bonus))
        if quick_item_move is not None and self._elapsed_ms() < 10:
            return quick_item_move

        # PHASE 2B: DEEP EVALUATION (30-80ms)
        # ──────────────────────────────────
        valid_actions = self._valid_actions(grid, my_pos, blocked)
        
        best_action = None
        best_score = -float('inf')
        
        for action in valid_actions:
            score = self._evaluate_move_smart(
                grid, my_pos, action, enemies, int(bombs_left), bomb_radius,
                blocked, danger_at, freedom_map
            )
            
            if score > best_score:
                best_score = score
                best_action = action
            
            # Time budget
            if self._elapsed_ms() > 85:
                break
        
        if best_action is not None:
            return best_action

        # FALLBACK
        return self._safe_wander(grid, my_pos, enemies, blocked, danger_at, valid_actions)

    # ═══ PHASE 2A: QUICK HEURISTIC CHECKS ═══════════════════════════════════

    def _quick_bomb_check(self, grid, my_pos, enemies, bomb_radius, blocked, danger_at):
        """
        Ultra-fast bomb check: if enemy in blast radius + direct line of sight
        """
        for ex, ey in enemies:
            if abs(my_pos[0] - ex) <= bomb_radius and abs(my_pos[1] - ey) <= bomb_radius:
                if self._line_clear(grid, my_pos, (ex, ey)):
                    # Check if can escape
                    blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
                    has_escape = any(
                        1 not in danger_at.get(self._next_pos(my_pos, a), set()) and 
                        a != 0
                        for a in [1, 2, 3, 4]
                    )
                    if has_escape:
                        return 5  # Place bomb
        return None

    def _quick_item_grab(self, grid, my_pos, blocked, danger_at, bombs_left, bomb_bonus):
        """
        Grab item if it's RIGHT HERE (distance ≤ 2) and useful
        """
        mx, my = my_pos
        for x in range(mx - 2, mx + 3):
            for y in range(my - 2, my + 3):
                if not self._in_bounds(grid, x, y):
                    continue
                
                cell = grid[x, y]
                
                # Bomb item: grab if low on bombs
                if cell == 3 and bombs_left <= 1:
                    for a in [1, 2, 3, 4]:
                        npos = self._next_pos(my_pos, a)
                        if npos == (x, y) and npos not in blocked:
                            if 1 not in danger_at.get(npos, set()):
                                return a
                
                # Radius item: grab if low on radius
                if cell == 4 and bomb_bonus <= 1:
                    for a in [1, 2, 3, 4]:
                        npos = self._next_pos(my_pos, a)
                        if npos == (x, y) and npos not in blocked:
                            if 1 not in danger_at.get(npos, set()):
                                return a
        
        return None

    # ═══ PHASE 2B: SMART EVALUATION ════════════════════════════════════════

    def _evaluate_move_smart(self, grid, pos, action, enemies, bombs_left, bomb_radius,
                            blocked, danger_at, freedom_map):
        """
        Balanced evaluation: combine heuristics + lookahead (depth 2-3)
        """
        next_pos = self._next_pos(pos, action)

        # Immediate danger check
        if next_pos in danger_at.get(1, set()):
            return -1000

        score = 0.0

        # ▼ Survival (30%)
        survival = self._eval_survival_smart(grid, pos, next_pos, action, bombs_left, 
                                            bomb_radius, blocked, danger_at)
        score += self.W_SURVIVAL * survival

        # ▼ Offense (25%)
        if action == 5:  # Bomb
            offense = self._eval_bomb_smart(grid, next_pos, enemies, bombs_left, 
                                           bomb_radius, blocked)
        else:
            offense = self._eval_position_for_offense(grid, next_pos, enemies, bombs_left, bomb_radius)
        score += self.W_OFFENSE * offense

        # ▼ Positioning (20%)
        positioning = self._eval_positioning(grid, next_pos, enemies, freedom_map, blocked)
        score += self.W_POSITION * positioning

        # ▼ Economy (15%)
        economy = self._eval_economy(grid, next_pos, enemies, bombs_left, bomb_radius - 1)
        score += self.W_ECONOMY * economy

        # ▼ Control (10%)
        control = self._eval_control(grid, next_pos, blocked, freedom_map)
        score += self.W_CONTROL * control

        return score

    def _eval_survival_smart(self, grid, pos, next_pos, action, bombs_left, bomb_radius,
                            blocked, danger_at):
        """
        Smart survival: check 2-3 steps ahead, not just immediate
        """
        # Lookahead depth: higher if bombs_left high
        lookahead_depth = 3 if bombs_left >= 2 else 2

        # Bomb placement?
        if action == 5:
            blast = self._blast_tiles(grid, next_pos[0], next_pos[1], bomb_radius)
            sim_danger = {k: set(v) for k, v in danger_at.items()}
            for tile in blast:
                if tile not in sim_danger:
                    sim_danger[tile] = set()
                sim_danger[tile].update({7, 8})

            sim_blocked = set(blocked)
            sim_blocked.add(next_pos)

            can_escape = self._check_lookahead_safety(grid, next_pos, sim_blocked, 
                                                     sim_danger, lookahead_depth)
            return 1.0 if can_escape else -2.0

        # Normal move: check danger ahead
        future_danger = sum(1 for t in range(1, lookahead_depth + 1) 
                           if t in danger_at.get(next_pos, set()))
        return 0.5 - future_danger * 0.3

    def _eval_bomb_smart(self, grid, bomb_pos, enemies, bombs_left, bomb_radius, blocked):
        """
        Bomb evaluation: direct hit + predicted position
        """
        if bombs_left <= 0:
            return -10.0

        blast = self._blast_tiles(grid, bomb_pos[0], bomb_pos[1], bomb_radius)
        score = 0.0

        # Direct hits
        for ex, ey in enemies:
            if (ex, ey) in blast and self._line_clear(grid, bomb_pos, (ex, ey)):
                score += 5.0

        # Boxes
        boxes = sum(1 for x, y in blast if grid[x, y] == 2)
        score += boxes * 1.5

        return score

    def _eval_position_for_offense(self, grid, next_pos, enemies, bombs_left, bomb_radius):
        """
        Position quality for future bombing
        """
        if bombs_left <= 0:
            return 0.0

        score = 0.0

        # Distance to kill spots
        for ex, ey in enemies:
            # Horizontal
            for dx in range(-bomb_radius, bomb_radius + 1):
                if self._line_clear(grid, (ex + dx, ey), (ex, ey)):
                    dist = abs(next_pos[0] - (ex + dx)) + abs(next_pos[1] - ey)
                    score += max(0, 5.0 - dist * 0.4)
            # Vertical
            for dy in range(-bomb_radius, bomb_radius + 1):
                if self._line_clear(grid, (ex, ey + dy), (ex, ey)):
                    dist = abs(next_pos[0] - ex) + abs(next_pos[1] - (ey + dy))
                    score += max(0, 5.0 - dist * 0.4)

        return score

    def _eval_positioning(self, grid, pos, enemies, freedom_map, blocked):
        """
        Positioning quality
        """
        freedom = freedom_map.get(pos, 0)
        score = 0.0

        if freedom <= 3:
            score -= 1.0
        elif freedom >= 8:
            score += 0.5

        open_neighbors = self._open_neighbors(grid, pos, blocked)
        score += open_neighbors * 0.2

        return score

    def _eval_economy(self, grid, pos, enemies, bombs_left, bomb_bonus):
        """
        Item value dynamic
        """
        mx, my = pos
        score = 0.0

        for x in range(max(0, mx - 4), min(grid.shape[0], mx + 5)):
            for y in range(max(0, my - 4), min(grid.shape[1], my + 5)):
                if grid[x, y] == 3 and bombs_left <= 1:
                    dist = abs(x - mx) + abs(y - my)
                    score += (5 - dist) * 0.2
                elif grid[x, y] == 4 and bomb_bonus <= 1:
                    dist = abs(x - mx) + abs(y - my)
                    score += (5 - dist) * 0.15

        return score

    def _eval_control(self, grid, pos, blocked, freedom_map):
        """
        Space control
        """
        neighbors_freedom = []
        for a in [1, 2, 3, 4]:
            npos = self._next_pos(pos, a)
            neighbors_freedom.append(freedom_map.get(npos, 0))

        avg = sum(neighbors_freedom) / len(neighbors_freedom) if neighbors_freedom else 0
        return min(1.0, avg / 6.0)

    # ═══ LOOKAHEAD & HELPERS ════════════════════════════════════════════════

    def _check_lookahead_safety(self, grid, start, blocked, danger_at, depth):
        """
        BFS lookahead: can survive from start for depth steps?
        """
        q = deque([(start, 0)])
        seen = {(start, 0)}

        while q:
            pos, t = q.popleft()
            if t > 0 and not any(ft in danger_at.get(pos, set()) for ft in range(t, depth + 2)):
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

    def _timed_escape_bfs(self, grid, start, blocked, danger_at, max_t=10):
        """
        Fast escape BFS
        """
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

    def _safe_wander(self, grid, my_pos, enemies, blocked, danger_at, valid_actions):
        """
        Safe fallback wander
        """
        best_action = 0
        best_score = -float('inf')

        for a in valid_actions:
            if a == 0:
                continue
            npos = self._next_pos(my_pos, a)
            if any(t in danger_at.get(npos, set()) for t in [1, 2, 3]):
                continue

            score = self._open_neighbors(grid, npos, blocked) * 1.0
            if enemies:
                min_dist = min(abs(npos[0] - ex) + abs(npos[1] - ey) for ex, ey in enemies)
                if 2 <= min_dist <= 5:
                    score += (6 - min_dist) * 0.5

            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    # ═══ BASIC HELPERS ══════════════════════════════════════════════════════

    def _elapsed_ms(self):
        return (time.time() - self._time_start) * 1000

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
