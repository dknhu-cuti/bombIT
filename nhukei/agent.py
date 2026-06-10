import random
from collections import deque
from typing import Optional

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

        bomb_score = self._bomb_score(grid,my_pos,enemies,blocked,danger_at,freedom_map,bomb_radius)
        #if bomb_score > 0:
            #print(
                #"bomb score:",
                #round(bomb_score, 1)
            #)
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
        

        # ── PRIORITY 3: TRAP ENEMY IN DEAD-END ───────────────────────────
        

        # ── PRIORITY 4: BOMB IF CAN HIT ENEMY ────────────────────────────
    


        # ── PRIORITY 6: BOMB BOXES ───────────────────────────────────────
      

        # ── PRIORITY 7: FARM BOXES ───────────────────────────────────────
        

        # ── PRIORITY 8: HUNT ENEMIES (fallback) ──────────────────────────
    

        # ── PRIORITY 9: STRATEGIC WANDER ─────────────────────────────────
        return self._evaluate_actions(
    grid,
    my_pos,
    enemies,
    blocked,
    danger_at,
    danger_soon,
    danger_now,
    valid_actions,
    freedom_map,
    int(bombs_left),
    int(bomb_bonus),
    bomb_radius
)
    def _trap_score(
        self,
        grid,
        my_pos,
        enemies,
        blocked,
        freedom_map,
        bomb_radius
    ):
        score = 0
        my_blast = self._blast_tiles(
            grid,
            my_pos[0],
            my_pos[1],
            bomb_radius
        )
        
        for ex, ey in enemies:

            enemy_pos = (ex, ey)
            if enemy_pos in my_blast:
                score += 80
            # địch đang trong vùng hẹp
            if freedom_map.get(enemy_pos, 0) <= 3:

                exit_pos = self._find_dead_end_exit(
                    grid,
                    enemy_pos,
                    blocked,
                    freedom_map
                )

                if exit_pos is None:
                    continue

                # mình đang đứng ngay cửa
                if my_pos == exit_pos:

                    score += 150

                # cửa nằm trong vùng nổ của bom
                elif exit_pos in self._blast_tiles(
                    grid,
                    my_pos[0],
                    my_pos[1],
                    bomb_radius
                ):

                    score += 100

        return score
    def _item_score(self, grid, pos, bombs_left, bomb_bonus):

        cell = grid[pos]

        score = 0

        if cell == 3:

            if bomb_bonus <= 1:
                score += 40
            else:
                score += 15

        elif cell == 4:

            if bombs_left <= 1:
                score += 40
            else:
                score += 15

        return score
    def _box_score(
    self,
    grid,
    pos,
    bomb_radius
):

        boxes = self._count_boxes_in_blast(
            grid,
            pos,
            bomb_radius
        )

        return boxes * 20
    def _bomb_score(
        self,
        grid,
        my_pos,
        enemies,
        blocked,
        danger_at,
        freedom_map,
        bomb_radius
    ):

        score = 0.0

        # box
        score += (
            self._count_boxes_in_blast(
                grid,
                my_pos,
                bomb_radius
            )
            * 20
        )

        # enemy
        if self._can_bomb_hit_enemy(
            grid,
            my_pos,
            enemies,
            bomb_radius
        ):
            score += 120
            enemy_reach = self._predict_enemy_reach(
                grid,
                enemies,
                blocked,
                depth=4
            )

            my_blast = self._blast_tiles(
                grid,
                my_pos[0],
                my_pos[1],
                bomb_radius
            )

            for tile in my_blast:

                reach_times = enemy_reach.get(tile)

                if reach_times:

                    earliest = min(reach_times)

                    # enemy có thể chạy vào vùng nổ sau vài turn
                    if earliest <= 4:
                        score += (
                            5 - earliest
                        ) * 25
        score += self._trap_score(
            grid,
            my_pos,
            enemies,
            blocked,
            freedom_map,
            bomb_radius
        )
        # safety
        if not self._can_escape_after_placing(
            grid,
            my_pos,
            blocked,
            danger_at,
            bomb_radius
        ):
            return -9999

        # open space
        score += (
            freedom_map.get(
                my_pos,
                0
            )
            * 2
        )

        return score           
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
        q: deque[tuple[tuple[int, int], Optional[int], int]] = deque(
    [(start, None, 0)]
)
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
        q: deque[tuple[tuple[int, int], Optional[int], int]] = deque(
    [(start, None, 0)]
)
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

    def _hunt_enemy_action(self,
                       grid,
                       my_pos,
                       enemies,
                       blocked,
                       danger_at,
                       bomb_radius,
                       bombs_left,
                       freedom_map):

        if not enemies or bombs_left <= 0:
            return None

        heat = self._enemy_heatmap(
            grid,
            enemies,
            blocked,
            freedom_map,
            depth=4
        )

        if not heat:
            return None

        best_heat = max(heat.values())

        hotspots = {
            pos
            for pos, score in heat.items()
            if score >= best_heat * 0.7
        }

        kill_spots = set()

        for ex, ey in hotspots:

            for dx in range(-bomb_radius, bomb_radius + 1):

                nx = ex + dx

                if not self._passable(grid, nx, ey):
                    continue

                if (nx, ey) in blocked:
                    continue

                if self._line_clear(grid, (nx, ey), (ex, ey)):
                    kill_spots.add((nx, ey))

            for dy in range(-bomb_radius, bomb_radius + 1):

                ny = ey + dy

                if not self._passable(grid, ex, ny):
                    continue

                if (ex, ny) in blocked:
                    continue

                if self._line_clear(grid, (ex, ny), (ex, ey)):
                    kill_spots.add((ex, ny))

        if my_pos in kill_spots:

            if self._can_escape_after_placing(
                grid,
                my_pos,
                blocked,
                danger_at,
                bomb_radius
            ):
                self.escape_mode = True
                return 5

            kill_spots.discard(my_pos)

        if kill_spots:

            move = self._timed_move_to_target(
                grid,
                my_pos,
                kill_spots,
                blocked,
                danger_at
            )

            if move is not None:
                return move

        move = self._timed_move_to_target(
            grid,
            my_pos,
            hotspots,
            blocked,
            danger_at
        )

        return move
    def _enemy_heatmap(
        self,
        grid,
        enemies,
        blocked,
        freedom_map,
        depth=4):

        heat = {}

        for enemy in enemies:

            q = deque([(enemy, 0, 10.0)])

            seen = {}

            while q:

                pos, dist, score = q.popleft()

                if score < 0.1:
                    continue

                heat[pos] = heat.get(pos, 0) + score

                if dist >= depth:
                    continue

                for a in [0, 1, 2, 3, 4]:

                    nx, ny = self._next_pos(pos, a)
                    npos = (nx, ny)

                    if not self._passable(grid, nx, ny):
                        continue

                    if npos in blocked:
                        continue

                    next_score = score * 0.75

                    # phạt đứng yên
                    if a == 0:
                        next_score *= 0.6

                    # thưởng vùng rộng
                    freedom = freedom_map.get(npos, 1)

                    next_score *= (
                        1.0 +
                        min(freedom, 10) / 10.0
                    )

                    # thưởng item
                    cell = grid[nx, ny]

                    if cell in [3, 4]:
                        next_score *= 1.5

                    old = seen.get(npos, -1)

                    if old >= next_score:
                        continue

                    seen[npos] = next_score

                    q.append(
                        (
                            npos,
                            dist + 1,
                            next_score
                        )
                    )

        return heat
    def _distance_map(
        self,
        grid,
        starts,
        blocked):

        dist = {}

        q = deque()

        for s in starts:
            q.append((s, 0))
            dist[s] = 0

        while q:

            pos, d = q.popleft()

            for a in [1,2,3,4]:

                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)

                if not self._passable(grid, nx, ny):
                    continue

                if npos in blocked:
                    continue

                if npos in dist:
                    continue

                dist[npos] = d + 1

                q.append(
                    (
                        npos,
                        d + 1
                    )
                )

        return dist
    def _territory_score(
        self,
        grid,
        my_pos,
        enemies,
        blocked):

        my_dist = self._distance_map(
            grid,
            [my_pos],
            blocked
        )

        enemy_dist = self._distance_map(
            grid,
            enemies,
            blocked
        )

        my_area = 0
        enemy_area = 0

        all_tiles = set(my_dist) | set(enemy_dist)
        for pos in all_tiles:

            md = my_dist.get(pos, 999)
            ed = enemy_dist.get(pos, 999)

            if md < ed:
                my_area += 1

            elif ed < md:
                enemy_area += 1

        return my_area - enemy_area
    def _predict_enemy_reach(self, grid, enemies, blocked, depth=4):
        reach_time = {}

        for enemy in enemies:
            q = deque([(enemy, 0)])
            seen = {(enemy, 0)}

            while q:
                pos, t = q.popleft()

                if pos not in reach_time:
                    reach_time[pos] = set()

                reach_time[pos].add(t)

                if t >= depth:
                    continue

                for a in [0, 1, 2, 3, 4]:
                    nx, ny = self._next_pos(pos, a)
                    npos = (nx, ny)

                    if not self._passable(grid, nx, ny):
                        continue

                    if npos in blocked:
                        continue

                    state = (npos, t + 1)

                    if state not in seen:
                        seen.add(state)
                        q.append((npos, t + 1))

        return reach_time

    
    def _evaluate_actions(
        self,
        grid,
        my_pos,
        enemies,
        blocked,
        danger_at,
        danger_soon,
        danger_now,
        valid_actions,
        freedom_map,
        bombs_left,
        bomb_bonus,
        bomb_radius
    ):

        heat = self._enemy_heatmap(
            grid,
            enemies,
            blocked,
            freedom_map,
            depth=4
        )

        enemy_reach = self._predict_enemy_reach(
            grid,
            enemies,
            blocked,
            depth=4
        )

        hunt_weight = 1.0
        territory_weight = 1.0
        farm_weight = 1.0
        item_weight = 1.0

        enemies_alive = len(enemies)

        if enemies_alive == 1:
            hunt_weight += 0.8

        if bombs_left >= 2:
            hunt_weight += 0.3

        if bomb_radius >= 3:
            hunt_weight += 0.5

        if bomb_bonus <= 1:
            item_weight += 0.7

        if bombs_left <= 1:
            item_weight += 0.5

        if enemies_alive <= 1:
            farm_weight *= 0.5

        candidate_actions = valid_actions.copy()

        if bombs_left > 0 and my_pos not in blocked:
            candidate_actions.append(5)

        best_action = 0
        best_score = -1e9

        for a in candidate_actions:

            # =========================
            # BOMB
            # =========================

            if a == 5:

                score = self._bomb_score(
                    grid,
                    my_pos,
                    enemies,
                    blocked,
                    danger_at,
                    freedom_map,
                    bomb_radius
                )

            # =========================
            # MOVE
            # =========================

            else:

                npos = self._next_pos(my_pos, a)

                score = 0.0

                if a == 0:
                    score -= 2 + self._stuck_count * 3

                if npos in danger_now:
                    continue

                if npos in danger_soon:
                    score -= 100

                score += (
                    self._open_neighbors(
                        grid,
                        npos,
                        blocked
                    ) * 5
                )

                score += (
                    freedom_map.get(npos, 0)
                    * 1.5
                )

                if enemies:

                    min_dist = min(
                        abs(npos[0] - ex)
                        + abs(npos[1] - ey)
                        for ex, ey in enemies
                    )

                    score += (
                        max(0, 10 - min_dist)
                        * 8
                        * hunt_weight
                    )

                    score += (
                        heat.get(npos, 0)
                        * 6
                        * hunt_weight
                    )
                    # ==========================
                    # ENEMY PREDICTION
                    # ==========================

                    reach_times = enemy_reach.get(npos)

                    if reach_times:

                        earliest = min(reach_times)

                        # tới sớm thì điểm cao
                        score += (
                            (5 - earliest)
                            * 10
                            * hunt_weight
                        )
                    territory = self._territory_score(
                        grid,
                        npos,
                        enemies,
                        blocked
                    )

                    score += (
                        territory
                        * 0.3
                        * territory_weight
                    )

                cell = grid[npos]

                if cell == 3:
                    score += 40 if bomb_bonus <= 1 else 15

                elif cell == 4:
                    score += 40 if bombs_left <= 1 else 15

                boxes = self._count_boxes_in_blast(
                    grid,
                    npos,
                    bomb_radius
                )

                score += (
                    boxes
                    * 20
                    * farm_weight
                )

            if score > best_score:
                best_score = score
                best_action = a

        if best_action == 5:
            self.escape_mode = True

        return best_action
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
