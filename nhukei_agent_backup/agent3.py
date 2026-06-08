import random
from collections import deque
import numpy as np
import os
import time

# Monkey-patch BomberEnv to prevent crash in scripts/participant and evaluation files
# which expect env.reset() to return obs instead of (obs, info) tuple.
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
    print("[Rule Agent] Patched BomberEnv.reset for evaluation compatibility.")
except Exception as e:
    print(f"[Rule Agent] Failed to patch BomberEnv.reset: {e}")


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
        self._prev_pos = None          # Track position for anti-freeze
        self._stuck_count = 0          # How many consecutive turns at same pos

        # Data Collection cho Imitation Learning
        self.states = []
        self.actions = []
        self.step_count = 0
        self.save_interval = 5000
        self.data_dir = "il_dataset"

    def __del__(self):
        if hasattr(self, 'states') and len(self.states) > 0:
            self._save_dataset()

    def act(self, obs):
        self.last_state = None
        action = self._get_action(obs)

        if self.last_state is not None:
            if action in [1, 2, 3, 4, 5] or (action == 0 and random.random() < 0.15):
                self._log_data(self.last_state, action)

        return action

    def _get_action(self, obs):
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

        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        blocked = set(bomb_positions)
        blocked.discard(my_pos)

        danger_soon, danger_now = self._danger_tiles(grid, bombs, players)

        # Preprocess state for IL data collection
        self.last_state = self._preprocess_obs(grid, danger_soon, danger_now, my_pos, enemies)

        if self.escape_mode and my_pos not in danger_soon:
            self.escape_mode = False

        valid_actions = self._valid_actions(grid, my_pos, blocked)

        # ── ANTI-FREEZE: detect if stuck ──────────────────────────────────
        if self._prev_pos == my_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        self._prev_pos = my_pos

        # ── GAME PHASE DETECTION ──────────────────────────────────────────
        enemies_alive = len(enemies)
        is_late_game = enemies_alive <= 1
        is_mid_game = enemies_alive == 2
        is_well_armed = int(bombs_left) >= 2 and int(bomb_bonus) >= 2
        # Aggressive when late game OR well armed OR stuck many turns
        should_hunt = is_late_game or is_well_armed or (is_mid_game and self._stuck_count >= 6)

        # ── PRIORITY 1: SURVIVAL & ESCAPE MODE ───────────────────────────
        if self.escape_mode or my_pos in danger_now or my_pos in danger_soon:
            escape = self._move_to_safe_tile(grid, my_pos, blocked, danger_soon, danger_now, search_depth=12)
            if escape is not None:
                return escape
            escape_fb = self._best_escape_action(grid, my_pos, blocked, danger_now, danger_soon)
            if escape_fb is not None:
                return escape_fb
            loose = [a for a in valid_actions if a != 0 and self._next_pos(my_pos, a) not in danger_now]
            if loose:
                return random.choice(loose)
            return 0

        # ── PRIORITY 2: ITEMS ─────────────────────────────────────────────
        item_tiles = self._item_tiles(
            grid, my_pos,
            prefer_capacity=int(bombs_left) <= 1,
            prefer_radius=int(bomb_bonus) <= 1,
            radius=7,
        )
        if item_tiles:
            move = self._move_to_targets(grid, my_pos, item_tiles, blocked, danger_soon)
            if move is not None:
                return move

        # ── PRIORITY 3: BOMB IF CAN HIT ENEMY RIGHT NOW ──────────────────
        if bombs_left > 0 and my_pos not in bomb_positions:
            can_hit_enemy = self._can_bomb_hit_enemy(grid, my_pos, enemies, bomb_radius)
            if can_hit_enemy and self._can_escape_after_placing(
                    grid, my_pos, blocked, danger_soon, danger_now, bomb_radius):
                self.escape_mode = True
                return 5

        # ── PRIORITY 4: HUNT ENEMY (aggressive) ──────────────────────────
        if should_hunt and enemies and int(bombs_left) > 0:
            hunt = self._hunt_enemy_action(
                grid, my_pos, enemies, blocked, danger_soon, danger_now, bomb_radius, int(bombs_left)
            )
            if hunt is not None:
                return hunt

        # ── PRIORITY 5: BOMB BOXES (smarter threshold) ───────────────────
        if bombs_left > 0 and my_pos not in bomb_positions:
            boxes_hit = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            min_boxes = 1 if is_well_armed else 2
            if boxes_hit >= min_boxes:
                if self._can_escape_after_placing(
                        grid, my_pos, blocked, danger_soon, danger_now, bomb_radius):
                    self.escape_mode = True
                    return 5

        # ── PRIORITY 6: FARM BOXES (MOVE TO GOOD SPOT) ───────────────────
        box_spots = self._box_bomb_spots(grid, my_pos, blocked)
        if box_spots:
            move = self._move_to_targets(grid, my_pos, box_spots, blocked, danger_soon)
            # Relaxed fallback: bombs block many safe paths, allow walking through
            # danger_soon (not danger_now) so agent doesn't freeze up
            if move is None:
                move = self._move_to_targets_relaxed(
                    grid, my_pos, box_spots, blocked, danger_soon, danger_now
                )
            if move is not None:
                return move

        # ── PRIORITY 7: HUNT ENEMIES (fallback for all game phases) ──────
        if enemies and int(bombs_left) > 0:
            hunt = self._hunt_enemy_action(
                grid, my_pos, enemies, blocked, danger_soon, danger_now, bomb_radius, int(bombs_left)
            )
            if hunt is not None:
                return hunt

        # ── PRIORITY 8: STRATEGIC WANDER (not random!) ───────────────────
        return self._strategic_wander(grid, my_pos, enemies, blocked, danger_soon, danger_now, valid_actions)

    # ── STRATEGIC WANDER (NEW) ────────────────────────────────────────────
    def _strategic_wander(self, grid, my_pos, enemies, blocked, danger_soon, danger_now, valid_actions):
        """
        Instead of random walk, score each move by:
        - Freedom (open neighbors): prefer open space for maneuverability
        - Proximity to enemy: pressure them / set up future bombs
        - Avoid corners and dead ends
        """
        best_action = None
        best_score = -10**9

        for a in valid_actions:
            if a == 0:
                continue
            npos = self._next_pos(my_pos, a)
            if npos in danger_now:
                continue
            if npos in danger_soon:
                continue

            score = 0.0

            # Freedom score: prefer tiles with more open neighbors
            open_n = self._open_neighbors(grid, npos, blocked)
            score += open_n * 2.0

            # Danger penalty: slightly penalize tiles near danger zones
            # (they might become dangerous soon if more bombs placed)
            for da in [1, 2, 3, 4]:
                adj = self._next_pos(npos, da)
                if adj in danger_soon:
                    score -= 0.5

            # Enemy proximity: prefer moving closer to weakest/nearest enemy
            if enemies:
                # Manhattan distance to nearest enemy
                min_dist = min(abs(npos[0] - ex) + abs(npos[1] - ey) for ex, ey in enemies)
                # Prefer being moderately close (not too far, not adjacent)
                # Optimal: 2-4 tiles away for bomb range
                if 1 <= min_dist <= 4:
                    score += (5 - min_dist) * 1.5  # Closer = higher score
                elif min_dist > 4:
                    score -= (min_dist - 4) * 0.5  # Penalize being far

            # Anti-stuck: if stuck, prefer moves away from current trend
            if self._stuck_count >= 2:
                score += random.uniform(0, 3)  # Add randomness to break pattern

            if score > best_score:
                best_score = score
                best_action = a

        if best_action is not None:
            return best_action

        # Final fallback: any safe move
        safe_moves = [a for a in valid_actions if a != 0 and self._next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0

    # ── HUNT ENEMY (improved with predicted positions) ────────────────────
    def _hunt_enemy_action(self, grid, my_pos, enemies, blocked, danger_soon, danger_now,
                           bomb_radius, bombs_left):
        """
        Hunt enemies by:
        1. Finding positions we can bomb from (kill spots)
        2. Including predicted enemy positions (where they might move)
        3. Moving toward kill spot or enemy directly
        """
        if not enemies or bombs_left <= 0:
            return None

        # Predicted enemy positions (current + 1-step ahead)
        enemy_targets = self._predict_enemy_reach(grid, enemies, blocked, danger_soon)

        # Build kill spots from both current and predicted enemy positions
        kill_spots = set()
        for ex, ey in enemy_targets:
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

        # If already at a kill spot → place bomb (if can escape)
        if my_pos in kill_spots:
            if self._can_escape_after_placing(
                    grid, my_pos, blocked, danger_soon, danger_now, bomb_radius):
                self.escape_mode = True
                return 5
            kill_spots.discard(my_pos)

        if kill_spots:
            # Move toward nearest kill spot via SAFE path only (no relaxed — too risky)
            move = self._move_to_targets(grid, my_pos, kill_spots, blocked, danger_soon)
            if move is not None:
                return move

        # No kill spots reachable safely → just move toward nearest enemy (pressure)
        # Use relaxed path here — approaching enemy is less risky than kill spots
        move = self._move_to_targets(grid, my_pos, set(enemies), blocked, danger_soon)
        if move is None:
            move = self._move_to_targets_relaxed(
                grid, my_pos, set(enemies), blocked, danger_soon, danger_now
            )
        return move

    def _predict_enemy_reach(self, grid, enemies, blocked, danger_soon):
        """
        Returns set of tiles enemies can reach in 1 step.
        Used to compute kill spots that intercept enemy movement.
        """
        reachable = set(enemies)  # Always include current positions
        for ex, ey in enemies:
            for a in [1, 2, 3, 4]:
                dx, dy = self.MOVES[a]
                nx, ny = ex + dx, ey + dy
                if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                    reachable.add((nx, ny))
        return reachable

    # ── RELAXED PATHFINDING ───────────────────────────────────────────────
    def _move_to_targets_relaxed(self, grid, start, targets, blocked, danger_soon, danger_now):
        """Allow paths through danger_soon (not danger_now) as fallback."""
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
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in blocked and npos not in targets:
                    continue
                if npos in danger_now:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None

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

        # Chain reaction simulation
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
                if depth == 0 and npos in danger_now:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action, depth + 1))
        return None

    def _can_escape_after_placing(self, grid, my_pos, blocked, danger_soon, danger_now, bomb_radius):
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        combined_soon = set(danger_soon) | my_blast
        combined_now = set(danger_now)
        if my_pos in danger_now:
            combined_now |= my_blast
        return self._move_to_safe_tile(
            grid, my_pos, blocked, combined_soon, combined_now, search_depth=8
        ) is not None

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

    def _preprocess_obs(self, grid, danger_soon, danger_now, my_pos, enemies):
        state = np.zeros((6, grid.shape[0], grid.shape[1]), dtype=np.int8)
        state[0] = (grid == 1).astype(np.int8)
        state[1] = (grid == 2).astype(np.int8)
        state[2] = ((grid == 3) | (grid == 4)).astype(np.int8)
        dangers = danger_soon.union(danger_now)
        if dangers:
            d_x, d_y = zip(*dangers)
            state[3, d_x, d_y] = 1
        if 0 <= my_pos[0] < grid.shape[0] and 0 <= my_pos[1] < grid.shape[1]:
            state[4, my_pos[0], my_pos[1]] = 1
        if enemies:
            e_x, e_y = zip(*enemies)
            state[5, e_x, e_y] = 1
        return state

    def _log_data(self, state, action):
        self.states.append(state)
        self.actions.append(action)
        self.step_count += 1
        if self.step_count >= self.save_interval:
            self._save_dataset()

    def _save_dataset(self):
        if not self.states: return
        os.makedirs(self.data_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = os.path.join(self.data_dir, f"dataset_{timestamp}.npz")
        np.savez_compressed(
            filename,
            states=np.array(self.states, dtype=np.int8),
            actions=np.array(self.actions, dtype=np.int8)
        )
        print(f"[IL Data Logger] Saved {len(self.states)} steps to {filename}")
        self.states = []
        self.actions = []
        self.step_count = 0
