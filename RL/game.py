import numpy as np
import gymnasium as gym
from gymnasium import spaces
from map import Map
from bomb import Bomb
from player import Player
from simple_rule_agent import SimpleRuleAgent


class BomberEnv(gym.Env):
    # N_ACTIONS = 6 # 0: STOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: PLACE_BOMB

    def __init__(self, width=13, height=13, max_steps=500, seed=None, bot1=None, bot2=None, bot3=None):
        super(BomberEnv, self).__init__()

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # Instantiate 3 simple bots for opponent players (ids 1, 2, 3)
        # Using SimpleRuleAgent (easier) so the RL agent can learn survival
        # and item collection before facing stronger opponents.
        self.bot1 = bot1 if bot1 is not None else SimpleRuleAgent(agent_id=1)
        self.bot2 = bot2 if bot2 is not None else SimpleRuleAgent(agent_id=2)
        self.bot3 = bot3 if bot3 is not None else SimpleRuleAgent(agent_id=3)

        # Observation space — mirrors _get_obs() exactly
        # map: (height, width) grid of cell types 0-4
        # players: (4, 5) array — [x, y, alive, bombs_left, bomb_radius_bonus]
        # bombs: fixed shape (height*width, 4) padded with -1 — [x, y, timer, owner_id]
        self.observation_space = spaces.Dict({
            "map": spaces.Box(
                low=0, high=4,
                shape=(self.height, self.width),
                dtype=np.int8
            ),
            "players": spaces.Box(
                low=np.int8(-1), high=np.int8(127),
                shape=(4, 5),
                dtype=np.int8
            ),
            "bombs": spaces.Box(
                low=np.int8(-1), high=np.int8(127),
                shape=(self.height * self.width, 4),
                dtype=np.int8
            ),
        })

        # Action space: 6 discrete actions for player 0
        self.action_space = spaces.Discrete(6)

        # Reward tracking
        self.old_stats = None

        # Initialise game state
        self.reset()

    def _get_obs(self):
        """Return observation with bombs padded to fixed shape (height*width, 4)."""
        max_bombs = self.height * self.width

        # Build raw bombs array
        if self.bombs:
            raw_bombs = np.array(
                [[b.x, b.y, b.timer, b.owner_id] for b in self.bombs],
                dtype=np.int8
            )
        else:
            raw_bombs = np.empty((0, 4), dtype=np.int8)

        # Pad with -1 rows to reach fixed size
        pad_rows = max_bombs - len(raw_bombs)
        padding = np.full((pad_rows, 4), -1, dtype=np.int8)
        bombs_obs = np.concatenate([raw_bombs, padding], axis=0) if len(raw_bombs) > 0 else padding

        return {
            "map": self.map.grid.astype(np.int8),
            "players": np.array(
                [[p.x, p.y, p.alive, p.bombs_left, p.bomb_radius_bonus] for p in self.players],
                dtype=np.int8
            ),
            "bombs": bombs_obs,
        }

    def reset(self, seed=None, options=None):
        # Required by gymnasium so that self._np_random is populated when seed is given
        super().reset(seed=seed)

        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)

        self.map = Map(self.width, self.height, seed=self.seed_val)
        self.players = [
            Player(0, 1, 1),
            Player(1, self.height - 2, self.width - 2),
            Player(2, 1, self.width - 2),
            Player(3, self.height - 2, 1)
        ]

        self.bombs = []
        self.current_step = 0

        # Initialise old_stats for reward calculation
        self.old_stats = self.players[0].stats.copy()

        # --- Exploration tracking: set of (x, y) tiles visited this episode ---
        p0_start = self.players[0]
        self.visited_tiles = {(p0_start.x, p0_start.y)}

        return self._get_obs(), {}

    def step(self, action):
        """
        action: single integer for Player 0 (the RL agent).
        Bots (players 1-3) act via TacticalRuleAgent.
        """
        self.current_step += 1
        pending_bombs = {}

        # Snapshot agent position BEFORE any movement this step
        p0_pre_x, p0_pre_y = self.players[0].x, self.players[0].y

        # Get current observation to feed the bots
        current_obs = self._get_obs()

        # Get bot actions
        action1 = self.bot1.act(current_obs)
        action2 = self.bot2.act(current_obs)
        action3 = self.bot3.act(current_obs)

        # Combine all actions: player 0 is the RL agent
        actions = [action, action1, action2, action3]

        # ── Game logic (identical to original BomberEnv.step) ──────────────
        for player_id, act in enumerate(actions):
            player = self.players[player_id]
            if player.alive == False:
                continue

            dx, dy = 0, 0
            if act == Player.LEFT:   dx = -1
            elif act == Player.RIGHT: dx = 1
            elif act == Player.UP:   dy = -1
            elif act == Player.DOWN:  dy = 1
            elif act == Player.PLACE_BOMB:
                if player.bombs_left <= 0:
                    continue
                if any(b.x == player.x and b.y == player.y for b in self.bombs):
                    continue
                radius = 1 + player.bomb_radius_bonus
                pos = (player.x, player.y)
                if pos not in pending_bombs or radius > pending_bombs[pos][1]:
                    pending_bombs[pos] = (player_id, radius)

            if dx != 0 or dy != 0:
                player.move(dx, dy, self.map.grid, self.players, self.bombs)

        # Resolve item collections after all movements
        tile_to_players = {}
        for p in self.players:
            if p.alive:
                pos = (p.x, p.y)
                if pos not in tile_to_players:
                    tile_to_players[pos] = []
                tile_to_players[pos].append(p)

        for (x, y), occupants in tile_to_players.items():
            cell = self.map.grid[x, y]
            if cell in [Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
                if len(occupants) == 1:
                    p = occupants[0]
                    if cell == Map.ITEM_RADIUS:
                        p.bomb_radius_bonus = min(p.bomb_radius_bonus + 1, Player.MAX_BOMB_RADIUS - 1)
                        p.stats['items'] += 1
                    elif cell == Map.ITEM_CAPACITY:
                        if p.max_bombs < Player.MAX_BOMB_CAPACITY:
                            p.max_bombs += 1
                            p.bombs_left += 1
                        p.stats['items'] += 1
                self.map.grid[x, y] = Map.GRASS

        for (bx, by), (owner_id, radius) in pending_bombs.items():
            self.bombs.append(Bomb(bx, by, owner_id, radius=radius))
            self.players[owner_id].bombs_left -= 1
            self.players[owner_id].stats['bombs'] += 1

        exploded_this_step = []
        for bomb in self.bombs:
            if bomb.step():
                exploded_this_step.append(bomb)

        # Chain reaction
        idx = 0
        while idx < len(exploded_this_step):
            bomb = exploded_this_step[idx]
            idx += 1
            explosion_tiles = self._get_explosion_tiles(bomb)
            for other_bomb in self.bombs:
                if other_bomb.exploded:
                    continue
                if (other_bomb.x, other_bomb.y) in explosion_tiles and other_bomb not in exploded_this_step:
                    other_bomb.exploded = True
                    exploded_this_step.append(other_bomb)

        if exploded_this_step:
            affected_tiles_map = {}
            for bomb in exploded_this_step:
                for tx, ty in self._get_explosion_tiles(bomb):
                    if (tx, ty) not in affected_tiles_map:
                        affected_tiles_map[(tx, ty)] = set()
                    affected_tiles_map[(tx, ty)].add(bomb.owner_id)
                p = self.players[bomb.owner_id]
                p.bombs_left = min(p.bombs_left + 1, p.max_bombs)

            self._apply_explosions(affected_tiles_map)
            self.bombs = [b for b in self.bombs if not b.exploded]

        self._spawn_random_items()
        # ── End of game logic ───────────────────────────────────────────────

        # ── Balanced reward calculation ───────────────────────────────────────
        p0 = self.players[0]

        # 1. Tiny base step cost — keeps training signal tight but not overwhelming
        reward = -0.01

        if not p0.alive:
            # 4. Balanced death penalty — reduced so the agent still dares to
            #    place bombs and explore risky areas.
            reward -= 15.0
        else:
            # Stat deltas since the last step
            kills_diff = p0.stats['kills'] - self.old_stats['kills']
            boxes_diff = p0.stats['boxes'] - self.old_stats['boxes']
            items_diff = p0.stats['items'] - self.old_stats['items']

            # 5. Enemy elimination — strong incentive for aggressive play
            reward += kills_diff * 15.0   # +15 per kill

            # 2. Box destruction — reward opening paths through obstacles
            reward += boxes_diff * 3.0    # +3 per box destroyed

            # Item pickup bonus (unchanged)
            reward += items_diff * 5.0    # +5 per item collected

            # 1. Exploration reward — visiting a new tile for the first time
            #    this episode encourages the agent to keep moving.
            current_pos = (p0.x, p0.y)
            if current_pos not in self.visited_tiles:
                reward += 0.03            # +0.03 for each newly explored tile
                self.visited_tiles.add(current_pos)

            # 3. Inactivity penalty — discourage standing still or bumping walls.
            #    Triggered when the agent chose STOP (action 0) OR tried to move
            #    but ended up on the exact same tile (invalid/blocked move).
            agent_stood_still = (p0.x == p0_pre_x and p0.y == p0_pre_y)
            if action == Player.STOP or (action in (Player.LEFT, Player.RIGHT,
                                                     Player.UP, Player.DOWN)
                                         and agent_stood_still):
                reward -= 0.01            # -0.01 per idle/blocked action

        self.old_stats = p0.stats.copy()
        # ── End of reward calculation ─────────────────────────────────────────

        terminated = sum(p.alive for p in self.players) <= 1 or not p0.alive
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_explosion_tiles(self, bomb):
        tiles = {(bomb.x, bomb.y)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in directions:
            for r in range(1, bomb.radius + 1):
                tx, ty = bomb.x + dx * r, bomb.y + dy * r
                if 0 <= tx < self.height and 0 <= ty < self.width:
                    if self.map.grid[tx, ty] == Map.WALL:
                        break
                    tiles.add((tx, ty))
                    if self.map.grid[tx, ty] == Map.BOX:
                        break
                    # explosion goes through players
                else:
                    break

        return tiles

    def _apply_explosions(self, affected_tiles_map):
        for (tx, ty), owner_ids in affected_tiles_map.items():
            for p in self.players:
                if p.x == tx and p.y == ty and p.alive:
                    p.alive = False
                    for oid in owner_ids:
                        if oid != p.id:
                            self.players[oid].stats['kills'] += 1
            if self.map.grid[tx, ty] == Map.BOX:
                self.map.grid[tx, ty] = Map.GRASS
                for oid in owner_ids:
                    self.players[oid].stats['boxes'] += 1
                rand = self.rng.random()
                if rand < 0.3:
                    self.map.grid[tx, ty] = Map.ITEM_RADIUS
                elif rand < 0.6:
                    self.map.grid[tx, ty] = Map.ITEM_CAPACITY
            # can destroy items
            elif self.map.grid[tx, ty] in [Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
                self.map.grid[tx, ty] = Map.GRASS

    def _spawn_random_items(self, spawn_prob=0.0003):
        for x in range(self.height):
            for y in range(self.width):
                if self.map.grid[x, y] != Map.GRASS:
                    continue
                if any(p.x == x and p.y == y and p.alive for p in self.players):
                    continue
                if self.rng.random() < spawn_prob * self.current_step / 165:
                    if self.rng.random() < 0.5:
                        self.map.grid[x, y] = Map.ITEM_RADIUS
                    else:
                        self.map.grid[x, y] = Map.ITEM_CAPACITY
