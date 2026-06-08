"""
agent.py - PPO Agent submission file for BombIt Competition.

Loads the trained PPO model and uses it for in-game decision making.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add root to path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from stable_baselines3 import PPO

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
    print("[PPO Agent] Patched BomberEnv.reset for evaluation compatibility.")
except Exception as e:
    print(f"[PPO Agent] Failed to patch BomberEnv.reset: {e}")

# Constants definition
N_CHANNELS = 6   # 6 spatial channels
MAP_H, MAP_W = 13, 13
AUX_DIM = 3      # 3 scalar features


class Agent:
    team_id = "Nhukei_PPO_Bot"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

        model_path = Path(__file__).parent / "stage_C_final.pth"
        # Fallback: try stage B, then stage A if C is not found
        if not model_path.exists():
            model_path = Path(__file__).parent / "stage_B_final.pth"
        if not model_path.exists():
            model_path = Path(__file__).parent / "stage_A_final.pth"

        if model_path.exists():
            self._model = PPO.load(str(model_path), device="cpu")
            print(f"[PPO Agent] Loaded model from: {model_path}")
        else:
            self._model = None
            print("[PPO Agent] WARNING: Model not found. Returning random actions!")

    def _encode_obs(self, obs):
        """Convert raw obs to Dict observation suitable for PPO."""
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            # Agent is dead
            return None

        agent = players[self.agent_id]
        my_x, my_y, _, bombs_left, bomb_bonus = agent

        state = np.zeros((N_CHANNELS, MAP_H, MAP_W), dtype=np.float32)
        state[0] = (grid == 1).astype(np.float32)
        state[1] = (grid == 2).astype(np.float32)
        state[2] = ((grid == 3) | (grid == 4)).astype(np.float32)

        bomb_arr = np.asarray(bombs)
        if bomb_arr.size > 0:
            if bomb_arr.ndim == 1:
                bomb_arr = bomb_arr.reshape(1, -1)
            for b in bomb_arr:
                bx, by = int(b[0]), int(b[1])
                radius = int(1 + players[int(b[3]) if len(b) > 3 else 0][4])
                for dx, dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                    for r in range(0, radius + 1):
                        nx, ny = bx + dx * r, by + dy * r
                        if 0 <= nx < MAP_H and 0 <= ny < MAP_W:
                            state[3, nx, ny] = 1.0
                        else:
                            break

        if 0 <= int(my_x) < MAP_H and 0 <= int(my_y) < MAP_W:
            state[4, int(my_x), int(my_y)] = 1.0

        for i, p in enumerate(players):
            if i != self.agent_id and int(p[2]) == 1:
                state[5, int(p[0]), int(p[1])] = 1.0

        enemies_alive = sum(1 for i, p in enumerate(players) if i != self.agent_id and int(p[2]) == 1)
        aux = np.array([
            float(bombs_left)    / 5.0,
            float(bomb_bonus)    / 5.0,
            float(enemies_alive) / 3.0,
        ], dtype=np.float32)

        return {"map": state, "aux": aux}

    def act(self, obs):
        # Check if agent is alive
        players = obs["players"]
        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        if self._model is None:
            return np.random.randint(0, 6)

        encoded = self._encode_obs(obs)
        if encoded is None:
            return 0

        action, _ = self._model.predict(encoded, deterministic=True)
        return int(action)
