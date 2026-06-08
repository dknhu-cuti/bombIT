"""
train_ppo.py — Script huấn luyện PPO cho BombIt Agent (Phase 3)

Cách chạy:
    # Giai đoạn A: Train với SmarterRuleAgent
    python nhukei/ppo_agent/train_ppo.py --stage A --timesteps 200000

    # Giai đoạn B: Train curriculum (Tactical + Self-play)
    python nhukei/ppo_agent/train_ppo.py --stage B --timesteps 500000 --load_model nhukei/ppo_agent/checkpoints/stage_A_final.zip

    # Giai đoạn C: Full self-play
    python nhukei/ppo_agent/train_ppo.py --stage C --timesteps 1000000 --load_model nhukei/ppo_agent/checkpoints/stage_B_final.zip
"""

import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import random
import time
from pathlib import Path
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# Đảm bảo root project trong sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine import BomberEnv
from nhukei.ppo_agent.reward import compute_reward

# ── Thư mục lưu model ──────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IL_MODEL_PATH  = root_dir / "best_il_model.pth"

# Kích thước bản đồ
MAP_H, MAP_W   = 13, 13
N_CHANNELS     = 6   # Giữ nguyên 6 kênh từ Phase 2 để warm-start từ IL
AUX_DIM        = 3   # bombs_left (norm), bomb_radius (norm), enemies_alive (norm)
N_ACTIONS      = 6

# ===========================================================================
# 1. GYMNASIUM WRAPPER
# ===========================================================================
class BombItEnv(gym.Env):
    """
    Bọc BomberEnv theo chuẩn Gymnasium để dùng với Stable-Baselines3.
    
    Observation space: Dict với 2 key:
        - "map":  Tensor (6, 13, 13) float32 — spatial state
        - "aux":  Vector (3,) float32        — scalar features
    Action space: Discrete(6)
    """
    metadata = {"render_modes": []}

    def __init__(self, agent_id: int = 0, opponent_provider=None, max_steps: int = 500, seed: int = 42):
        super().__init__()
        self.agent_id          = agent_id
        self.opponent_provider = opponent_provider  # Callable -> Agent object
        self.max_steps         = max_steps
        self._base_seed        = seed
        self._ep_count         = 0

        # Khai báo không gian observation (Dict space)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0.0, high=1.0, shape=(N_CHANNELS, MAP_H, MAP_W), dtype=np.float32),
            "aux": spaces.Box(low=0.0, high=1.0, shape=(AUX_DIM,),                dtype=np.float32),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Khởi tạo engine game
        self._env    = BomberEnv(max_steps=max_steps, seed=seed)
        self._prev_obs = None
        self._opponents = []

    def _make_opponents(self):
        """Lấy danh sách opponent từ provider (hỗ trợ curriculum)."""
        if self.opponent_provider is None:
            # Import locally để tránh circular dependency
            import sys
            sys.path.insert(0, str(root_dir))
            from agent import SmarterRuleAgent
            return [SmarterRuleAgent(i) for i in range(1, 4)]
        return self.opponent_provider(self._ep_count)

    def _encode_obs(self, obs):
        return encode_obs_for_ppo(obs, self.agent_id)

    def reset(self, seed=None, options=None):
        self._ep_count += 1
        ep_seed       = (self._base_seed + self._ep_count) % (2**31)
        raw_obs, _    = self._env.reset(seed=ep_seed)
        self._opponents = self._make_opponents()
        self._prev_obs = None
        obs = self._encode_obs(raw_obs)
        return obs, {}

    def step(self, action):
        # Thu thập action của tất cả agents
        actions = [None] * 4
        actions[self.agent_id] = int(action)
        for opp in self._opponents:
            try:
                opp_action = opp.act(self._raw_obs if hasattr(self, '_raw_obs') else self._env._get_obs())
            except Exception:
                opp_action = 0
            actions[opp.agent_id] = opp_action

        raw_next_obs, terminated, truncated = self._env.step(actions)
        done = terminated or truncated

        # Tính reward
        reward = compute_reward(self._prev_obs, raw_next_obs, agent_id=self.agent_id)

        self._prev_obs = raw_next_obs
        self._raw_obs  = raw_next_obs

        obs = self._encode_obs(raw_next_obs)
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


def encode_obs_for_ppo(obs, agent_id):
    """Chuyển raw obs từ engine sang (map_tensor, aux_vector) cho agent cụ thể."""
    grid    = obs["map"]
    players = obs["players"]
    bombs   = obs["bombs"]
    agent   = players[agent_id]

    my_x, my_y, _, bombs_left, bomb_bonus = agent

    # ── 6 CHANNELS ──
    state = np.zeros((N_CHANNELS, MAP_H, MAP_W), dtype=np.float32)
    state[0] = (grid == 1).astype(np.float32)          # Walls
    state[1] = (grid == 2).astype(np.float32)          # Boxes
    state[2] = ((grid == 3) | (grid == 4)).astype(np.float32)  # Items

    # Kênh 3: Danger (đơn giản hóa)
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

    # Kênh 4: Self
    if 0 <= int(my_x) < MAP_H and 0 <= int(my_y) < MAP_W:
        state[4, int(my_x), int(my_y)] = 1.0

    # Kênh 5: Enemies
    for i, p in enumerate(players):
        if i != agent_id and int(p[2]) == 1:
            state[5, int(p[0]), int(p[1])] = 1.0

    # ── AUX SCALARS ──
    enemies_alive = sum(1 for i, p in enumerate(players) if i != agent_id and int(p[2]) == 1)
    aux = np.array([
        float(bombs_left)  / 5.0,
        float(bomb_bonus)  / 5.0,
        float(enemies_alive) / 3.0,
    ], dtype=np.float32)

    return {"map": state, "aux": aux}


# ===========================================================================
# 2. FEATURE EXTRACTOR (CNN + AUX BRANCH)
# ===========================================================================
class BombItFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor cho PPO với:
    - CNN Branch xử lý tensor map 6 kênh (warm-start từ IL weights)
    - MLP Branch xử lý 3 scalar phụ
    Output: Vector feature 256 + 32 = 288 dim
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 288):
        super().__init__(observation_space, features_dim)

        map_space = observation_space["map"]  # (6, 13, 13)
        aux_space = observation_space["aux"]  # (3,)

        n_channels = map_space.shape[0]
        aux_dim    = aux_space.shape[0]

        # CNN Branch — giống hệt BombItCNN ở train_il.py
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 13 * 13, 256),
            nn.ReLU(),
        )

        # AUX (Scalar) Branch
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
        )

        # Tính actual output dim
        self._features_dim = 256 + 32

    def forward(self, observations):
        map_feat = self.cnn(observations["map"])
        aux_feat = self.aux_mlp(observations["aux"])
        return torch.cat([map_feat, aux_feat], dim=1)


# ===========================================================================
# 3. HÀM NẠP TRỌNG SỐ IL (WARM-START)
# ===========================================================================
def load_il_weights_into_ppo(ppo_model, il_model_path: str):
    """
    Nạp các lớp Conv2D và FC từ best_il_model.pth vào CNN Branch của PPO.
    Chỉ nạp các key trùng tên, bỏ qua các key không khớp (ví dụ lớp fc_block[2]).
    """
    if not Path(il_model_path).exists():
        print(f"[Warm-start] WARNING: IL model not found: {il_model_path}. Training from scratch.")
        return

    il_state = torch.load(il_model_path, map_location="cpu")
    ppo_extractor_state = ppo_model.policy.features_extractor.cnn.state_dict()

    # Map key từ BombItCNN sang CNN Branch của extractor
    key_map = {
        "conv_block.0.weight":  "0.weight",
        "conv_block.0.bias":    "0.bias",
        "conv_block.2.weight":  "2.weight",
        "conv_block.2.bias":    "2.bias",
        "fc_block.0.weight":    "5.weight",
        "fc_block.0.bias":      "5.bias",
    }

    transferred = 0
    for il_key, cnn_key in key_map.items():
        if il_key in il_state and cnn_key in ppo_extractor_state:
            if il_state[il_key].shape == ppo_extractor_state[cnn_key].shape:
                ppo_extractor_state[cnn_key] = il_state[il_key]
                transferred += 1

    ppo_model.policy.features_extractor.cnn.load_state_dict(ppo_extractor_state)
    print(f"[Warm-start] Successfully loaded {transferred}/{len(key_map)} layers from IL model!")


# ===========================================================================
# 4. CURRICULUM OPPONENT PROVIDER
# ===========================================================================
class CurriculumOpponentProvider:
    """
    Quản lý danh sách opponent thay đổi theo giai đoạn (Curriculum Learning).

    Stage A: Toàn bộ SmarterRuleAgent
    Stage B: Mix 50% TacticalRuleAgent + 50% snapshot self-play
    Stage C: 100% pool self-play (rolling snapshot)
    """
    STAGE_THRESHOLDS = {
        "A": float("inf"),  # Stage A không tự chuyển, phải restart
        "B": float("inf"),
        "C": float("inf"),
    }

    def __init__(self, stage: str = "A", snapshot_pool: list = None):
        self.stage         = stage
        self.snapshot_pool = snapshot_pool or []  # List các đường dẫn file .zip

    def add_snapshot(self, path: str):
        """Thêm snapshot mới vào pool. Giữ tối đa 5 bản."""
        self.snapshot_pool.append(path)
        if len(self.snapshot_pool) > 5:
            self.snapshot_pool.pop(0)

    def __call__(self, ep_count: int) -> list:
        """Trả về list 3 opponent agents (agent_id 1, 2, 3)."""
        opponents = []
        for opp_id in range(1, 4):
            opp = self._pick_opponent(opp_id)
            opponents.append(opp)
        return opponents

    def _pick_opponent(self, opp_id: int):
        import sys
        sys.path.insert(0, str(root_dir))
        from agent import SmarterRuleAgent, TacticalRuleAgent
        
        if self.stage == "A":
            return SmarterRuleAgent(opp_id)
        elif self.stage == "B":
            if self.snapshot_pool and random.random() < 0.5:
                return self._load_snapshot_agent(opp_id)
            return TacticalRuleAgent(opp_id)
        else:  # Stage C
            if self.snapshot_pool:
                return self._load_snapshot_agent(opp_id)
            return TacticalRuleAgent(opp_id)

    def _load_snapshot_agent(self, opp_id: int):
        """Load một snapshot ngẫu nhiên từ pool làm opponent."""
        snap_path = random.choice(self.snapshot_pool)
        return SnapshotAgent(opp_id, snap_path)


class SnapshotAgent:
    """Agent wrapper load PPO model từ file zip để dùng làm opponent."""
    def __init__(self, agent_id: int, model_path: str):
        self.agent_id = agent_id
        self._model   = PPO.load(model_path, device="cpu")

    def act(self, obs):
        try:
            # obs is raw_obs from the engine
            encoded_obs = encode_obs_for_ppo(obs, self.agent_id)
            action, _ = self._model.predict(encoded_obs, deterministic=False)
            return int(action)
        except Exception as e:
            print(f"SnapshotAgent error: {e}")
            return 0


# ===========================================================================
# 5. CALLBACK: TỰ ĐỘNG LƯU SNAPSHOT CHO SELF-PLAY
# ===========================================================================
class SelfPlayCallback(BaseCallback):
    """
    Sau mỗi `snapshot_interval` steps:
    - Lưu snapshot của model hiện tại.
    - Cập nhật opponent pool.
    - In log tỉ lệ thắng.
    """
    def __init__(self, snapshot_interval: int, provider: CurriculumOpponentProvider,
                 save_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.snapshot_interval = snapshot_interval
        self.provider          = provider
        self.save_dir          = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot    = 0

    def _on_training_start(self) -> None:
        self._last_snapshot = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot >= self.snapshot_interval:
            snap_path = str(self.save_dir / f"snapshot_{self.num_timesteps}.zip")
            self.model.save(snap_path)
            self.provider.add_snapshot(snap_path)
            self._last_snapshot = self.num_timesteps
            if self.verbose:
                print(f"\n[SelfPlay] Snapshot saved at {snap_path}. Pool size: {len(self.provider.snapshot_pool)}")
        return True


# ===========================================================================
# 6. HÀM TRAIN CHÍNH
# ===========================================================================
def train(stage: str, total_timesteps: int, load_model: str = None, seed: int = 42):
    print(f"\n{'='*60}")
    print(f"  PHASE 3 - PPO Training | Stage {stage} | {total_timesteps:,} steps")
    print(f"{'='*60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Khởi tạo curriculum provider
    provider = CurriculumOpponentProvider(stage=stage)

    # Tạo môi trường
    env = BombItEnv(agent_id=0, opponent_provider=provider, max_steps=500, seed=seed)

    # Policy kwargs: dùng BombItFeaturesExtractor tùy chỉnh
    policy_kwargs = {
        "features_extractor_class":  BombItFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 288},
        "net_arch":                  [128, 64],  # Lớp FC sau extractor
    }

    # Khởi tạo hoặc load PPO model
    if load_model and Path(load_model).exists():
        print(f"[Load] Load model from: {load_model}")
        model = PPO.load(load_model, env=env, device=device)
    else:
        model = PPO(
            policy          = "MultiInputPolicy",
            env             = env,
            learning_rate   = 3e-4,
            n_steps         = 1024,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,   # Khuyến khích khám phá
            verbose         = 1,
            device          = device,
            seed            = seed,
            policy_kwargs   = policy_kwargs,
        )

        # ── WARM-START TỪ IL ──
        if stage == "A":
            load_il_weights_into_ppo(model, str(IL_MODEL_PATH))

    # Callbacks
    self_play_cb = SelfPlayCallback(
        snapshot_interval = 50_000,
        provider          = provider,
        save_dir          = str(CHECKPOINT_DIR / f"stage_{stage}_snapshots"),
    )
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = str(CHECKPOINT_DIR),
        name_prefix = f"ppo_stage_{stage}",
    )

    # ── TRAIN ──
    model.learn(
        total_timesteps = total_timesteps,
        callback        = [self_play_cb, checkpoint_cb],
        reset_num_timesteps = (load_model is None),
    )

    # Lưu model cuối cùng
    final_path = str(CHECKPOINT_DIR / f"stage_{stage}_final")
    model.save(final_path)
    print(f"\n[DONE] Stage {stage} complete! Model saved at: {final_path}.zip")
    return model


# ===========================================================================
# 7. ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent cho BombIt (Phase 3)")
    parser.add_argument("--stage",      type=str,  default="A", choices=["A", "B", "C"],
                        help="Giai đoạn curriculum: A=Smarter, B=Mixed, C=SelfPlay")
    parser.add_argument("--timesteps",  type=int,  default=200_000,
                        help="Tổng số timesteps huấn luyện")
    parser.add_argument("--load_model", type=str,  default=None,
                        help="Đường dẫn file .zip để tiếp tục train")
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    train(
        stage           = args.stage,
        total_timesteps = args.timesteps,
        load_model      = args.load_model,
        seed            = args.seed,
    )
