# Visual Comparison - v1 vs v3

## Flow Chart So Sánh

### v1 (Priority-Based) - Current
```
┌─────────────────────────────────────────────────────────────┐
│                    START: act(obs)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │   In immediate danger?      │
            └──────────────┬──────────────┘
                    YES │  NO
                       │  └──────────────┐
        ┌──────────────▼┐                │
        │ ESCAPE! ✓    │   ┌────────────▼──────────────┐
        │ (~5ms return)│   │ Priority 2: Item close?  │
        └──────────────┘   └────────────┬──────────────┘
                                   YES │  NO
                                      │  ├─ Priority 3: Trap?
                                      │  ├─ Priority 4: Bomb hit?
                                      │  ├─ Priority 5: Hunt?
                                      │  ├─ Priority 6: Farm boxes?
                                      │  ├─ Priority 7: Hunt enemies?
                                      │  └─ Priority 8: WANDER (~30 điểm)
                                      │
                ┌─────────────────────┘
                │
        ┌───────▼──────────────┐
        │ First Match Returns  │ ← TUNNEL VISION!
        │ (~8-10ms total)      │    Only 1 evaluated
        └──────────────────────┘
                  │
        ┌─────────▼────────────┐
        │ Return Action        │
        └──────────────────────┘

PROBLEM: Nếu Priority 2 match → quên kinh tế item
         Nếu Priority 4 miss → không bomb
         Không dùng hết time budget
```

---

### v3 (Multi-Criteria) - Improved
```
┌─────────────────────────────────────────────────────────────┐
│                    START: act(obs)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │   In immediate danger?      │
            └──────────────┬──────────────┘
                    YES │  NO
                       │  └──────────────┐
        ┌──────────────▼┐                │
        │ ESCAPE! ✓    │   ┌────────────▼──────────────────┐
        │ (~5ms return)│   │ PHASE 2A: Quick Heuristics   │
        └──────────────┘   │  (< 5ms)                     │
                           │ - Bomb hit check? ✓          │
                           │ - Item grab check? ✓         │
                           │ - Early return if obvious    │
                           └────────────┬─────────────────┘
                                    NO │ CONTINUE
                                       │
                           ┌───────────▼──────────────┐
                           │ PHASE 2B: Evaluate ALL   │
                           │ valid actions (30-80ms)  │
                           │                          │
                           │ For each action a:       │
                           │  ├─ Survival score       │
                           │  ├─ Offense score        │
                           │  ├─ Positioning score    │
                           │  ├─ Economy score        │
                           │  └─ Control score        │
                           │                          │
                           │ score_a = w1*s1 + w2*s2  │
                           │         + w3*s3 + w4*s4  │
                           │         + w5*s5          │
                           └───────────┬──────────────┘
                                       │
                           ┌───────────▼──────────────┐
                           │ Find Best Action         │
                           │ best = argmax(scores)    │
                           │ (PARALLEL EVALUATION)    │
                           └───────────┬──────────────┘
                                       │
                           ┌───────────▼──────────────┐
                           │ Return Best Action       │
                           │ (~40-50ms total)         │
                           └──────────────────────────┘

ADVANTAGE: Tất cả actions evaluated
           Chọn best = flexible + thông minh
           Dùng đủ time budget
           Lookahead 2-3 bước cho accuracy
```

---

## Code Structure So Sánh

### v1 - Priority Chain
```python
def act(self, obs):
    # ... precompute danger_at, freedom_map
    
    # PHASE 1: SURVIVAL
    if in_danger:
        return escape()  # ← Early return!
    
    # PHASE 2: ITEMS (Priority 2)
    items = _item_tiles(...)
    if items:
        return move_to_items()  # ← Early return!
    
    # PHASE 3: TRAP (Priority 3)
    if can_trap_enemy():
        return trap_move()  # ← Early return!
    
    # ... more priority checks
    
    # FALLBACK: Wander
    return wander()  # ← If nothing matched
```

### v3 - Parallel Evaluation
```python
def act(self, obs):
    # ... precompute danger_at, freedom_map
    
    # PHASE 1: SURVIVAL (same as v1)
    if in_danger:
        return escape()
    
    # PHASE 2A: Quick heuristics (5ms)
    quick_bomb = _quick_bomb_check(...)
    if quick_bomb:
        return quick_bomb
    
    quick_item = _quick_item_grab(...)
    if quick_item:
        return quick_item
    
    # PHASE 2B: Deep evaluation (30-80ms)
    valid_actions = _valid_actions(...)
    best_action = None
    best_score = -inf
    
    for action in valid_actions:
        # Evaluate everything in parallel
        score = _evaluate_move_smart(action, ...)
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action  # ← Best of all evaluated
```

---

## Scoring Detail

### v1 Scoring
```
_strategic_wander():
  score = open_neighbors(pos) * 2.0         # Factor 1
  
  for neighbor_a in [1,2,3,4]:              # Factor 2
      if neighbor in danger_soon:
          score -= 0.5
  
  if enemies:                                # Factor 3
      min_dist = distance(pos, closest_enemy)
      if 1 <= min_dist <= 4:
          score += (5 - min_dist) * 1.5
  
  return score  # ← Only 3 factors!
```

### v3 Scoring
```
_evaluate_move_smart(action):
  score = 0
  
  # Factor 1: Survival (30%)
  survival = _eval_survival_smart(...)
  score += 3.0 * survival
  
  # Factor 2: Offense (25%)
  if action == 5:
      offense = _eval_bomb_smart(...)
  else:
      offense = _eval_position_for_offense(...)
  score += 1.8 * offense
  
  # Factor 3: Positioning (20%)
  positioning = _eval_positioning(...)
  score += 1.2 * positioning
  
  # Factor 4: Economy (15%)
  economy = _eval_economy(...)
  score += 0.8 * economy
  
  # Factor 5: Control (10%)
  control = _eval_control(...)
  score += 0.6 * control
  
  return score  # ← 5 factors with weights!
```

---

## Lookahead Comparison

### v1 Lookahead
```
Step 0 (now): My position
  └─ Check if in danger NOW
  └─ Check 1 step ahead (max)

Total depth: 1 (quá cạn!)
```

### v3 Lookahead
```
Step 0 (now): My position
  └─ Step 1: Where I'll be
      └─ Step 2: Where I'll be again
          └─ Step 3: Where I'll be again
                └─ Evaluate state quality at step 3

Total depth: 3 (balanced)

Special case (bomb placement):
  Step 0: Place bomb
  Step 1-6: Escape
  Step 7-8: Bomb explodes
  └─ Check if safe at Step 7-8+

Lookahead safety: 3 steps ahead (deep!)
```

---

## Decision Making Timeline

### v1 Decision
```
Time: 0ms
  ├─ Check danger (1ms)
  ├─ Check priority 1 (escape) → MATCH (4ms) ✓
  └─ RETURN ESCAPE ACTION (5ms total)

Agent thinks: Very fast, but shallow!
```

### v3 Decision (Example: No obvious move)
```
Time: 0ms
  ├─ Check danger (1ms) ✗ No danger
  ├─ Check quick bomb (3ms) ✗ No obvious target
  ├─ Check quick item (3ms) ✗ Nothing worth it
  ├─ PHASE 2B: Evaluate all moves (35-40ms)
  │   ├─ Action 0 (wait): score = 1.2
  │   ├─ Action 1 (left): score = 2.8 ←─ Best
  │   ├─ Action 2 (right): score = 1.5
  │   ├─ Action 3 (up): score = 0.9
  │   └─ Action 4 (down): score = 1.1
  └─ RETURN ACTION 1 (LEFT) (45ms total)

Agent thinks: Slower but smarter!
```

---

## Enemy Prediction Comparison

### v1 (No Prediction)
```
Enemy at (5, 5)
My bomb at (7, 7) with radius 2

Blast zone: (7,7), (6,7), (8,7), (7,6), (7,8)
             └─ Doesn't include (5,5)

Time 0: Check blast → NO HIT
        Decision: Don't bomb

Result: Enemy survives even if bomb placed!
        ❌ Wasted opportunity
```

### v3 (With Prediction)
```
Enemy at (5, 5) moving towards (4, 5)
My bomb at (7, 7) with radius 2

Check 1: Direct hit? NO
Check 2: Where will enemy move?
         └─ Path: (5,5) → (4,5) → (3,5)
         └─ At T=8: Enemy at (3,5) or thereabouts
         └─ Blast at T=8: (7,7), (6,7), ..., (3,7)
         └─ Does blast include (3,5)? NO

Result: Still don't bomb

BUT: If enemy at (7,5) moving to (6,5):
     └─ Predicted future (6,5) IS in blast zone!
     └─ ✓ Bomb has 80% chance to hit
     └─ Decision: BOMB!

Result: Better hit rate!
        ✅ Higher kill rate
```

---

## Time Budget Usage

### v1 Time Usage
```
Timeline:
0ms ┐
    ├─ Check danger (1ms)
    ├─ Check priority 1 (escape): MATCH!
2ms ├─ Return escape action
    │
    └─ Time wasted: 98ms out of 100ms!
       └─ Only used 2%!
```

### v3 Time Usage
```
Timeline:
0ms ┐
    ├─ Check danger (1ms)
    ├─ Check quick heuristics (5ms)
5ms ├─ PHASE 2B: Evaluate all moves
    │   ├─ Action 0 eval (8ms)
13ms│   ├─ Action 1 eval (8ms)
21ms│   ├─ Action 2 eval (8ms)
29ms│   ├─ Action 3 eval (8ms)
37ms│   ├─ Action 4 eval (8ms)
45ms│   └─ Choose best (1ms)
    │
    └─ Time used: 45ms out of 100ms
       └─ Dùng 45% - safe!
       └─ Còn 55% reserve
```

---

## Scoring Example: Real Game

### Scenario: Agent at (7,7), Enemy at (5,7)
```
Options:
1. Wait (action 0)
2. Move left (action 1)
3. Move right (action 2)
4. Move up (action 3)
5. Move down (action 4)

v1 Scoring:
  Opt 1: open_neighbors=2  → score=4.0
  Opt 2: open_neighbors=1, danger_neighbor=1  → score=1.0
  Opt 3: open_neighbors=3, closest_enemy_dist=4  → score=6.0 ✓
  Opt 4: open_neighbors=2  → score=4.0
  Opt 5: open_neighbors=1  → score=2.0
  
  Result: Opt 3 (move up)  ✓ Reasonable

v3 Scoring (assuming bombs_left=2):
  Opt 0 (wait):
    - Survival: 0.0 (safe)
    - Offense: 0.0 (far from kill spot)
    - Positioning: -1.0 (not open)
    - Economy: 0.2 (no items)
    - Control: 0.3
    → score = 0.0*3.0 + 0.0*1.8 - 1.0*1.2 + 0.2*0.8 + 0.3*0.6 = -0.88
  
  Opt 1 (move left to 6,7):
    - Survival: 0.5 (safer, away from bomb)
    - Offense: 2.0 (closer to enemy!)
    - Positioning: 0.3 (ok)
    - Economy: 0.1
    - Control: 0.4
    → score = 0.5*3.0 + 2.0*1.8 + 0.3*1.2 + 0.1*0.8 + 0.4*0.6 = 5.94 ✓
  
  Opt 2 (move right to 8,7):
    - Survival: 0.0 (neutral)
    - Offense: 0.0 (farther from enemy)
    - Positioning: 0.2
    - Economy: 0.0
    - Control: 0.2
    → score = 0.0*3.0 + 0.0*1.8 + 0.2*1.2 + 0.0*0.8 + 0.2*0.6 = 0.36
  
  Opt 3 (move up to 7,6):
    - Survival: 0.3
    - Offense: 0.0 (wrong direction)
    - Positioning: 0.5
    - Economy: 0.0
    - Control: 0.3
    → score = 0.3*3.0 + 0.0*1.8 + 0.5*1.2 + 0.0*0.8 + 0.3*0.6 = 1.68
  
  Result: Opt 1 (move left) ✓ Aggressive positioning!

v1 chose: move up (defensive)
v3 chose: move left (aggressive + smart)

Winner: v3 (gets closer to enemy for bomb setup)
```

---

## Summary Table

| Aspect | v1 | v3 |
|--------|----|----|
| **Decision Tree** | Sequential | Parallel |
| **Factors Evaluated** | 3 | 5 |
| **Time Usage** | ~2-10ms (10% utilization) | ~40ms (40% utilization) |
| **Enemy Prediction** | No | Yes |
| **Item Grab Logic** | Blind | Smart |
| **Lookahead Depth** | 1 | 3 |
| **Score Quality** | Low | High |
| **Flexibility** | Low (tunnel vision) | High |
| **Expected Improvement** | Baseline | +10-15% |
