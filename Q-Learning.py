
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define Grid Dimensions
GRID_ROWS = 4
GRID_COLS = 4

# Special States & Rewards
START = (3, 0)
GOAL = (0, 3)
WALL = (1, 1)

REWARD_GOAL = 10.0
REWARD_WALL = -1.0
REWARD_DEFAULT = 0.0

# Hyperparameters
ALPHA = 0.5   # Learning rate
GAMMA = 0.9   # Discount factor
EPISODES = 2000

# Actions: 0: Up, 1: Down, 2: Left, 3: Right
ACTIONS = ['Up', 'Down', 'Left', 'Right']
ACTION_MOVES = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Left': (0, -1),
    'Right': (0, 1)
}

#  Environment Step Function

def step(state, action_idx):
    if state == GOAL:
        return state, 0, True

    action_name = ACTIONS[action_idx]
    dr, dc = ACTION_MOVES[action_name]
    next_r = state[0] + dr
    next_c = state[1] + dc

    # Check boundary collision
    if next_r < 0 or next_r >= GRID_ROWS or next_c < 0 or next_c >= GRID_COLS:
        next_state = state
    else:
        next_state = (next_r, next_c)

    # Determine reward and terminal condition
    if next_state == WALL:
        reward = REWARD_WALL
        is_terminal = False
        next_state = state
    elif next_state == GOAL:
        reward = REWARD_GOAL
        is_terminal = True
    else:
        reward = REWARD_DEFAULT
        is_terminal = False

    return next_state, reward, is_terminal

#  Q-Learning Algorithm Implementation

def run_q_learning():
    # Initialize Q-table with zeros: shape (4, 4, 4)
    q_table = np.zeros((GRID_ROWS, GRID_COLS, len(ACTIONS)))

    for episode in range(EPISODES):
        state = START
        is_terminal = False
        steps = 0

        while not is_terminal and steps < 100:
            steps += 1

            # Epsilon-greedy action selection
            epsilon = max(0.1, 1.0 - episode / 1000)
            if np.random.rand() < epsilon:
                action_idx = np.random.choice(len(ACTIONS))
            else:
                action_idx = np.argmax(q_table[state[0], state[1]])

            next_state, reward, is_terminal = step(state, action_idx)

            # Target Q-value calculation
            if is_terminal:
                max_next_q = 0.0
            else:
                max_next_q = np.max(q_table[next_state[0], next_state[1]])

            td_target = reward + GAMMA * max_next_q
            td_error = td_target - q_table[state[0], state[1], action_idx]

            # Q-value update rule
            q_table[state[0], state[1], action_idx] += ALPHA * td_error

            state = next_state

    return q_table

# Run training
q_table = run_q_learning()

#  Display Optimal Q-Table

print("\n--- Q-Table ---")

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        print(f"State ({r}, {c}): {q_table[r, c]}")

#  Extract Optimal Policy

policy = np.full((GRID_ROWS, GRID_COLS), '', dtype=object)

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if (r, c) == WALL:
            policy[r, c] = 'WALL'
        elif (r, c) == GOAL:
            policy[r, c] = 'GOAL'
        else:
            best_action = np.argmax(q_table[r, c])
            policy[r, c] = ACTIONS[best_action]

print("\n--- Optimal Policy ---")
print(pd.DataFrame(policy))

#  Generate Optimal Path
state = START
path = [state]
visited = set()
max_steps = GRID_ROWS * GRID_COLS * 4

for _ in range(max_steps):
    if state == GOAL:
        break

    if state in visited:
        print("Loop detected while generating path.")
        break

    visited.add(state)
    best_action = np.argmax(q_table[state[0], state[1]])
    next_state, reward, is_terminal = step(state, best_action)

    if next_state == state and not is_terminal:
        # Choose the next-best action if the best action does not move
        action_values = np.argsort(q_table[state[0], state[1]])[::-1]
        moved = False

        for action_idx in action_values:
            candidate_state, candidate_reward, candidate_terminal = step(state, action_idx)
            if candidate_state != state or candidate_terminal:
                next_state = candidate_state
                is_terminal = candidate_terminal
                break

    if next_state == state:
        break

    state = next_state
    path.append(state)

print("\n--- Optimal Path ---")
print(path)

#  Visualize Grid World and Optimal Path

grid = np.zeros((GRID_ROWS, GRID_COLS))

for r, c in path:
    grid[r, c] = 1

plt.figure(figsize=(7, 7))
plt.imshow(grid, cmap='Blues')

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if (r, c) == START:
            label = 'START'
        elif (r, c) == GOAL:
            label = 'GOAL'
        elif (r, c) == WALL:
            label = 'WALL'
        elif (r, c) in path:
            label = 'PATH'
        else:
            label = ''

        plt.text(c, r, label, ha='center', va='center', color='black')

plt.xticks(range(GRID_COLS))
plt.yticks(range(GRID_ROWS))
plt.title('Q-Learning Grid World Optimal Path')
plt.grid(True)
plt.show()

# Display Optimal Actions

print("\n--- Optimal Actions for Each State ---")

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if (r, c) == WALL:
            print(f"State ({r}, {c}): WALL")
        elif (r, c) == GOAL:
            print(f"State ({r}, {c}): GOAL")
        else:
            best_action = np.argmax(q_table[r, c])
            print(f"State ({r}, {c}): {ACTIONS[best_action]}")