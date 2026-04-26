from graph_builder import build_graph
from graph_traversal import get_path, explain_path
from rl_bandit import Bandit
from learner_state import init_db, save

# Step 1: Build graph
G = build_graph()

# Step 2: Get path
path = get_path(G, "Calculus")

print("Graph-CoT Reasoning:")
explain_path(path)

# Step 3: RL
bandit = Bandit()
difficulty = bandit.choose()

print("Difficulty:", difficulty)

# Step 4: Update RL
reward = 0.8
bandit.update(difficulty, reward)

# Step 5: Save
init_db()
save("user1", "Calculus", reward)

print("Done")
