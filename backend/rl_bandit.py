class Bandit:
    def __init__(self):
        self.actions = ["easy", "medium", "hard"]
        self.values = {"easy": 0, "medium": 0, "hard": 0}

    def choose(self):
        return max(self.values, key=self.values.get)

    def update(self, action, reward):
        self.values[action] += reward
