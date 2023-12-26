from agrl.rl.utils import scheduler

class Test:
    def __init__(self):
        self.sigma = 100
        self.q = 50

t = Test()
for _ in range(2):
    scheduler(t, ['sigma', 'q'], [0.5, 2])
print(t.sigma)
print(t.q)