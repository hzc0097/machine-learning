class playerDict(dict):
    def __init__(self):
        self = dict()

    def addPlayer(self, name, score):
        self.setdefault(name, [])
        self[name].append(score)
        self[name].append(1)

    def incrementPlayer(self, name, score):
        self[name][0] = ((self[name][0] * self[name][1]) + score) / (self[name][1] + 1)
        self[name][1] += 1