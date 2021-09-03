import pickle

with open('ResultNN.txt', 'rb') as f:
    log1 = pickle.load(f)
f.close()
with open('ResultNN2.txt', 'rb') as f:
    log2 = pickle.load(f)
f.close()
with open('ResultNN3.txt', 'rb') as f:
    log3 = pickle.load(f)
f.close()

final = [log1, log2, log3]
res0 = log1[0]
accs = []
time1 = []
time2 = []
for log in final:
    accs.append(log[1])
    time1.append(log[2])
    time2.append(log[3])

res1 = [sum(e)/len(e) for e in zip(*accs)]
res2 = [sum(e)/len(e) for e in zip(*time1)]
res3 = [sum(e)/len(e) for e in zip(*time2)]

result = [res0, res1, res2, res3]

with open("ResultNeural.txt", 'wb') as f:
    pickle.dump(result, f)
f.close()
