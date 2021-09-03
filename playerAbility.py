import pandas as pd
import numpy as np
import json
import pickle
from playerDict import playerDict as playDict

with open('weightOutput.txt', 'rb') as f:
    weightList = pickle.load(f)
f.close()

df = pd.read_csv('combinedEdited.csv')
df.drop(['win'],1,inplace=True)
noName = df.drop(['summonerName'],1)

scoreDict = playDict()

for j in range(len(df)):
    thisGamePlayerScore = 0
    name = df.loc[j][0]
    for i in range(len(weightList)):
        if (weightList[i] < 0):
            continue
        else:
            thisGamePlayerScore += weightList[i] * noName.loc[j][i]
    if name not in scoreDict:
        scoreDict.addPlayer(name, thisGamePlayerScore)
    else:
        scoreDict.incrementPlayer(name, thisGamePlayerScore)
print(scoreDict)
with open('playerScores.txt', 'wb') as f:
    pickle.dump(scoreDict, f)
f.close()
with open('playerScoresPlain.txt', 'w') as f:
    f.write(json.dumps(scoreDict))
f.close()







