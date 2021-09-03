import pandas as pd
import pickle

with open('playerScores.txt', 'rb') as f:
    scoreDict = pickle.load(f)
f.close()
df = pd.DataFrame.from_dict(scoreDict)
print(df)
df = df.T
print(df)
df.columns = ['overallScore', 'totalGamesPlayed']
df.to_csv(r'playerAbilityScores.csv', index = True, header=True)
df.to_parquet(r'playerAbilityScore.parquet', index=True)
