import pandas as pd

PAS = pd.read_csv('playerAbilityScores.csv')
DB = pd.read_parquet('biggerCombined.parquet')

sum = 0
for i in range(len(PAS)):
    sum += PAS['overallScore'][i]
avgScore = sum/len(PAS)
print(f"avg Score: {avgScore}")

print(len(DB))

print(DB.loc[DB['summonerName']=='Hudsenne'])

for i in range(len(PAS)):
    df = DB.loc[DB['summonerName']==PAS['summonerName'][i]]
    DB.loc[DB['summonerName']==PAS['summonerName'][i], 'PAS'] = PAS['overallScore'][i]
    print(i)

print(DB)

DB.to_parquet('biggerCombinedPAS2.parquet')