import requests
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

r = requests.get('https://ddragon.leagueoflegends.com/cdn/11.6.1/data/en_US/champion.json')
champion_data = r.json()['data']
champion_list = list(map(lambda x: x.replace(" ", "").lower(), champion_data.keys()))
print(champion_list)
columns = ['champion', 'position'] + champion_list

matchup_data = {}

path = 'D:\\lolproject\\opgg\\matchups.jl'
f = open(path, 'r')
for line in f:
    obj = json.loads(line)
    if obj['champion'] not in matchup_data:
        matchup_data[obj['champion']] = {}
    if obj['position'] not in matchup_data[obj['champion']]:
        matchup_data[obj['champion']][obj['position']] = {}
    matchup_data[obj['champion']][obj['position']][obj['matchup']] = obj['winrate']
f.close()

data = []
for champion in champion_list:
    if champion in matchup_data:
        for position in matchup_data[champion].keys():
            stats = ['0.5'] * len(champion_list)
            for matchup in matchup_data[champion][position]:
                if matchup == 'wukong':
                    opponent = 'monkeyking'
                elif matchup == 'cho\'gath':
                    opponent = 'chogath'
                elif matchup == 'nunu&willump':
                    opponent = 'nunu'
                elif matchup == 'vel\'koz':
                    opponent = 'velkoz'
                elif matchup == 'kai\'sa':
                    opponent = 'kaisa'
                elif matchup == 'kog\'maw':
                    opponent = 'kogmaw'
                elif matchup == 'kha\'zix':
                    opponent = 'khazix'
                elif matchup == 'rek\'sai':
                    opponent = 'reksai'
                elif matchup == 'dr.mundo':
                    opponent = 'drmundo'
                else:
                    opponent = matchup
                stats[champion_list.index(opponent)] = str(matchup_data[champion][position][matchup])
            row = [champion, position] + stats
            data.append(row)
df = pd.DataFrame(data, columns=columns)
table = pa.Table.from_pandas(df)
pq.write_table(table, 'winrate.parquet')