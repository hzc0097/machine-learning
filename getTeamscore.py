import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import matplotlib.pyplot as plt

# raw_df = pd.read_parquet("./challenger/challenger_combined.parquet", engine="pyarrow")
# winrate_df = pd.read_parquet("./opgg/winrate.parquet", engine="pyarrow")
# print(winrate_df)

# raw_columns = raw_df.columns.values
# df_champs = raw_df[['gameId', 'teamId', 'win', 'championId', 'role', 'lane']]

# champions = {}
# with open('champion.json', encoding="utf8") as f:
#     champ_data = json.load(f)
#     champ_data = champ_data['data']
#     for champion in champ_data:
#         champ_obj = champ_data[champion]
#         champions[champ_obj['key']] = champ_obj['id'].lower()

# def get_champion_name(championId):
#     if str(championId) in champions:
#         return champions[str(championId)]
#     return None
# df_champs['championName'] = df_champs['championId'].apply(get_champion_name)
# print(df_champs.role.unique())
# print(df_champs.lane.unique())
# print(winrate_df.position.unique())

# def convert_role(api_role, api_lane):
#     if api_lane == 'JUNGLE':
#         return 'jungle'
#     elif api_lane == 'TOP':
#         if api_role == 'SOLO':
#             return 'top'
#         else:
#             return 'jungle'
#     elif api_lane == 'MIDDLE':
#         return 'mid'
#     else:
#         if api_lane == 'BOTTOM':
#             if api_role == 'DUO_CARRY':
#                 return 'bot'
#             if api_role == 'DUO_SUPPORT':
#                 return 'support'
#     return None

# all_win_rates = []
# for first in range(0, len(df_champs), 10):
#     single_game = df_champs[first:first+10]
#     print(f'processing games {first} to {first+10}')
#     processed = [False] * 10
#     count = 0
#     win_rate = [0.5] * 10
#     for i in range(len(single_game)):
#         row = single_game.iloc[i]
#         own_champ_name = row['championName']
#         own_position = convert_role(row['role'], row['lane'])
#         if processed[count]:
#             continue
#         if row['lane'] == 'NONE':
#             win_rate[count] = np.nan
#             processed[count] = True
#         else:
#             for j in range(count + 1, 10):
#                 opponent_position = convert_role(single_game.iloc[j]['role'], single_game.iloc[j]['lane'])
#                 if own_position == opponent_position:
#                     opponent_champ_name = single_game.iloc[j]['championName']
#                     own_winrate = winrate_df[(winrate_df['champion'] == own_champ_name) & (winrate_df['position'] == own_position)]
#                     if len(own_winrate) > 0:
#                         win_rate[count] = own_winrate.iloc[0][opponent_champ_name]
#                         processed[count] = True
#                     opponent_winrate = winrate_df[(winrate_df['champion'] == opponent_champ_name) & (winrate_df['position'] == opponent_position)]
#                     if len(opponent_winrate) > 0:
#                         win_rate[j] = opponent_winrate.iloc[0][own_champ_name]
#                         processed[j] = True
#                     break
#         count += 1
#     all_win_rates.extend(win_rate)

# df_champs['winRate'] = all_win_rates
# df_champs.dropna()
# print(df_champs.info())
# df_champs['winRate'] = df_champs['winRate'].astype(str)
# table = pa.Table.from_pandas(df_champs)
# pq.write_table(table, 'winrate_inserted.parquet')
old_data = pd.read_parquet("winrate_inserted.parquet", engine="pyarrow")
print(old_data.info())
old_data['winRate'] = old_data['winRate'].astype(float)
old_data.hist(column='winRate', bins=100)
plt.savefig('winrate_distribution.png')
plt.show()

old_data['zscore'] = (old_data.winRate - old_data.winRate.mean())/old_data.winRate.std(ddof=0)
old_data.hist(column='zscore', bins=100)
plt.savefig('zscore_distribution.png')
plt.show()
# table = pa.Table.from_pandas(old_data)
# pq.write_table(table, 'winrate_zscore.parquet')