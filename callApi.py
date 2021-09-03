from riotwatcher import LolWatcher, ApiError
import requests
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import math

endpoint = "https://na1.api.riotgames.com"
API_KEY = "RGAPI-467e2119-8010-4936-8b75-9bacc1651417"
lol_watcher = LolWatcher(API_KEY)

my_region = "na1"

# me = lol_watcher.summoner.by_name(my_region, "pseudonym117")
# print(me)
# url = "https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/Doublelift?api_key=" + API_KEY
# response = requests.get(url)
# data = response.json()
# print(data)

def get_challenger_players_info():
    challengers = lol_watcher.league.challenger_by_queue(my_region, "RANKED_SOLO_5x5")
    print("Total " + str(len(challengers["entries"])) + " players")
    challenger_accounts = []
    for player in challengers["entries"]:
        print("Getting info for " + player["summonerName"])
        challenger_accounts.append({
            "account_id": lol_watcher.summoner.by_id(my_region, player["summonerId"])["accountId"],
            "summoner_name": player["summonerName"]
        })
    return challenger_accounts

columns = [
    "gameId",
    "gameDuration",
    "summonerName",
    "teamId",
    "win",
    "firstBlood",
    "firstTower",
    "firstInhibitor",
    "firstBaron",
    "firstDragon",
    "firstRiftHerald",
    "teamTowerKills",
    "teamInhibitorKills",
    "baronKills",
    "dragonKills",
    "riftHeraldKills",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
    "championId",
    "spell1Id",
    "spell2Id",
    "role",
    "lane",
]
stats_fields = [
    "item0",
    "item1",
    "item2",
    "item3",
    "item4",
    "item5",
    "item6",
    "kills",
    "deaths",
    "assists",
    "largestKillingSpree",
    "largestMultiKill",
    "killingSprees",
    "longestTimeSpentLiving",
    "doubleKills",
    "tripleKills",
    "quadraKills",
    "pentaKills",
    "totalDamageDealt",
    "magicDamageDealt",
    "physicalDamageDealt",
    "trueDamageDealt",
    "largestCriticalStrike",
    "totalDamageDealtToChampions",
    "magicDamageDealtToChampions",
    "physicalDamageDealtToChampions",
    "trueDamageDealtToChampions",
    "totalHeal",
    "totalUnitsHealed",
    "damageSelfMitigated",
    "damageDealtToObjectives",
    "damageDealtToTurrets",
    "visionScore",
    "timeCCingOthers",
    "totalDamageTaken",
    "magicalDamageTaken",
    "physicalDamageTaken",
    "trueDamageTaken",
    "goldEarned",
    "goldSpent",
    "turretKills",
    "inhibitorKills",
    "totalMinionsKilled",
    "neutralMinionsKilled",
    "neutralMinionsKilledTeamJungle",
    "neutralMinionsKilledEnemyJungle",
    "totalTimeCrowdControlDealt",
    "champLevel",
    "visionWardsBoughtInGame",
    "sightWardsBoughtInGame",
    "wardsPlaced",
    "wardsKilled",
    "firstBloodKill",
    "firstBloodAssist",
    "firstTowerKill",
    "firstTowerAssist",
    "firstInhibitorKill",
    "firstInhibitorAssist",
    "perk0",
    "perk0Var1",
    "perk0Var2",
    "perk0Var3",
    "perk1",
    "perk1Var1",
    "perk1Var2",
    "perk1Var3",
    "perk2",
    "perk2Var1",
    "perk2Var2",
    "perk2Var3",
    "perk3",
    "perk3Var1",
    "perk3Var2",
    "perk3Var3",
    "perk4",
    "perk4Var1",
    "perk4Var2",
    "perk4Var3",
    "perk5",
    "perk5Var1",
    "perk5Var2",
    "perk5Var3",
    "perkPrimaryStyle"
    "perkSubStyle",
    "statPerk0",
    "statPerk1",
    "statPerk2",
]
columns.extend(stats_fields)
timeline_fields = [
    "creepsPerMinDeltas",
    "xpPerMinDeltas",
    "goldPerMinDeltas",
    "csDiffPerMinDeltas",
    "xpDiffPerMinDeltas",
    "damageTakenPerMinDeltas",
    "damageTakenDiffPerMinDeltas",
]
possible_times = ["0-10", "10-20", "20-30", "30-end"]
for fields in timeline_fields:
    for time_range in possible_times:
        columns.append(fields + time_range) 

def get_player_matches(account_info, begin_index=0):
    old_data = None
    game_count = 0
    if begin_index != 0:
        old_data = pd.read_parquet("./challenger/" + account_info["account_id"] + ".parquet", engine="pyarrow")
        game_count = len(old_data.index) / 10
    game_stats = []
    try:
        while game_count < 500:
            print("Getting match list for account " + str(account_info) + ", beginIndex = " + str(begin_index))
            match_list = lol_watcher.match.matchlist_by_account(my_region, account_info["account_id"], begin_index=begin_index)
            is_target_season = True
            for match in match_list["matches"]:
                if match["lane"] != "NONE":
                    game_version, match_data = get_match_information(match["gameId"])
                    if int(game_version.split(".")[0]) < 10:
                        is_target_season = False
                        break
                    if match_data != None:
                        game_stats += match_data
                        game_count += 1
                begin_index += 1
            if not is_target_season:
                break
        f2 = open("last_match.pckl", "rb")
        resume_info = pickle.load(f2)
        f2.close()
        prev_error_data_idx = None
        for idx, val in enumerate(resume_info):
            if val["account_info"] == account_info:
                prev_error_data_idx = idx
                break
        if prev_error_data_idx != None:
            del resume_info[prev_error_data_idx]
        f = open("last_match.pckl", "wb")
        pickle.dump(resume_info, f)
        f.close()
        df = pd.DataFrame(game_stats, columns=columns)
        if old_data is not None:
            df = pd.concat([old_data, df])
        return df
    except Exception as e:
        print(e)
        f2 = open("last_match.pckl", "rb")
        resume_info = pickle.load(f2)
        f2.close()
        prev_error_data_idx = None
        for idx, val in enumerate(resume_info):
            if val["account_info"] == account_info:
                prev_error_data_idx = idx
                break
        if prev_error_data_idx == None:
            resume_info.append({
                "account_info": account_info,
                "begin_index": begin_index,
                "error_msg": e,
            })
        else:
            resume_info[prev_error_data_idx] = {
                "account_info": account_info,
                "begin_index": begin_index,
                "error_msg": e,
            }
        f = open("last_match.pckl", "wb")
        pickle.dump(resume_info, f)
        f.close()
        df = pd.DataFrame(game_stats, columns=columns)
        if old_data is not None:
            df = pd.concat([old_data, df])
        return df

def get_match_information(game_id):
    print("Getting match info for gameId " + str(game_id))
    match = lol_watcher.match.by_id(my_region, game_id)
    patch_no = match["gameVersion"]
    if match["queueId"] != 420 and match["queueId"] != 440:
        return (patch_no, None)
    participantIdentities = match["participantIdentities"]
    game_info = []
    for player in match["participants"]:
        team_info = match["teams"][0 if player["teamId"] == 100 else 1] 
        row = [
            match["gameId"],
            match["gameDuration"],
            participantIdentities[int(player["participantId"]) - 1]["player"]["summonerName"],
            player["teamId"],
            1 if team_info["win"] == "Win" else 0,
            int(team_info["firstBlood"]),
            int(team_info["firstTower"]),
            int(team_info["firstInhibitor"]),
            int(team_info["firstBaron"]),
            int(team_info["firstDragon"]),
            int(team_info["firstRiftHerald"]),
            team_info["towerKills"],
            team_info["inhibitorKills"],
            team_info["baronKills"],
            team_info["dragonKills"],
            team_info["riftHeraldKills"],
            team_info["bans"][0]["championId"],
            team_info["bans"][1]["championId"],
            team_info["bans"][2]["championId"],
            team_info["bans"][3]["championId"],
            team_info["bans"][4]["championId"],
            player["championId"],
            player["spell1Id"],
            player["spell2Id"],
            player["timeline"]["role"],
            player["timeline"]["lane"],
        ]
        # stats
        for field in stats_fields:
            if field in player["stats"]:
                row.append(player["stats"][field])
            else:
                row.append(None)
        # timeline
        for field in timeline_fields:
            field_data = []
            if field in player["timeline"]:
                for time_range in possible_times:
                    if time_range in player["timeline"][field]:
                        field_data.append(player["timeline"][field][time_range])
                    else:
                        field_data.append(None)
            else:
                for time_range in possible_times:
                    field_data.append(None)
            row.extend(field_data)
        game_info.append(row)
    return (patch_no, game_info)

def write_to_parquet(data, filename):
    table = pa.Table.from_pandas(data)
    pq.write_table(table, filename)

# challenger_players = get_challenger_players_info()
# f = open("na_challenger_players.pckl", "wb")
# pickle.dump(challenger_players, f)
# f.close()

# resume_info = []
# f = open("last_match.pckl", "wb")
# pickle.dump(resume_info, f)
# f.close()

f1 = open("na_challenger_players.pckl", "rb")
f2 = open("last_match.pckl", "rb")
resume_info = pickle.load(f2)
f2.close()
if len(resume_info) > 0:
    for idx, val in enumerate(resume_info):
        print(f"Resume info total players: {len(resume_info)}, Begin player {str(idx + 1)}")
        if "error_msg" in val:
            if val["error_msg"].response.status_code != 404:
                data = get_player_matches(val["account_info"], val["begin_index"])
                print(data)
                if not data.empty:
                    write_to_parquet(data, "./challenger/" + val["account_info"]["account_id"] + ".parquet")
            else:
                print("Skipped due to 404")
        else:
            data = get_player_matches(val["account_info"], val["begin_index"])
            print(data)
            if not data.empty:
                write_to_parquet(data, "./challenger/" + val["account_info"]["account_id"] + ".parquet")
else:
    challenger_players = pickle.load(f1)
    f1.close()
    for idx, val in enumerate(challenger_players):
        print(f"Total players: {len(challenger_players)}, Begin player {str(idx + 1)}")
        data = get_player_matches(val)
        print(data)
        if not data.empty:
            write_to_parquet(data, "./challenger/" + val["account_id"] + ".parquet")