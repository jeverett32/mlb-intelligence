import db
import requests
import pandas as pd

def find_missing():
    df_db = db.get_games_df(season=2026)
    pks_db = set(df_db[df_db['home_win'].notna()]['game_pk'])
    
    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2026&gameType=R"
    data = requests.get(url).json()
    
    missing = []
    for date_info in data.get('dates', []):
        for g in date_info.get('games', []):
            if g['status']['abstractGameState'] == 'Final' and g['gamePk'] not in pks_db:
                missing.append({
                    'game_pk': g['gamePk'],
                    'game_date': date_info['date'],
                    'away_team': g['teams']['away']['team']['name'],
                    'home_team': g['teams']['home']['team']['name'],
                    'status': g['status']['detailedState'],
                    'away_score': g['teams']['away'].get('score'),
                    'home_score': g['teams']['home'].get('score')
                })
                
    return missing

missing = find_missing()
print(f"Found {len(missing)} missing games:")
for m in missing:
    print(f"{m['game_date']} | {m['game_pk']} | {m['away_team']} @ {m['home_team']} | {m['status']} | {m['away_score']}-{m['home_score']}")
