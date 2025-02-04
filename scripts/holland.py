import requests
import pandas as pd

# API Bilgileri
api_key = '5c941910aa9347c89d1bde6f3fc673bb'
url = 'http://api.football-data.org/v4/competitions/DED/standings'

headers = {'X-Auth-Token': api_key}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    teams = data['standings'][0]['table']

    team_data = []
    
    for team in teams:
        played_games = team['playedGames'] if team['playedGames'] > 0 else 1
        won = team['won']
        lost = team['lost']
        points = team['points']
        goal_difference = team['goalDifference']
        goals_for = team['goalsFor']
        goals_against = team['goalsAgainst']
        
        power_score = ((won / played_games) * 50) + (goal_difference * 2) + ((points / played_games) * 10)

        team_info = {
            "Position": team['position'],
            "Team": team['team']['name'],
            "Played": played_games,
            "Won": won,
            "Lost": lost,
            "Points": points,
            "Goals For": goals_for,
            "Goals Against": goals_against,
            "Goal Difference": goal_difference,
            "Power Score": round(power_score, 2)
        }
        team_data.append(team_info)

    df = pd.DataFrame(team_data)

    df = df.sort_values(by="Power Score", ascending=False).reset_index(drop=True)

    df.to_csv("initial_data/leagues/eredivise_standings.csv", index=False)
else:
    print(f"Error! Status Code: {response.status_code}")
    print(response.text)
