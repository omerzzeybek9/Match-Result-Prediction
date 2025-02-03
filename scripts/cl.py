import requests
import pandas as pd

api_key = '5c941910aa9347c89d1bde6f3fc673bb'
url = 'http://api.football-data.org/v4/competitions/CL/standings'

headers = {
    'X-Auth-Token': api_key
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    teams = data['standings'][0]['table']
    
    team_data = []
    
    for team in teams:
        team_info = {
            "position": team['position'],
            "team": team['team']['name'],
            "played": team['playedGames'],
            "won": team['won'],
            "lost": team['lost'],
            "points": team['points'],
            "goals_for": team['goalsFor'],
            "goals_against": team['goalsAgainst'],
            "goal_difference": team['goalDifference'],
            "avg_goals_scored": team['goalsFor'] / team['playedGames'] if team['playedGames'] > 0 else 0,
            "avg_goals_conceded": team['goalsAgainst'] / team['playedGames'] if team['playedGames'] > 0 else 0
        }
        team_data.append(team_info)
    
    df = pd.DataFrame(team_data)
    

    df.to_csv("initial_data/cl/champions_league_standings.csv", index=False)
else:
    print(f"Error! Status Code: {response.status_code}")
    print(response.text)
