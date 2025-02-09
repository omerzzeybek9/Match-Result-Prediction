import requests
import pandas as pd

api_key = '5c941910aa9347c89d1bde6f3fc673bb'
standings_url = 'http://api.football-data.org/v4/competitions/PL/standings'
matches_url = 'http://api.football-data.org/v4/competitions/PL/matches'
headers = {'X-Auth-Token': api_key}

response = requests.get(standings_url, headers=headers)
if response.status_code == 200:
    data = response.json()
    teams = data['standings'][0]['table']
    
    standings_data = []
    for team in teams:
        played_games = team['playedGames'] if team['playedGames'] > 0 else 1
        won = team['won']
        lost = team['lost']
        points = team['points']
        goal_difference = team['goalDifference']
        goals_for = team['goalsFor']
        goals_against = team['goalsAgainst']
        
        power_score = ((won / played_games) * 50) + (goal_difference * 2) + ((points / played_games) * 10)

        standings_data.append({
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
        })

    df_standings = pd.DataFrame(standings_data)
else:
    print(f"Standings API Error! Status Code: {response.status_code}")
    print(response.text)
    exit()

response = requests.get(matches_url, headers=headers)
if response.status_code == 200:
    data = response.json()
    matches = data['matches']

    team_stats = {}

    for match in matches:
        if match['status'] != 'FINISHED': 
            continue

        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        home_goals = match['score']['fullTime']['home']
        away_goals = match['score']['fullTime']['away']

        if home_team not in team_stats:
            team_stats[home_team] = {"Home Played": 0, "Home Wins": 0, "Home Losses": 0, "Home GF": 0, "Home GA": 0,
                                     "Away Played": 0, "Away Wins": 0, "Away Losses": 0, "Away GF": 0, "Away GA": 0}
        if away_team not in team_stats:
            team_stats[away_team] = {"Home Played": 0, "Home Wins": 0, "Home Losses": 0, "Home GF": 0, "Home GA": 0,
                                     "Away Played": 0, "Away Wins": 0, "Away Losses": 0, "Away GF": 0, "Away GA": 0}

        team_stats[home_team]["Home Played"] += 1
        team_stats[home_team]["Home GF"] += home_goals
        team_stats[home_team]["Home GA"] += away_goals
        if home_goals > away_goals:
            team_stats[home_team]["Home Wins"] += 1
        elif home_goals < away_goals:
            team_stats[home_team]["Home Losses"] += 1

        team_stats[away_team]["Away Played"] += 1
        team_stats[away_team]["Away GF"] += away_goals
        team_stats[away_team]["Away GA"] += home_goals
        if away_goals > home_goals:
            team_stats[away_team]["Away Wins"] += 1
        elif away_goals < home_goals:
            team_stats[away_team]["Away Losses"] += 1

    df_home_away = pd.DataFrame.from_dict(team_stats, orient='index').reset_index()
    df_home_away.rename(columns={'index': 'Team'}, inplace=True)
else:
    print(f"Matches API Error! Status Code: {response.status_code}")
    print(response.text)
    exit()

df_combined = pd.merge(df_standings, df_home_away, on="Team", how="left")

df_combined.to_csv("initial_data/leagues/premier_league_stats.csv", index=False)

