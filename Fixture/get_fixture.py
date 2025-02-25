import requests
import pandas as pd


API_KEY = "api_key"

LEAGUE_ID = 2001

url = f"https://api.football-data.org/v4/competitions/{LEAGUE_ID}/matches"

headers = {"X-Auth-Token": API_KEY}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    
    matches = []
    for match in data["matches"]:
        matches.append({
            "Matchday": match["matchday"],
            "Date": match["utcDate"],
            "Home Team": match["homeTeam"]["name"],
            "Away Team": match["awayTeam"]["name"],
            "Home Score": match["score"]["fullTime"]["home"],
            "Away Score": match["score"]["fullTime"]["away"]
        })

    df_fixtures = pd.DataFrame(matches)
    df_fixtures.to_csv("fixtures_cl.csv", index=False)
    print(df_fixtures.head())  

else:
    print(f"Hata! Status Code: {response.status_code}")
    print(response.text)