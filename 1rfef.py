# exporta_wyscout.py
import requests
import pandas as pd
import time

TOKEN = "7801baaf5aa45f9391aca25b7b7abd4618c83a5a"
BASE_URL = "https://searchapi.wyscout.com/api/v1/search/results.json"

params = {
    "search[women_mode]": "false",
    "search[time_frame]": "-5443",
    "search[competition]": "-602", #ID de la competición con el signo - delante(613:2RFEF, 602: 1RFEF)
    "search[youth_stats]": "false",
    "count": 100,
    "sort": "market_value desc",
    "language": "es",
    "token": TOKEN,
    "columns": ""
}

columns = [
    "name", "id", "image", "current_team_logo", "current_team_color", "birth_country_name",
    "passport_country_names", "current_team_name", "market_value", "total_matches",
    "minutes_on_field", "last_club_name", "positions", "age", "contract_expires",
    "goals", "xg_shot", "assists", "xg_assist", "duels_avg", "duels_won", "foot", "height",
    "weight", "on_loan", "successful_defensive_actions_avg", "defensive_duels_avg",
    "defensive_duels_won", "aerial_duels_avg", "aerial_duels_won", "tackle_avg",
    "possession_adjusted_tackle", "shot_block_avg", "interceptions_avg",
    "possession_adjusted_interceptions", "fouls_avg", "yellow_cards", "yellow_cards_avg",
    "red_cards", "red_cards_avg", "successful_attacking_actions_avg", "goals_avg",
    "non_penalty_goal", "non_penalty_goal_avg", "xg_shot_avg", "head_goals", "head_goals_avg",
    "shots", "shots_avg", "shots_on_target_percent", "goal_conversion_percent",
    "assists_avg", "crosses_avg", "accurate_crosses_percent", "cross_from_left_avg",
    "successful_cross_from_left_percent", "cross_from_right_avg",
    "successful_cross_from_right_percent", "cross_to_goalie_box_avg", "dribbles_avg",
    "successful_dribbles_percent", "offensive_duels_avg", "offensive_duels_won",
    "touch_in_box_avg", "progressive_run_avg", "accelerations_avg", "received_pass_avg",
    "received_long_pass_avg", "foul_suffered_avg", "passes_avg", "accurate_passes_percent",
    "forward_passes_avg", "successful_forward_passes_percent", "back_passes_avg",
    "successful_back_passes_percent", "vertical_passes_avg", "successful_vertical_passes_percent",
    "short_medium_pass_avg", "accurate_short_medium_pass_percent", "long_passes_avg",
    "successful_long_passes_percent", "average_pass_length", "average_long_pass_length",
    "xg_assist_avg", "shot_assists_avg", "pre_assist_avg", "pre_pre_assist_avg",
    "smart_passes_avg", "accurate_smart_passes_percent", "key_passes_avg",
    "passes_to_final_third_avg", "accurate_passes_to_final_third_percent",
    "pass_to_penalty_area_avg", "accurate_pass_to_penalty_area_percent", "through_passes_avg",
    "successful_through_passes_percent", "deep_completed_pass_avg", "deep_completed_cross_avg",
    "progressive_pass_avg", "successful_progressive_pass_percent", "conceded_goals",
    "conceded_goals_avg", "shots_against", "shots_against_avg", "clean_sheets", "save_percent",
    "xg_save", "xg_save_avg", "prevented_goals", "prevented_goals_avg", "back_pass_to_gk_avg",
    "goalkeeper_exits_avg", "gk_aerial_duels_avg", "free_kicks_taken_avg",
    "direct_free_kicks_taken_avg", "direct_free_kicks_on_target_percent", "corners_taken_avg",
    "penalties_taken", "penalties_conversion_percent"
]

params["columns"] = ",".join(columns)
headers = {"User-Agent": "Mozilla/5.0"}

all_players = []
print("🔍 Obteniendo la primera página...")
params["page"] = 0
response = requests.get(BASE_URL, headers=headers, params=params)

if response.status_code != 200:
    print(f"❌ Error HTTP {response.status_code}")
    print(response.text)
    exit(1)

try:
    data = response.json()
    if "players" not in data:
        print("⚠️ La respuesta no contiene la clave 'players'.\nContenido recibido:")
        print(data)
        exit(1)
    all_players.extend(data["players"])
    total_pages = data.get("meta", {}).get("page_count", 1)
    print(f"📄 Total de páginas detectadas: {total_pages}")
except Exception as e:
    print(f"❌ Error al analizar JSON: {e}")
    print(response.text)
    exit(1)

for page in range(1, total_pages):
    params["page"] = page
    print(f"⬇️ Descargando página {page + 1}/{total_pages}")
    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if "players" in data:
                all_players.extend(data["players"])
            else:
                print(f"⚠️ Página {page} sin 'players'. JSON recibido:")
                print(data)
        else:
            print(f"⚠️ Error HTTP {response.status_code} en página {page}")
    except Exception as e:
        print(f"❌ Error en página {page}: {e}")
    time.sleep(0.5)

print("👀 Ejemplo de jugadores descargados:")
for p in all_players[:10]:
    print(f"{p['name']} – {p.get('current_team_name')} – {p.get('domestic_competition_name')}")

if all_players:
    df = pd.DataFrame(all_players).fillna(0)
    if 'positions' in df.columns:
        df['positions'] = df['positions'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else x)
    df.to_csv("2RFEFv1.csv", index=False, encoding="utf-8", sep=";")
    print("✅ Exportación finalizada: '2RFEFv1.csv'")
else:
    print("⚠️ No se obtuvieron jugadores. No se exporta archivo.")