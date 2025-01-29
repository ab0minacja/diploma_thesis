import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from itertools import product
import time


def start_algorithm():
    start_time = time.time()

    clean_data_path = "eclean_data_updated.csv"
    user_database_path = "user_database.csv"

    # Wczytanie plików
    clean_data = pd.read_csv(clean_data_path)
    user_database = pd.read_csv(user_database_path, header=0)

    user_database.columns = user_database.columns.str.strip()

    shared_columns = [col for col in user_database.columns if col in clean_data.columns]
    shared_columns.append("Exercise")  # Dodanie kolumny "Exercise" do analizy

    df = pd.read_csv(clean_data_path, usecols=shared_columns + ["Exercise Type ID", "Larger Body Parts"])
    df = df.dropna()

    X = df.drop(["Exercise", "Exercise Type ID", "Larger Body Parts"], axis=1)
    y = df["Exercise"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    unique_points = len(pd.DataFrame(X_scaled).drop_duplicates())
    n_clusters = 14
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)

    last_user = user_database.iloc[-1]
    common_columns = [col for col in X.columns if col in user_database.columns]
    new_user = last_user[common_columns].to_dict()

    dictionaries = {}
    dictionary_path = "dictionaries"
    for filename in os.listdir(dictionary_path):
        name, ext = os.path.splitext(filename)
        if ext == ".json":
            with open(os.path.join(dictionary_path, filename), "r") as file:
                dictionaries[name] = json.load(file)

    def encode_value(dictionary, value):
        return dictionary.get(value, 0)

    def generate_combinations(column_value):
        if pd.isna(column_value) or not isinstance(column_value, str):
            return [[]]  # Brak danych
        values = [v.strip() for v in column_value.split(",")]
        return [list(comb) for comb in product(values, repeat=1)]

    larger_body_parts_combinations = generate_combinations(new_user.get("Larger Body Parts", ""))
    equipment_required_combinations = generate_combinations(new_user.get("Equipment Required_y", ""))
    target_gender_combinations = generate_combinations(new_user.get("Target Gender", ""))
    time_per_workout_combinations = generate_combinations(new_user.get("Time Per Workout", ""))

    print("Rozpoczęto testowanie kombinacji...")
    results = []

    for lbp_comb in larger_body_parts_combinations:
        for eq_comb in equipment_required_combinations:
            for tg_comb in target_gender_combinations:
                for tpw_comb in time_per_workout_combinations:
                    # Tworzenie kodowanego użytkownika
                    new_user_coded = {col: 0 for col in X.columns}
                    for col in X.columns:
                        if col == "Larger Body Parts":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_lbp", {}), v) for v in lbp_comb)
                        elif col == "Equipment Required_y":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_ery", {}), v) for v in eq_comb)
                        elif col == "Target Gender":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_tg", {}), v) for v in tg_comb)
                        elif col == "Time Per Workout":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_dpw2", {}), v) for v in tpw_comb)
                        else:
                            new_user_coded[col] = encode_value(dictionaries.get(f"encode_{col.lower()}", {}), new_user.get(col, 0))

                    new_user_scaled = scaler.transform(pd.DataFrame([new_user_coded], columns=X.columns))
                    cluster = kmeans.predict(new_user_scaled)[0]
                    cluster_exercises = df.loc[kmeans.labels_ == cluster]

                    print(f"Cluster: {cluster}, Exercises Found: {len(cluster_exercises)}")

                    used_exercises_global = set()

                    days_per_week = int(new_user.get("Days Per Week", 3))
                    plan = []

                    for day in range(1, days_per_week + 1):
                        daily_exercises = []  
                        used_exercises_daily = set()  
                        
                        muscle_groups = [mg.strip() for mg in new_user.get("Larger Body Parts", "").split(",") if mg.strip()]
                        if not muscle_groups:
                            muscle_groups = df["Larger Body Parts"].unique().tolist()
                        
                        for mg in muscle_groups:
                            group_exercises = cluster_exercises[cluster_exercises["Larger Body Parts"] == mg]
                            for _, exercise in group_exercises.iterrows():
                                exercise_id = exercise["Exercise"]
                                if exercise_id not in used_exercises_daily and exercise_id not in used_exercises_global:
                                    daily_exercises.append(
                                        {
                                            "Exercise": exercise_id,
                                            "Muscle Group": mg
                                        }
                                    )
                                    used_exercises_daily.add(exercise_id)
                                    used_exercises_global.add(exercise_id)
                                
                                if len(daily_exercises) >= 6:  
                                    break
                            if len(daily_exercises) >= 6:
                                break
                        
                        plan.append({"Day": day, "Exercises": daily_exercises})

    with open(f"users/{last_user['Nickname']}.json", "w") as json_file:
        json.dump(plan, json_file, indent=4)

    print("Final plan:", plan)
start_algorithm()
