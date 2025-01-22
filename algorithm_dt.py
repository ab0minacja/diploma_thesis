import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import os
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
def start_algorithm():
    clean_data_path = "eclean_data_updated.csv"
    user_database_path = "user_database.csv"

    # Wczytanie plików
    clean_data = pd.read_csv(clean_data_path)
    user_database = pd.read_csv(user_database_path, header=0)
 

    # Usunięcie spacji z nazw kolumn w user_database
    user_database.columns = user_database.columns.str.strip()

    # Znalezienie wspólnych kolumn między clean_data i user_database
    shared_columns = [col for col in user_database.columns if col in clean_data.columns]
    print(shared_columns)
    # Dodanie kolumny "Exercise" do clean_data
    shared_columns.append("Exercise")

    # Wczytanie clean_data tylko z wybranymi kolumnami
    df = pd.read_csv(clean_data_path, usecols=shared_columns)
    df = df.dropna()

    # Podział na cechy i target
    X = df.drop("Exercise", axis=1)
    y = df["Exercise"]

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Trenowanie modelu
    model = RandomForestClassifier(n_estimators=30, max_depth=44, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    '''recall =[]
    accuracy =[]
    f1_s =[]
    precision =[]
    n = 26
    for _ in range(n-1):
        sample_indices = np.random.choice(len(y_test), size=100, replace=False) 
        y_test_sampled = np.array(y_test)[sample_indices]
        y_pred_sampled = np.array(y_pred)[sample_indices]
        
        accuracy.append(accuracy_score(y_test_sampled, y_pred_sampled))
        f1_s.append(f1_score(y_test_sampled, y_pred_sampled, average='macro'))
        precision.append(precision_score(y_test_sampled, y_pred_sampled, average='macro'))
        recall.append(recall_score(y_test_sampled, y_pred_sampled, average='macro'))

    print("Dokładność modelu:", sum(accuracy)/len(accuracy))
    print("Miara F1 modelu:", sum(f1_s)/len(f1_s))
    print("Miara precyzji:", sum(precision)/len(precision))
    print("Miara czułości", sum(recall)/len(recall))

    plt.figure(figsize=(10, 6))
    # Wykresy dla każdej z miar
    plt.plot(range(1, n), accuracy, label='Miara dokładności', color='blue')
    plt.plot(range(1, n), f1_s, label='Miara F1', color='green')
    plt.plot(range(1, n), precision, label='Miara precyzji', color='red')
    plt.plot(range(1, n), recall, label='Miara czułości', color='purple')

    # Dodanie etykiet i tytułu
    plt.xlabel('Numer iteracji',fontsize=14)
    plt.ylabel('Uzyskana ocena',fontsize=14)
    plt.title('Przebieg zmian ocen jakości modelu uzyskany za pomocą różnych miar dla poszczególnych iteracji')
    plt.legend()

    # Pokaż wykres
    plt.show()
'''
    # Wczytanie danych nowego użytkownika
    last_user = user_database.iloc[-1]
    common_columns = [col for col in X.columns if col in user_database.columns]
    new_user = last_user[common_columns].to_dict()
    nickname = last_user["Nickname"]
    # Wczytanie słowników kodowania
    dictionaries = {}
    dictionary_path = "dictionaries"
    for filename in os.listdir(dictionary_path):
        name, ext = os.path.splitext(filename)
        if ext == ".json":
            with open(os.path.join(dictionary_path, filename), "r") as file:
                dictionaries[name] = json.load(file)

    # Funkcja do kodowania danych użytkownika
    def encode_value(dictionary, value):
        return dictionary.get(value, 0)
    #mapping values 
    if new_user["Target Gender"] == "Male":
        new_user["Target Gender"] = 'Male, Other'
    elif new_user["Target Gender"] == "Female":
        new_user["Target Gender"] = 'Female, Other'
    else:
        new_user["Target Gender"] = "Other"

    if new_user["Equipment Required_y"] == "None":
        new_user["Equipment Required_y"] = "Bodyweight"
    elif new_user["Equipment Required_y"] == "All listed":
        new_user["Equipment Required_y"] == 'Barbell,EZ Bar,Dumbbell,Machine,Bodyweight,Cable,Exercise Ball,Bench,Other,Bands,Chains,Kettle Bells,Isolation,Tire,Box,Landmine,Trap Bar,Jump Rope'
    else:
        pass

    if new_user["Time Per Workout"] == "One hour":
        new_user["Time Per Workout"] = "Less than hour,One hour"
    elif new_user["Time Per Workout"] == "One and a half hours":
        new_user["One and a half hours"] = "Less than hour,One hour,One and a half hours"
    elif new_user["Time Per Workout"] == "Two hours and more":
        new_user["Two hours and more"] = "Less than hour,One hour,One and a half hours,Two hours and more'"
    else:
        pass

    if new_user["Larger Body Parts"] == "Full Body":
        new_user["Larger Body Parts"] = 'Chest&Shoulders,Back,Arms,Core,Legs,Glutes'
    else: pass
    print(new_user["Target Gender"])
    def generate_combinations(column_value):
        # Dzieli wartości rozdzielone przecinkiem i tworzy listę unikalnych kombinacji
        if pd.isna(column_value) or not isinstance(column_value, str):
            return [[]]  # Brak danych
        values = [v.strip() for v in column_value.split(",")]
        return [list(comb) for comb in product(values, repeat=1)]

    # Testowanie różnych kombinacji
    larger_body_parts_combinations = generate_combinations(new_user.get("Larger Body Parts", ""))
    equipment_required_combinations = generate_combinations(new_user.get("Equipment Required_y", ""))
    target_gender_combinations = generate_combinations(new_user.get("Target Gender", ""))
    time_per_workout_combinations = generate_combinations(new_user.get("Time Per Workout", ""))
    print("Rozpoczęto testowanie kombinacji...")
    results = []
    exercise_to_muscle = dictionaries.get("encode_etm", {})

    for lbp_comb in larger_body_parts_combinations:
        for eq_comb in equipment_required_combinations:
            for tg_comb in target_gender_combinations:
                for tpw_comb in time_per_workout_combinations:
                    # Tworzenie kodowanego użytkownika
                    new_user_coded = {}
                    for col in X_train.columns:
                        if col == "Larger Body Parts":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_lbp", {}), v) for v in lbp_comb)
                        elif col == "Equipment Required_y":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_ery", {}), v) for v in eq_comb)
                        elif col == "Target Gender":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_tg", {}), v) for v in tg_comb)
                        elif col == "Time Per Workout":
                            new_user_coded[col] = sum(encode_value(dictionaries.get("encode_dpw2", {}), v) for v in tpw_comb)
                        elif col == "Experience Level":
                            new_user_coded[col] = encode_value(dictionaries.get("encode_el", {}), new_user.get(col, 0))
                        elif col == "Main Goal":
                            new_user_coded[col] = encode_value(dictionaries.get("encode_mg", {}), new_user.get(col, 0))
                        elif col == "Days Per Week":
                            new_user_coded[col] = encode_value(dictionaries.get("encode_dpw", {}), new_user.get(col, 0))
                        else:
                            pass
                    # Tworzenie DataFrame
                    print(new_user_coded)
                    new_user_df = pd.DataFrame([new_user_coded], columns=X_train.columns)

                    # Predykcja i prawdopodobieństwa
                    probabilities = model.predict_proba(new_user_df)
                    top_n = 10
                    top_classes_indices = probabilities[0].argsort()[-top_n:][::-1]
                    exercise_to_type_id = clean_data.set_index("Exercise")["Exercise Type ID"].to_dict()
                    # Generowanie top_exercises z grupą mięśniową
                    top_exercises = [
                        (
                            model.classes_[i],  # Ćwiczenie
                            probabilities[0][i],  # Prawdopodobieństwo
                            encode_value(exercise_to_muscle, str(model.classes_[i])),
                            exercise_to_type_id.get(model.classes_[i])
                            
                        )
                        for i in top_classes_indices
                    ]               

                    # Debugowanie - Wyświetlenie top_exercises z grupami mięśniowymi
                    print("Top Exercises with Muscle Groups:")
                    for exercise, prob, muscle_group, ex_type in top_exercises:
                        print(f"Exercise: {exercise}, Probability: {prob:.2%}, Muscle Group: {muscle_group}, ex_type: {ex_type}")

        # Wyświetlanie wyników
    print("\n=== Wyniki kombinacji ===\n")
    for result in results:
        print(f"Larger Body Parts: {', '.join(result['Larger Body Parts'])}")
        print(f"Equipment Required: {', '.join(result['Equipment Required_y'])}")
        print(f"Target Gender: {', '.join(result['Target Gender'])}")
        print(f"Time Per Workout: {', '.join(result['Time Per Workout'])}")
        print("Top Exercises:")
        for exercise, probability in result['Top Exercises']:
            print(f"  - {exercise}: {probability:.2%}")
        print("-" * 50)  # Separator między wynikami
    def generate_plan(top_exercises, muscle_groups, days_per_week, time_per_workout, nickname=nickname):
        """
        Tworzy plan treningowy na podstawie top_exercises.

        Args:
        - top_exercises: Lista ćwiczeń z prawdopodobieństwami i grupami mięśniowymi.
        - muscle_groups: Lista grup mięśniowych zadeklarowanych przez użytkownika.
        - days_per_week: Liczba dni treningowych podana przez użytkownika.
        - time_per_workout: Czas treningu podany przez użytkownika.
        - nickname: Nazwa użytkownika.

        Returns:
        - plan: Lista dni z przydzielonymi ćwiczeniami.
        """
        # Mapowanie czasu na zakres liczby ćwiczeń
        time_to_exercise_range = {
            "Less than hour": (4, 5),
            "One hour": (5, 6),
            "One and a half hours": (6, 7),
            "Two hours and more": (7, 7),
        }

        # Określ zakres liczby ćwiczeń dla danego czasu treningu
        min_exercises, max_exercises = time_to_exercise_range.get(time_per_workout, (4, 7))

        plan = []
        used_exercises = set()

        for day in range(1, int(days_per_week) + 1):
            daily_plan = []
            daily_exercise_types = set()

            # Dodaj ćwiczenia dla każdej zadeklarowanej grupy mięśniowej
            for muscle in muscle_groups:
                muscle_exercises = [
                    (exercise, prob, muscle_group, exercise_type)
                    for exercise, prob, muscle_group, exercise_type in top_exercises
                    if muscle_group == muscle
                    and exercise not in used_exercises
                    and exercise_type not in daily_exercise_types
                ]
                muscle_exercises.sort(key=lambda x: x[1], reverse=True)

                if muscle_exercises:
                    selected_exercise = muscle_exercises[0]
                    daily_plan.append(selected_exercise)
                    used_exercises.add(selected_exercise[0])
                    daily_exercise_types.add(selected_exercise[3])

            # Dodaj dodatkowe ćwiczenia do osiągnięcia minimalnej liczby
            additional_exercises = [
                (exercise, prob, muscle_group, exercise_type)
                for exercise, prob, muscle_group, exercise_type in top_exercises
                if exercise not in used_exercises
            ]
            additional_exercises.sort(key=lambda x: x[1], reverse=True)

            while len(daily_plan) < min_exercises and additional_exercises:
                selected_exercise = additional_exercises.pop(0)
                daily_plan.append(selected_exercise)
                used_exercises.add(selected_exercise[0])
                daily_exercise_types.add(selected_exercise[3])

            # Jeśli nadal nie osiągnięto minimalnej liczby ćwiczeń, pozwól na powtórzenie exercise_type_id
            if len(daily_plan) < min_exercises:
                repeated_exercises = [
                    (exercise, prob, muscle_group, exercise_type)
                    for exercise, prob, muscle_group, exercise_type in top_exercises
                    if exercise not in [ex[0] for ex in daily_plan]
                ]
                repeated_exercises.sort(key=lambda x: x[1], reverse=True)

                while len(daily_plan) < min_exercises and repeated_exercises:
                    selected_exercise = repeated_exercises.pop(0)
                    daily_plan.append(selected_exercise)

            # Ogranicz do maksymalnej liczby ćwiczeń
            daily_plan = daily_plan[:max_exercises]

            # Dodaj dzień do planu
            plan.append({
                "User": nickname,
                "Day": day,
                "Exercises": [{
                    "Exercise": ex[0],
                    "Probability": ex[1],
                    "Muscle Group": ex[2],
                    "Exercise Type ID": ex[3]
                } for ex in daily_plan]
            })

        return plan

    print(new_user)
    nickname = nickname = last_user["Nickname"]
    muscle_groups = [new_user_coded["Larger Body Parts"]]
    days_per_week = new_user["Days Per Week"]
    time_per_workout = new_user["Time Per Workout"]
    plan = generate_plan(top_exercises, muscle_groups, days_per_week, time_per_workout, nickname)
    print(plan)
    def convert_plan_to_serializable(plan):
        return [
            {
                "User": nickname,
                "Day": int(item["Day"]),
                "Exercises": [
                    {
                        "Exercise": int(ex["Exercise"]),
                        "Probability": float(ex["Probability"]),
                        "Muscle Group": int(ex["Muscle Group"])
                    }
                    for ex in item["Exercises"]
                ]
            }
            for item in plan
        ]

    serializable_plan = convert_plan_to_serializable(plan)

    with open(f"users/{nickname}.json", "w") as json_file:
        json.dump(serializable_plan, json_file, indent=4)

if __name__ == "__main__":
    start_algorithm()
