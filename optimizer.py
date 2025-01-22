import pandas as pd
import json
import random
import csv
import os

def find_muscle_group(exercise_id, data):
    """Funkcja do odczytu grupy mięśniowej na podstawie ID ćwiczenia."""
    for day in data:
        for exercise in day["Exercises"]:
            if exercise["Exercise"] == exercise_id:
                return exercise["Muscle Group"]
    return None  # Jeśli ćwiczenie nie zostanie znalezione


def extract_values(row, exercise):
    """Wyciąga wartości dla konkretnego ćwiczenia z kolumny Exercises Tried."""
    if pd.isna(row):  # Obsługa wartości NaN
        return 0  # Wartość domyślna
    try:
        exercises = dict(item.split(": ") for item in row.split("; "))
        return int(exercises.get(exercise, 0))
    except Exception as e:
        print(f"Błąd parsowania wiersza '{row}': {e}")
        return 0



def decode_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def optimize():
    try:
        # Odczyt danych użytkownika
        user_database_path = "user_database.csv"
        user_database = pd.read_csv(user_database_path, header=0)
        last_user = user_database.iloc[-1]
        new_user = last_user.to_dict()

        # Pobranie danych użytkownika
        nickname = new_user["Nickname"]
        goal = new_user["Main Goal"]
        e_lvl = new_user["Experience Level"]
        t_gender = new_user["Target Gender"]
        bodyweight = float(new_user["Bodyweight"])
        '''
        # Wyświetlenie problematycznych wierszy
        missing_exercises = user_database[user_database["Exercises Tried"].isna()]
        print("Brakujące dane dla Exercises Tried:")
        print(missing_exercises)


        # Wyciągnięcie wyników ćwiczeń
        user_database["Deadlift"] = user_database["Exercises Tried"].apply(lambda x: extract_values(x, "Deadlift"))
        user_database["Squat"] = user_database["Exercises Tried"].apply(lambda x: extract_values(x, "Squat"))
        user_database["Bench Press"] = user_database["Exercises Tried"].apply(lambda x: extract_values(x, "Bench Press"))

        # Wyświetlenie wyników
        print(user_database[["Nickname", "Deadlift", "Squat", "Bench Press"]])
        deadlift = user_database["Deadlift"].iloc[-1]
        squat = user_database["Squat"].iloc[-1]
        b_press = user_database["Bench Press"].iloc[-1]'''

        # Wczytanie danych użytkownika z pliku JSON
        with open(f"users/{nickname}.json", "r") as json_file:
            data = json.load(json_file)
        with open(f"dictionaries/encode_lbp.json", "r") as json_file:
            encode_lbp = json.load(json_file)
        with open(f"dictionaries/encode_ex.json", "r") as json_file:
            encode_ex = json.load(json_file)
        weights = {}
        pause, sets, reps = 0, 0, 0

        # Określanie wag w zależności od poziomu zaawansowania
                # Określanie wag w zależności od poziomu zaawansowania
        def assign_beginner_weights(bodyweight, t_gender, goal):
            if goal == "Build Muscle":
                const = random.randrange(55,85,5)/100
                return {
                    1: bodyweight * (0.16 if t_gender == "Woman" else 0.29)*const,
                    2: bodyweight * (0.2 if t_gender == "Woman" else 0.41)*const,
                    3: bodyweight * (0.1 if t_gender == "Woman" else 0.13)*const,
                    4: bodyweight * (0.2 if t_gender == "Woman" else 0.21)*const,
                    5: bodyweight * (0.3 if t_gender == "Woman" else 0.59)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 0.41)*const,
                }

            elif goal == "Increase Endurance":
                const = random.randrange(40,65,5)/100
                return {
                    1: bodyweight * (0.16 if t_gender == "Woman" else 0.29)*const,
                    2: bodyweight * (0.2 if t_gender == "Woman" else 0.41)*const,
                    3: bodyweight * (0.1 if t_gender == "Woman" else 0.13)*const,
                    4: bodyweight * (0.2 if t_gender == "Woman" else 0.21)*const,
                    5: bodyweight * (0.3 if t_gender == "Woman" else 0.59)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 0.41)*const,
                }
            elif  goal == "Increase Strength":
                const = random.randrange(75,105,5)/100
                return {
                    1: bodyweight * (0.16 if t_gender == "Woman" else 0.29)*const,
                    2: bodyweight * (0.2 if t_gender == "Woman" else 0.41)*const,
                    3: bodyweight * (0.1 if t_gender == "Woman" else 0.13)*const,
                    4: bodyweight * (0.2 if t_gender == "Woman" else 0.21)*const,
                    5: bodyweight * (0.3 if t_gender == "Woman" else 0.59)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 0.41)*const,
                }
            else: 
                const = random.randrange(40,85,5)/100
                return {
                    1: bodyweight * (0.16 if t_gender == "Woman" else 0.29)*const,
                    2: bodyweight * (0.2 if t_gender == "Woman" else 0.41)*const,
                    3: bodyweight * (0.1 if t_gender == "Woman" else 0.13)*const,
                    4: bodyweight * (0.2 if t_gender == "Woman" else 0.21)*const,
                    5: bodyweight * (0.3 if t_gender == "Woman" else 0.59)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 0.41)*const,
                }
        def assign_intermediate_weights(bodyweight, t_gender, goal):
            if goal == "Build Muscle":
                const = random.randrange(55,85,5)/100
                return {
                    1: bodyweight * (0.47 if t_gender == "Woman" else 0.73)*const,
                    2: bodyweight * (0.41 if t_gender == "Woman" else 0.96)*const,
                    3: bodyweight * (0.3 if t_gender == "Woman" else 0.51)*const,
                    4: bodyweight * (0.6 if t_gender == "Woman" else 0.79)*const,
                    5: bodyweight * (1.24 if t_gender == "Woman" else 1.64)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 1.12)*const,
                }
            elif goal == "Increase Endurance":
                const = random.randrange(40,65,5)/100
                return {
                    1: bodyweight * (0.47 if t_gender == "Woman" else 0.73)*const,
                    2: bodyweight * (0.41 if t_gender == "Woman" else 0.96)*const,
                    3: bodyweight * (0.3 if t_gender == "Woman" else 0.51)*const,
                    4: bodyweight * (0.6 if t_gender == "Woman" else 0.79)*const,
                    5: bodyweight * (1.24 if t_gender == "Woman" else 1.64)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 1.12)*const,
                }
            elif  goal == "Increase Strength":
                const = random.randrange(75,105,5)/100
                return {
                    1: bodyweight * (0.47 if t_gender == "Woman" else 0.73)*const,
                    2: bodyweight * (0.41 if t_gender == "Woman" else 0.96)*const,
                    3: bodyweight * (0.3 if t_gender == "Woman" else 0.51)*const,
                    4: bodyweight * (0.6 if t_gender == "Woman" else 0.79)*const,
                    5: bodyweight * (1.24 if t_gender == "Woman" else 1.64)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 1.12)*const,
                }
            else: 
                const = random.randrange(40,85,5)/100
                return {
                    1: bodyweight * (0.47 if t_gender == "Woman" else 0.73)*const,
                    2: bodyweight * (0.41 if t_gender == "Woman" else 0.96)*const,
                    3: bodyweight * (0.3 if t_gender == "Woman" else 0.51)*const,
                    4: bodyweight * (0.6 if t_gender == "Woman" else 0.79)*const,
                    5: bodyweight * (1.24 if t_gender == "Woman" else 1.64)*const,
                    6: bodyweight * (0.3 if t_gender == "Woman" else 1.12)*const,
                }
        def assign_advanced_weights(bodyweight, t_gender, goal):
            if goal == "Build Muscle":
                const = random.randrange(55,85,5)/100
                return {
                    1: bodyweight * (0.7 if t_gender == "Woman" else 1.09)*const,
                    2: bodyweight * (0.96 if t_gender == "Woman" else 1.44)*const,
                    3: bodyweight * (0.56 if t_gender == "Woman" else 0.78)*const,
                    4: bodyweight * (0.9 if t_gender == "Woman" else 1.26)*const,
                    5: bodyweight * (1.91 if t_gender == "Woman" else 2.41)*const,
                    6: bodyweight * (1.3 if t_gender == "Woman" else 1.72)*const,
                }
            elif goal == "Increase Endurance":
                const = random.randrange(40,65,5)/100
                return {
                    1: bodyweight * (0.7 if t_gender == "Woman" else 1.09)*const,
                    2: bodyweight * (0.96 if t_gender == "Woman" else 1.44)*const,
                    3: bodyweight * (0.56 if t_gender == "Woman" else 0.78)*const,
                    4: bodyweight * (0.9 if t_gender == "Woman" else 1.26)*const,
                    5: bodyweight * (1.91 if t_gender == "Woman" else 2.41)*const,
                    6: bodyweight * (1.3 if t_gender == "Woman" else 1.72)*const,
                }
            elif  goal == "Increase Strength":
                const = random.randrange(75,105,5)/100
                return {
                    1: bodyweight * (0.7 if t_gender == "Woman" else 1.09)*const,
                    2: bodyweight * (0.96 if t_gender == "Woman" else 1.44)*const,
                    3: bodyweight * (0.56 if t_gender == "Woman" else 0.78)*const,
                    4: bodyweight * (0.9 if t_gender == "Woman" else 1.26)*const,
                    5: bodyweight * (1.91 if t_gender == "Woman" else 2.41)*const,
                    6: bodyweight * (1.3 if t_gender == "Woman" else 1.72)*const,
                }  
            else:
                const = random.randrange(40,85,5)/100
                return {
                    1: bodyweight * (0.7 if t_gender == "Woman" else 1.09)*const,
                    2: bodyweight * (0.96 if t_gender == "Woman" else 1.44)*const,
                    3: bodyweight * (0.56 if t_gender == "Woman" else 0.78)*const,
                    4: bodyweight * (0.9 if t_gender == "Woman" else 1.26)*const,
                    5: bodyweight * (1.91 if t_gender == "Woman" else 2.41)*const,
                    6: bodyweight * (1.3 if t_gender == "Woman" else 1.72)*const,
                }
        if e_lvl == "Beginner":
            weights = assign_beginner_weights(bodyweight, t_gender, goal)
        elif e_lvl == "Intermediate":
            weights = assign_intermediate_weights(bodyweight, t_gender, goal)
        else:
            weights = assign_advanced_weights(bodyweight, t_gender, goal)

        '''else:  # Średniozaawansowani i zaawansowani
            weights = {}
            if int(squat) > 0:
                weights.update({
                    4: squat * 0.5,
                    5: squat * 0.5,
                    6: squat * 0.3
                })
            if int(deadlift) > 0:
                weights.update({
                    2: deadlift * 0.4,
                    3: deadlift * 0.1,
                })
            if int(b_press) > 0:
                weights.update({
                    1: b_press * 0.4,
                    3: (deadlift * 0.1 + b_press * 0.3) / 2,
                })
            # Sprawdzenie brakujących wartości
            if not weights:  # Jeśli brak danych dla ćwiczeń
                print("Brak danych dla squat, deadlift i bench press. Przypisano wartości jak dla początkującego.")
                weights = assign_beginner_weights(bodyweight, t_gender)'''            

        # Funkcja do dopasowania liczby serii i powtórzeń
        def calculate_sets_reps(goal):
            if goal == "Build Muscle":
                return random.randint(3, 5), random.randint(6, 12), 10 * random.randint(3, 9)
            elif goal == "Increase Strength":
                return random.randint(4, 6), random.randint(1, 6), 20 * random.randint(6, 15)
            elif goal == "Increase Endurance":
                return random.randint(2, 4), random.randint(12, 20), 15 * random.randint(1, 4)
            else:
                return random.randint(3, 6), random.randint(6, 12), 10 * random.randint(3, 6)

        # Tworzenie planu na podstawie danych z JSON
        workout_plan = []
        for day in data:
            for exercise in day["Exercises"]:
                exercise_id = exercise["Exercise"]
                muscle_group = find_muscle_group(exercise_id, data)

                if muscle_group is not None:
                    # Obliczanie unikalnych wartości dla każdego ćwiczenia
                    sets, reps, pause = calculate_sets_reps(goal)

                    # Pobranie ciężaru dla danej grupy mięśniowej
                    weight = weights.get(muscle_group, 0)

                    # Dodanie ćwiczenia do planu
                    workout_plan.append(
                        {
                            "User": nickname,
                            "Day": day["Day"],
                            "Exercise": decode_value(encode_ex, exercise_id),
                            "Muscle Group": decode_value(encode_lbp, muscle_group),
                            "Sets": sets,
                            "Reps": reps,
                            "Pause (sec)": pause,
                            "Weight (kg)": round(weight / 2.5) * 2.5,

                        }
                    )

        # Wyświetlenie planu treningowego
        print("Plan treningowy:")
        for exercise in workout_plan:
            print(exercise)

        # Zapis planu treningowego do pliku CSV
        csv_file = "user_workout_plan.csv"
        csv_columns = ["User", "Day", "Exercise", "Muscle Group", "Sets", "Reps", "Pause (sec)", "Weight (kg)"]
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline='', encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)

            # Jeśli plik nie istnieje, zapisz nagłówki
            if not file_exists:
                writer.writeheader()

            # Zapisz dane ćwiczeń
            writer.writerows(workout_plan)

        print(f"Plan treningowy zapisano w pliku: {csv_file}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    optimize()
