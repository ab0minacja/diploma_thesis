import pandas as pd

INCREASE_CONSTANT = 0.05

def update_weights(progress, plan):
    for _, row in progress.iterrows():
        exercise_id = row['Exercise']
        user_feedback = row['Status']
        max_weight = row.get('Actual Weight (kg)', None)

        if exercise_id in plan['Exercise'].values:
            current_weight = progress.loc[progress['Exercise'] == exercise_id, "Predicted Weight"].values

            if len(current_weight) > 0:
                current_weight = current_weight[0]
            else:
                continue  # Brak wartości "Predicted Weight" dla tego ćwiczenia, przejdź do następnego

            if user_feedback == 'OK':
                new_weight = current_weight * 1.025
            elif user_feedback == 'Too Light' and max_weight is not None:
                new_weight = max_weight * 1.025
            elif user_feedback == 'Too Light' and max_weight is None:
                new_weight = current_weight * (1 + INCREASE_CONSTANT)
            elif user_feedback == 'Too Heavy' and max_weight is not None:
                new_weight = max_weight * 1.025
            elif user_feedback == 'Too Heavy' and max_weight is None:
                new_weight = current_weight * 0.975
            else:
                continue

            plan.loc[plan['Exercise'] == exercise_id, 'Weight (kg)'] = round(new_weight / 2.5) * 2.5

    return plan

def progress_weight():
    progress_file = 'user_progress.csv'
    workout_plan_file = 'user_workout_plan.csv'

    user_progress = pd.read_csv(progress_file)
    workout_plan = pd.read_csv(workout_plan_file)

    updated_workout_plan = update_weights(user_progress, workout_plan)
    updated_workout_plan.to_csv(workout_plan_file, index=False)

    print("Zaktualizowano plan treningowy w pliku 'user_workout_plan.csv'.")

if __name__ == "__main__":
    progress_weight()
