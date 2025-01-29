import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
import pandas as pd
import algorithm_dt
import optimizer
import progress

def show_start_panel():
    """Wyświetla panel startowy z opcjami Log In i Register."""
    start_window = tk.Toplevel()
    start_window.title("Welcome")
    start_window.geometry("500x400")
    start_window.resizable(False, False)

    tk.Label(start_window, text="Welcome to Workout Planner", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(start_window, text="Choose an option:", font=("Arial", 14)).pack(pady=10)

    tk.Button(start_window, text="Log In", command=lambda: show_login_panel(start_window), width=20, height=2).pack(pady=10)
    tk.Button(start_window, text="Register", command=lambda: show_register_panel(start_window), width=20, height=2).pack(pady=10)


def show_login_panel(start_window):
    """Pokazuje panel logowania."""
    start_window.destroy()

    login_window = tk.Toplevel()
    login_window.title("Log In")
    login_window.geometry("500x400")
    login_window.resizable(False, False)

    tk.Label(login_window, text="Log In", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(login_window, text="Enter your nickname:", font=("Arial", 14)).pack(pady=10)

    username_var = tk.StringVar()
    tk.Entry(login_window, textvariable=username_var, font=("Arial", 12), width=25).pack(pady=10)

    def login():
        nickname = username_var.get()
        if not nickname:
            messagebox.showerror("Error", "Nickname cannot be empty.")
            return

        if not os.path.isfile("user_workout_plan.csv"):
            messagebox.showerror("Error", "No workout plan file found.")
            return

        data = pd.read_csv("user_workout_plan.csv")
        user_data = data[data["User"] == nickname]

        if user_data.empty:
            messagebox.showerror("Error", "Nickname not found. Please register.")
        else:
            messagebox.showinfo("Success", f"Welcome back, {nickname}!")
            login_window.destroy()
            display_workout_plan(nickname, user_data)

    tk.Button(login_window, text="Log In", command=login, width=15, height=2).pack(pady=20)
    tk.Button(login_window, text="Back", command=lambda: [login_window.destroy(), show_start_panel()], width=15, height=2).pack()


def show_register_panel(start_window):
    """Pokazuje panel rejestracji."""
    start_window.destroy()
    root.deiconify()

def validate_tab1():
    if not nick_var.get():
        messagebox.showerror("Error", "Please enter your nickname.")
        return False
    if gender_var.get() == "Select your gender:":
        messagebox.showerror("Error", "Please select your gender.")
        return False
    return True


def validate_tab2():
    if experience_var.get() == "Set your experience level:":
        messagebox.showerror("Error", "Please select your experience level.")
        return False
    if goal_var.get() == "Set your training goal:":
        messagebox.showerror("Error", "Please select your main goal.")
        return False
    if days_var.get() not in ["1", "2", "3", "4", "5", "6", "7"]:
        messagebox.showerror("Error", "Please select the number of days per week.")
        return False
    if time_var.get() == "Preferred workout duration:":
        messagebox.showerror("Error", "Please select your preferred workout duration.")
        return False
    return True


def validate_tab3():
    if not any(var.get() for var in body_parts_vars.values()):
        messagebox.showerror("Error", "Please select at least one target muscle group.")
        return False
    return True


def validate_tab4():
    if not any(var.get() for var in equipment_vars.values()):
        messagebox.showerror("Error", "Please select at least one type of equipment.")
        return False
    return True


def validate_tab5():
    exercises_valid = any(var.get() for var, _ in exercises_vars.values())
    if exercises_valid:
        for exercise, (var, weight_var) in exercises_vars.items():
            if var.get() and (not weight_var.get().isdigit() or int(weight_var.get()) <= 0):
                messagebox.showerror("Error", f"Please enter a valid max weight for {exercise}.")
                return False
    return True


def next_tab(tab_index):
    if tab_index == 0 and not validate_tab1():
        return
    if tab_index == 1 and not validate_tab2():
        return
    if tab_index == 2 and not validate_tab3():
        return
    if tab_index == 3 and not validate_tab4():
        return
    notebook.select(tab_index + 1)


def submit_form():
    """Zapisuje dane użytkownika, uruchamia algorithm_dt i optimizer, a następnie umożliwia przejście do planu."""
    if not validate_tab5():
        return

    form_data = {
        "Nickname": nick_var.get(),
        "Target Gender": gender_var.get(),
        "Bodyweight": bodyweight_var.get(),
        "Experience Level": experience_var.get(),
        "Main Goal": goal_var.get(),
        "Days Per Week": days_var.get(),
        "Target Muscle Group": ", ".join([part for part, var in body_parts_vars.items() if var.get()]),
        "Time Per Workout": time_var.get(),
        "Equipment Required": ", ".join([equip for equip, var in equipment_vars.items() if var.get()]),
        "Exercises Tried": "; ".join([f"{exercise}: {weight_var.get()}" for exercise, (var, weight_var) in exercises_vars.items() if var.get()]),
    }

    file_name = "user_database.csv"
    file_exists = os.path.isfile(file_name)

    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=form_data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(form_data)

    messagebox.showinfo("Form Submitted", "Your data has been saved successfully!")

    try:
        algorithm_dt.start_algorithm()  
        print("algorithm_dt finished successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Error while running algorithm_dt: {e}")
        return

    
    try:
        optimizer.optimize()  
        print("optimizer finished successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Error while running optimizer: {e}")
        return

    
    show_go_to_plan_button(nick_var.get(), is_registration=True)


def show_go_to_plan_button(nickname, is_registration=False):
    """Wyświetla przycisk przejścia do planu po zakończeniu optymalizacji."""
    if not is_registration:
        return  # Wyświetla przycisk tylko dla nowo zarejestrowanego użytkownika

    go_to_plan_window = tk.Toplevel()
    go_to_plan_window.title("Optimization Complete")
    go_to_plan_window.geometry("300x150")

    tk.Label(go_to_plan_window, text="Optimization Completed!", font=("Arial", 14)).pack(pady=20)
    tk.Button(
        go_to_plan_window,
        text="Go to Plan",
        command=lambda: [go_to_plan_window.destroy(), root.withdraw(), load_and_display_plan(nickname)],
        bg="#90EE90",
        font=("Arial", 12)
    ).pack(pady=20)




def update_progress(nickname, week, day, day_data, progress_file="user_progress.csv"):
    """Aktualizuje postępy użytkownika w pliku CSV i uruchamia funkcję update_weights."""
    header = ["User", "Week Completed", "Day", "Exercise", "Actual Weight (kg)", "Status", "Predicted Weight"]

    # Wczytanie istniejących danych
    if os.path.isfile(progress_file):
        try:
            progress_data = pd.read_csv(progress_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read progress file: {e}")
            return
    else:
        progress_data = pd.DataFrame(columns=header)

    new_data = pd.DataFrame(day_data, columns=header)

    # Usunięcie istniejących wpisów dla tego użytkownika, tygodnia i dnia
    progress_data = progress_data[
        ~(
            (progress_data["User"] == nickname) & 
            (progress_data["Week Completed"] == week) & 
            (progress_data["Day"] == day)
        )
    ]
    progress_data = pd.concat([progress_data, new_data], ignore_index=True)
    try:
        progress_data.to_csv(progress_file, index=False)
        messagebox.showinfo("Success", f"Progress for Day {day} saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save progress: {e}")
        return

    try:
        workout_plan = pd.read_csv("user_workout_plan.csv")
        updated_plan = progress.update_weights(progress_data, workout_plan)
        updated_plan.to_csv("user_workout_plan.csv", index=False)
        messagebox.showinfo("Success", "Workout plan updated successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update workout plan: {e}")


def display_workout_plan(nickname, user_data):
    """Wyświetla plan treningowy użytkownika z dodatkowymi funkcjami."""
    plan_window = tk.Toplevel()
    plan_window.title(f"{nickname}'s Workout Plan")
    plan_window.geometry("1000x700")  
    plan_window.resizable(False, False)

    main_notebook = ttk.Notebook(plan_window)
    main_notebook.pack(fill="both", expand=True, padx=10, pady=10)

    style = ttk.Style()
    style.theme_use("default")
    style.map("TNotebook.Tab", background=[("selected", "#cce7ff")])

    weeks = 4
    days_per_week = user_data["Day"].max()
    daily_results = {} 

    def get_personal_best(exercise_name):
        """Pobiera najlepszy wynik użytkownika dla danego ćwiczenia."""
        progress_file = "user_progress.csv"
        if os.path.isfile(progress_file):
            try:
                progress_data = pd.read_csv(progress_file)
                user_data = progress_data[(progress_data["User"] == nickname) & (progress_data["Exercise"] == exercise_name)]
                if not user_data.empty:
                    return user_data["Actual Weight (kg)"].max()
            except Exception as e:
                print(f"Error reading progress file: {e}")
        return "N/A"

    for week in range(1, weeks + 1):
        week_tab = ttk.Notebook(main_notebook)
        main_notebook.add(week_tab, text=f"Week {week}")

        for day in range(1, days_per_week + 1):
            day_data = user_data[user_data["Day"] == day]
            if day_data.empty:
                continue


            day_frame = ttk.Frame(week_tab)
            week_tab.add(day_frame, text=f"Day {day}")

            daily_results[day] = []  

            for _, exercise in day_data.iterrows():
                exercise_frame = tk.LabelFrame(day_frame, text=exercise["Exercise"], font=("Arial", 10, "bold"))
                exercise_frame.pack(fill="x", pady=5, padx=10)

                personal_best = get_personal_best(exercise["Exercise"])

                tk.Label(
                    exercise_frame,
                    text=f"Muscle Part: {exercise['Muscle Group']} | Sets: {exercise['Sets']} | Reps: {exercise['Reps']} | Default Weight: {exercise['Weight (kg)']} kg | Personal Best: {personal_best} kg",
                    font=("Arial", 10),
                ).pack(pady=5, anchor="w")

                feedback_var = tk.StringVar(value="Ok")

                tk.Label(exercise_frame, text="Rate difficulty:", font=("Arial", 10)).pack(side="left", padx=5)
                feedback_menu = ttk.Combobox(
                    exercise_frame,
                    textvariable=feedback_var,
                    values=["Too Light", "Ok", "Too Heavy"],
                    state="readonly",
                    width=12,
                )
                feedback_menu.pack(side="left", padx=5)


                tk.Label(exercise_frame, text="Have you tried different weight?", font=("Arial", 10)).pack(side="left", padx=5)
                dropdown_var = tk.StringVar(value="No")  # Domyślna wartość to "No"
                dropdown_menu = ttk.Combobox(
                    exercise_frame,
                    textvariable=dropdown_var,
                    values=["No", "Yes"],
                    state="readonly",
                    width=5,
                    font=("Arial", 10),
                )
                dropdown_menu.pack(side="left", padx=5)

                # Pole do wpisywania wagi
                weight_var = tk.StringVar()
                weight_entry = tk.Entry(exercise_frame, textvariable=weight_var, width=8, font=("Arial", 10))
                weight_entry.pack(side="left", padx=5)

      
                def save_exercise_feedback(e, fv, wv, dv, d):
                    """Zapisuje dane ćwiczenia w daily_results."""
                    # Automatycznie ustawianie domyślnej wagi i statusu, jeśli wybrano "Ok" i "No"
                    if dv.get() == "No" and fv.get() == "Ok":
                        weight_to_save = e["Weight (kg)"] 
                        feedback_to_save = "Ok"
                    else:
                        weight_to_save = e["Weight (kg)"] if dv.get() == "No" else wv.get()
                        feedback_to_save = fv.get()

                    if d not in daily_results:
                        daily_results[d] = []

                    daily_results[d] = [entry for entry in daily_results[d] if entry[3] != e["Exercise"]]

                    daily_results[d].append(
                        [nickname, week, d, e["Exercise"], weight_to_save, feedback_to_save,e["Weight (kg)"]]
                    )

                    print(f"Updated daily_results for Day {d}: {daily_results[d]}")



                feedback_menu.bind(
                    "<<ComboboxSelected>>",
                    lambda e, ex=exercise, fv=feedback_var, wv=weight_var, dv=dropdown_var, d=day: save_exercise_feedback(ex, fv, wv, dv, d)
                )
                weight_var.trace_add(
                    "write",
                    lambda *args, ex=exercise, fv=feedback_var, wv=weight_var, dv=dropdown_var, d=day: save_exercise_feedback(ex, fv, wv, dv, d)
                )
                dropdown_var.trace_add(
                    "write",
                    lambda *args, ex=exercise, fv=feedback_var, wv=weight_var, dv=dropdown_var, d=day: save_exercise_feedback(ex, fv, wv, dv, d)
                )


            save_button = tk.Button(
                day_frame,
                text=f"Save Progress for Day {day}",
                command=lambda d=day: update_progress(nickname, week, d, daily_results[d]),
                width=30,
                height=1,
                font=("Arial", 10),
                bg="#90ee90",
                )

            save_button.pack(pady=10)
    buttons_frame = tk.Frame(plan_window)
    buttons_frame.pack(fill="x", pady=10)

    close_button = tk.Button(
        buttons_frame,
        text="Close",
        command=plan_window.destroy,
        width=20,
        height=2,
        font=("Arial", 12),
        bg="#f08080",
    )
    close_button.pack(side="right", padx=10)


def load_and_display_plan(nickname):
    """Wczytuje i wyświetla plan treningowy dla podanego użytkownika."""
    if not os.path.isfile("user_workout_plan.csv"):
        messagebox.showerror("Error", "No workout plan file found.")
        return

    data = pd.read_csv("user_workout_plan.csv")
    user_data = data[data["User"] == nickname]

    if user_data.empty:
        messagebox.showerror("Error", "No workout plan found for this user.")
    else:
        display_workout_plan(nickname, user_data)

# --- Główne okno Tkinter ---
root = tk.Tk()
root.title("Workout Planner")
root.geometry("500x600")
root.withdraw()  
# --- Dodaj styl dla zakładek notebooka ---
style = ttk.Style()
style.theme_use("default")
style.configure("TNotebook", background="#FFFFFF", borderwidth=0) 
style.configure("TNotebook.Tab", 
                background="#E0E0E0",  
                foreground="black",  
                padding=[10, 5],  
                font=("Arial", 10, "bold"), 
                anchor="center")  
style.map("TNotebook.Tab", 
          background=[("selected", "#C0C0C0")],  
          foreground=[("selected", "black")])  

# --- Notebook ---
notebook = ttk.Notebook(root, style="TNotebook")
notebook.pack(pady=10, expand=True)


root.geometry("700x700") 



frame1 = tk.Frame(notebook )
frame2 = tk.Frame(notebook )
frame3 = tk.Frame(notebook )
frame4 = tk.Frame(notebook )
frame5 = tk.Frame(notebook )

notebook.add(frame1, text="Personal Info")
notebook.add(frame2, text="Workout Preferences")
notebook.add(frame3, text="Target Muscle Group")
notebook.add(frame4, text="Equipment")
notebook.add(frame5, text="Exercises Tried")

# Frame 1: Personal Info
nick_var = tk.StringVar()
gender_var = tk.StringVar(value="Select your gender:")
bodyweight_var = tk.StringVar(value="0")

inner_frame1 = tk.Frame(frame1)
inner_frame1.pack(expand=True, pady=30) 

tk.Label(inner_frame1, text="Nickname:").grid(row=0, column=0, pady=5, sticky="w", padx=10)
tk.Entry(inner_frame1, textvariable=nick_var).grid(row=0, column=1, pady=5, padx=10)

tk.Label(inner_frame1, text="Target Gender:").grid(row=1, column=0, pady=5, sticky="w", padx=10)
gender_menu = ttk.Combobox(inner_frame1, textvariable=gender_var, values=["Male", "Female", "Other"], state="readonly")
gender_menu.grid(row=1, column=1, pady=5, padx=10)

tk.Label(inner_frame1, text="Bodyweight (kg):").grid(row=2, column=0, pady=5, sticky="w", padx=10)
tk.Entry(inner_frame1, textvariable=bodyweight_var).grid(row=2, column=1, pady=5, padx=10)

tk.Button(inner_frame1, text="Next", command=lambda: next_tab(0)).grid(row=3, column=0, columnspan=2, pady=20)

# Frame 2: Workout Preferences
experience_var = tk.StringVar(value="Set your experience level:")
goal_var = tk.StringVar(value="Set your training goal:")
days_var = tk.StringVar(value="3")
time_var = tk.StringVar(value="Preferred workout duration:")

inner_frame2 = tk.Frame(frame2)
inner_frame2.pack(expand=True, pady=30)

tk.Label(inner_frame2, text="Experience Level:").grid(row=0, column=0, pady=5, sticky="w", padx=10)
experience_menu = ttk.Combobox(inner_frame2, textvariable=experience_var, values=["Beginner", "Intermediate", "Advanced"], state="readonly")
experience_menu.grid(row=0, column=1, pady=5, padx=10)

tk.Label(inner_frame2, text="Main Goal:").grid(row=1, column=0, pady=5, sticky="w", padx=10)
goal_menu = ttk.Combobox(inner_frame2, textvariable=goal_var, values=["Build Muscle", "Increase Endurance", "Lose Fat", "General Fitness", "Increase Strength", "Sports Performance"], state="readonly")
goal_menu.grid(row=1, column=1, pady=5, padx=10)

tk.Label(inner_frame2, text="Days Per Week:").grid(row=2, column=0, pady=5, sticky="w", padx=10)
days_menu = ttk.Combobox(inner_frame2, textvariable=days_var, values=["1", "2", "3", "4", "5", "6", "7"], state="readonly")
days_menu.grid(row=2, column=1, pady=5, padx=10)

tk.Label(inner_frame2, text="Time Per Workout:").grid(row=3, column=0, pady=5, sticky="w", padx=10)
time_menu = ttk.Combobox(inner_frame2, textvariable=time_var, values=["Less than hour", "One hour", "One and a half hours", "Two hours and more"], state="readonly")
time_menu.grid(row=3, column=1, pady=5, padx=10)

tk.Button(inner_frame2, text="Next", command=lambda: next_tab(1)).grid(row=4, column=0, columnspan=2, pady=20)

# Frame 3: Target Muscle Group
body_parts_vars = {}

inner_frame3 = tk.Frame(frame3)
inner_frame3.pack(expand=True, pady=30)

tk.Label(inner_frame3, text="Target Muscle Group:").grid(row=0, column=0, columnspan=2, pady=10, sticky="w")

row_index = 1
col_index = 0

for i, part in enumerate(["Full Body", "Chest&Shoulders", "Back", "Arms", "Core", "Legs", "Glutes"]):
    var = tk.BooleanVar()
    chk = tk.Checkbutton(inner_frame3, text=part, variable=var)
    chk.grid(row=row_index, column=col_index, pady=5, padx=10, sticky="w")
    body_parts_vars[part] = var

    col_index += 1
    if col_index > 1:
        col_index = 0
        row_index += 1

tk.Button(inner_frame3, text="Next", command=lambda: next_tab(2)).grid(row=row_index + 1, column=0, columnspan=2, pady=20)

# Frame 4: Equipment
equipment_vars = {}

inner_frame4 = tk.Frame(frame4)
inner_frame4.pack(expand=True, pady=30)

tk.Label(inner_frame4, text="Select the equipment you have access to:").pack(pady=10)
equipment_frame = tk.Frame(inner_frame4)
equipment_frame.pack(pady=10)
for equipment in ["None", "All listed", "Barbell", "EZ Bar", "Dumbbell", "Machine", "Cable", "Bands", "Exercise Ball", "Bench", "Chains", "Kettle Bells", "Tire", "Box", "Landmine", "Trap Bar", "Jump Rope"]:
    var = tk.BooleanVar()
    chk = tk.Checkbutton(equipment_frame, text=equipment, variable=var)
    chk.pack(anchor="w")
    equipment_vars[equipment] = var

tk.Button(inner_frame4, text="Next", command=lambda: next_tab(3)).pack(pady=20)

# Frame 5: Exercises Tried
exercises_vars = {}

inner_frame5 = tk.Frame(frame5)
inner_frame5.pack(expand=True, pady=30)

tk.Label(inner_frame5, text="Have you tried any of these exercises?").pack(pady=10)
for exercise in ["Bench Press", "Squat           ", "Deadlift       "]:
    var = tk.BooleanVar()
    weight_var = tk.StringVar(value="0")
    frame = tk.Frame(inner_frame5)
    frame.pack(anchor="w", pady=5)
    
    chk = tk.Checkbutton(frame, text=exercise, variable=var)
    chk.pack(side="left", padx=5)
    
    tk.Label(frame, text="Max weight (kg):").pack(side="left", padx=5)
    
    entry = tk.Entry(frame, textvariable=weight_var, width=10)
    entry.pack(side="left", padx=(5, 20))  
    
    exercises_vars[exercise] = (var, weight_var)


tk.Button(inner_frame5, text="Submit", command=submit_form).pack(pady=20)

show_start_panel()
root.mainloop()
