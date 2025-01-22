import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
# Wczytanie clean_data tylko z wybranymi kolumnami
df = pd.read_csv("eclean_data_updated.csv")
df = df.dropna()

# Podział na cechy i target
X = df.drop("Exercise", axis=1)
y = df["Exercise"]
def simulate_max_depth_effect(X, y):
    max_depths = range(1, 51)  # Testujemy max_depth od 1 do 20
    mean_accuracies = []

    for max_depth in max_depths:
        # Tworzymy model drzewa decyzyjnego
        tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        
        # Ocena modelu przy pomocy cross-validation
        scores = cross_val_score(tree_model, X, y, cv=3, scoring='accuracy')
        mean_accuracies.append(scores.mean())

    # Znajdowanie najlepszej wartości max_depth
    best_max_depth = max_depths[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, mean_accuracies, marker='o')
    plt.title("Wpływ ograniczenia maksymalnej głębokości drzew decyzyjnych na dokładność modelu utworzonego przy pomocy lasu losowego")
    plt.xlabel("Maksymalna głębokość")
    plt.ylabel("Średnia dokładność modelu")
    plt.grid(True)
    plt.axvline(x=best_max_depth, color='r', linestyle='--', label=f"Głębokość gwarantująca najwyższą dokładność = {best_max_depth}")
    plt.legend()
    plt.show()

    print(f"Najlepsza wartość max_depth: {best_max_depth} z dokładnością: {best_accuracy:.2%}")


def simulate_n_estimators_effect(X, y):
    n_estimators_range = range(10, 210, 10)  # Testujemy liczby drzew od 10 do 200 (co 10)
    mean_accuracies = []

    for n_estimators in n_estimators_range:
        # Tworzymy model lasu losowego
        forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        # Ocena modelu przy pomocy cross-validation
        scores = cross_val_score(forest_model, X, y, cv=5, scoring='accuracy')
        mean_accuracies.append(scores.mean())

    # Znajdowanie najlepszej wartości n_estimators
    best_n_estimators = n_estimators_range[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, mean_accuracies, marker='o')
    plt.title("Wpływ ograniczenia maksymalnej liczby drzew decyzyjnych na dokładność modelu utworzonego przy pomocy lasu losowego")
    plt.xlabel("Liczba drzew decyzyjnych")
    plt.ylabel("Średnia dokładność modelu")
    plt.grid(True)
    plt.axvline(x=best_n_estimators, color='r', linestyle='--', label=f"Liczba drzew gwarantująca najwyższą dokładność = {best_n_estimators}")
    plt.legend()
    plt.show()

    print(f"Najlepsza wartość n_estimators: {best_n_estimators} z dokładnością: {best_accuracy:.2%}")

import time
def measure_time_and_accuracy(X, y, max_depths, n_estimators, num_trials=10):
    results = []

    for max_depth in max_depths:
        total_time = 0
        total_accuracy = 0

        for _ in range(num_trials):
            # Tworzymy model lasu losowego
            forest_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            
            # Mierzenie czasu treningu
            start_time = time.time()
            scores = cross_val_score(forest_model, X, y, cv=5, scoring='accuracy')
            end_time = time.time()

            # Aktualizacja sum dla średnich
            elapsed_time = end_time - start_time
            mean_accuracy = scores.mean()
            total_time += elapsed_time
            total_accuracy += mean_accuracy

        # Obliczanie średnich wyników z 10 prób
        avg_time = total_time / num_trials
        avg_accuracy = total_accuracy / num_trials

        # Zapis wyników
        results.append({
            'Maksymalna głębokość drzew decyzyzjnych': max_depth,
            'Maksymalna liczba drzew decyzyjnych': n_estimators,
            'Średni czas pracy modelu (s)': round(avg_time, 2),
            'Średnia miara dokładności wyniku': round(avg_accuracy, 2)
        })

    return results

    # Definiowanie parametrów
max_depths = [25, 44]  # Ograniczenia głębokości
n_estimators = 30      # Liczba drzew

    # Wywołanie funkcji


def simulate_n_estimators_time_accuracy(X, y):
    n_estimators_range = range(10, 210, 10)  # Testujemy liczby drzew od 10 do 200 (co 10)
    mean_accuracies = []
    training_times = []

    for n_estimators in n_estimators_range:
        # Tworzymy model lasu losowego
        forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        # Mierzenie czasu treningu
        start_time = time.time()
        scores = cross_val_score(forest_model, X, y, cv=5, scoring='accuracy')
        end_time = time.time()

        # Zapis wyników
        training_times.append(end_time - start_time)
        mean_accuracies.append(scores.mean())

    # Znajdowanie najlepszej wartości n_estimators
    best_n_estimators = n_estimators_range[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)


''' # Tworzenie wykresów
    plt.figure(figsize=(12, 8))

    # Wykres dokładności
    plt.subplot(2, 1, 1)
    plt.plot(n_estimators_range, mean_accuracies, marker='o', label="Przebieg zmian dokładności")
    plt.title("Wpływ ograniczenia maksymalnej liczby drzew decyzyjnych na dokładność modelu utworzonego przy pomocy lasu losowego z uwzględnieniem czasu działania algorytmu")
    plt.xlabel("Liczba drzew decyzyjnych")
    plt.ylabel("Średnia dokładność")
    plt.grid(True)
    plt.axvline(x=best_n_estimators, color='r', linestyle='--', label=f"Liczba drzew gwarantująca najwyższą dokładność = {best_n_estimators}")
    plt.legend()

    # Wykres czasu treningu
    plt.subplot(2, 1, 2)
    plt.plot(n_estimators_range, training_times, marker='o', label="Czas trenowania modelu", color='orange')
    plt.xlabel("Liczba drzew decyzyjnych")
    plt.ylabel("Czas treningu (s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Najlepsza wartość n_estimators: {best_n_estimators} z dokładnością: {best_accuracy:.2%}")
    '''

results = measure_time_and_accuracy(X, y, max_depths, n_estimators)
results_df = pd.DataFrame(results)
print(results_df)
# Wyświetlenie tabeli w eleganckim stylu
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_df.columns))))
plt.title("Wyniki analizy modelu", fontsize=12, pad=15)
plt.show()

# Zapis tabeli do pliku (opcjonalne)
results_df.to_csv("results_analysis.csv", index=False)