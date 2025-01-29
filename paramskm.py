import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
df = pd.read_csv("eclean_data_updated.csv")
df = df.dropna()

X = df.drop(["Exercise", "Exercise Type ID", "Larger Body Parts"], axis=1)
y = df["Exercise"]

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def test_kmeans_with_different_centroids(X_scaled):
    init_methods = ['k-means++', 'random']
    n_clusters_range = range(2, 20)  # Test liczby klastrów od 2 do 10
    results = []

    for init_method in init_methods:
        silhouette_scores = []
        training_times = []

        for n_clusters in n_clusters_range:
            start_time = time.time()
            kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=42, n_init="auto")
            kmeans.fit(X_scaled)
            end_time = time.time()

            silhouette = silhouette_score(X_scaled, kmeans.labels_)

            silhouette_scores.append(silhouette)
            training_times.append(end_time - start_time)

        results.append({
            'init_method': init_method,
            'silhouette_scores': silhouette_scores,
            'training_times': training_times
        })

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(n_clusters_range, silhouette_scores, marker='o', label=f"Miara Silhouette dla doboru losowego")
        plt.title(f"Porównanie metod inicjalizacji centroidów - symulacja dla doboru losowego")
        plt.xlabel("Liczba klastrów")
        plt.ylabel("Wynik miary Silhouette")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(n_clusters_range, training_times, marker='o', label=f"Czas treningu dla doboru losowego", color='orange')
        plt.xlabel("Liczba klastrów")
        plt.ylabel("Czas treningu (s)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    for result in results:
        best_cluster_count = n_clusters_range[np.argmax(result['silhouette_scores'])]
        best_score = max(result['silhouette_scores'])
        print(f"Metoda: {result['init_method']}, Najlepsza liczba klastrów: {best_cluster_count}, "
              f"Silhouette Score: {best_score:.2f}")



def evaluate_n_clusters(X_scaled):
    n_clusters_range = range(2, 20)  # Liczba klastrów od 2 do 10
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        silhouette_scores.append(silhouette_score(X_scaled, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))

    plt.figure(figsize=(15, 10))

    # Davies-Bouldin Index
    plt.subplot(2, 1, 1)
    plt.plot(n_clusters_range, davies_bouldin_scores, marker='o', label="metryka Daviesa-Bouldina", color='orange')
    plt.title("Symulacja wyników w celu dobrania odpowiedniej ilości klastrów - ocena metryką Daviesa-Bouldina")
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Wynik oceny metryką Daviesa-Bouldina")
    plt.grid(True)
    plt.legend()

    # Calinski-Harabasz Index
    plt.subplot(2, 1, 2)
    plt.plot(n_clusters_range, calinski_harabasz_scores, marker='o', label=" metryka Calinskiego-Harabasza", color='green')
    plt.title("Symulacja wyników w celu dobrania odpowiedniej ilości klastrów - ocena metryką Calinskiego-Harabasza")
    plt.xlabel("Liczba klastrów")
    plt.ylabel("Wynik oceny metryką Calinskiego-Harabasza")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    optimal_davies_bouldin = n_clusters_range[np.argmin(davies_bouldin_scores)]
    optimal_calinski_harabasz = n_clusters_range[np.argmax(calinski_harabasz_scores)]

    print("Optymalna liczba klastrów według metryki Davies-Bouldin:", optimal_davies_bouldin)
    print("Optymalna liczba klastrów według metryki Calinski-Harabasz:", optimal_calinski_harabasz)


test_kmeans_with_different_centroids(X_scaled)
evaluate_n_clusters(X_scaled)
