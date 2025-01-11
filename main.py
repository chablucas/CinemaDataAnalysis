import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'cinemas.csv'

# EXERCICE 1 :

cinema = pd.read_csv(file_path, sep=';', encoding='utf-8')
print(cinema.head())

print("Valeurs manquantes par colonne :")
print(cinema.isnull().sum()) 

cinema = cinema.dropna() 
print("DataFrame après traitement des valeurs manquantes et incohérentes :")
print(cinema.head()) 

columns_of_interest = ['fauteuils', 'écrans', 'entrées 2021', 'entrées 2022']
print("Statistiques descriptives des colonnes numériques principales :")
print(cinema[columns_of_interest].describe())

# EXERCICE 2 :

cinema['entrées_par_fauteuil'] = cinema['entrées 2022'] / cinema['fauteuils']
entrées_moyennes_par_région = cinema.groupby('région administrative')['entrées_par_fauteuil'].mean()

print("Entrées moyennes par fauteuil pour chaque région en 2022 :")
for region, value in entrées_moyennes_par_région.items():
    print(f"- {region}: {value:.2f}")

meilleurs_résultats = entrées_moyennes_par_région.nlargest(3)
print("\nLes 3 régions avec les meilleurs résultats en termes d'entrées moyennes par fauteuil :")
for region, value in meilleurs_résultats.items():
    print(f"- {region}: {value:.2f}")

pires_résultats = entrées_moyennes_par_région.nsmallest(3)
print("\nLes 3 régions avec les pires résultats en termes d'entrées moyennes par fauteuil :")
for region, value in pires_résultats.items():
    print(f"- {region}: {value:.2f}")

top_10_régions = entrées_moyennes_par_région.nlargest(10)

plt.figure(figsize=(10, 6))
top_10_régions.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Entrées moyennes par fauteuil pour les 10 meilleures régions (2022)", fontsize=14)
plt.ylabel("Entrées moyennes par fauteuil", fontsize=12)
plt.xlabel("Région administrative", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

# EXERCICE 3 :

correlation_ecrans_entrees = cinema['écrans'].corr(cinema['entrées 2022'])
correlation_fauteuils_entrees = cinema['fauteuils'].corr(cinema['entrées 2022'])

print(f"Corrélation entre le nombre d'écrans et les entrées annuelles (2022) : {correlation_ecrans_entrees:.2f}")
print(f"Corrélation entre le nombre de fauteuils et les entrées annuelles (2022) : {correlation_fauteuils_entrees:.2f}")

def plot_regression(x, y, x_label, y_label, title):
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    plt.scatter(x, y, alpha=0.7, label='Données')
    
    plt.plot(x, regression_line, color='red', label=f"Régression linéaire (y = {slope:.2f}x + {intercept:.2f})")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_regression(
    x=cinema['écrans'],
    y=cinema['entrées 2022'],
    x_label="Nombre d'écrans",
    y_label="Entrées annuelles (2022)",
    title="Relation entre le nombre d'écrans et les entrées annuelles (2022)"
)

plot_regression(
    x=cinema['fauteuils'],
    y=cinema['entrées 2022'],
    x_label="Nombre de fauteuils",
    y_label="Entrées annuelles (2022)",
    title="Relation entre le nombre de fauteuils et les entrées annuelles (2022)"
)