import pandas as pd
import matplotlib.pyplot as plt

def get_clean_worktime(work_time_path):
    return (pd.read_csv(work_time_path, sep=';', encoding='utf-8')
            .drop_duplicates()
            .fillna({'Temps annuel de travail (SNCF)': 0,
                     'Temps annuel de travail (France)': 0,
                     'Commentaires': ''})
            .astype({'Date': int,
                     'Temps annuel de travail (SNCF)': int,
                     'Temps annuel de travail (France)': int})
            .assign(Commentaires=lambda x: x['Commentaires'].str.strip()))

# work_time= pd.read_csv("./data/temps-de-travail-annuel-depuis-1851.csv")

def get_interesting_colums(df, columns):
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f"Columns missing")
        return pd.DataFrame()
    return df[columns]

work_time_file = get_clean_worktime('data/temps-de-travail-annuel-depuis-1851.csv')
work_time_columns = [
    "Date",
    "Temps annuel de travail (SNCF)",
    "Temps annuel de travail (France)"
]
work_time_filtered_columns = get_interesting_colums(work_time_file, work_time_columns)
work_time_filtered_rows = work_time_filtered_columns[
    (work_time_filtered_columns['Date'].astype(str) == '2017') | (work_time_filtered_columns['Date'].astype(str) == '2018')
    ]
work_time = work_time_filtered_rows

frequentation_file = pd.read_csv("./data/frequentation-gares.csv", sep=";")
frequentation_columns = [
    "Nom de la gare",
    "Code postal",
    "Total Voyageurs + Non voyageurs 2017",
    "Total Voyageurs + Non voyageurs 2018"
]
frequentation = get_interesting_colums(frequentation_file, frequentation_columns)

frequentation_filtered_columns = get_interesting_colums(frequentation_file, frequentation_columns)
frequentation_filtered_rows = frequentation_filtered_columns[
    frequentation_filtered_columns[
    'Code Postal'
    ].astype(str).str[:1] == '7'].head(3)
frequentation = frequentation_filtered_rows

# ----------------------------------

print("Frequentation pretty data")

def get_frequentation_diagram(frequentation):
    frequentation_df = pd.DatafFrame(frequentation)
    frequentation_df.set_index('Nom de la gare') [
        ['Total Voyageurs + Non voyageurs 2017',
         'Total Voyageurs + Non voyageurs 2018']
         ].plot(kind='bar', figsize=(12, 6))
    
    plt.title('Comparaison fréquentation des gares entre 2017 et 2018')
    plt.ylabel('Total voyageurs')
    plt.xlabel('Nom de la gare')
    plt.xticks(rotation=45)
    plt.grid(axys='y')
    plt.legend(title="Année")
    plt.show()

get_frequentation_diagram(frequentation)

# --------------------------
# Display

print("Work Time")
print(work_time)
print("\n\n")

print("Frequentation")
print(frequentation)
print("\n\n")

# --------------------------
# Deep learning

df = pd.DataFrame(frequentation)
df["Croissance annuelle"] = (
    (df["Total Voyaheurs 2023"] - df["Total Voyaheurs 2022"]) / df["Total Voyaheurs 2022"])
df.head()

df["Total Voyageurs + Non voyageurs 2024 (estimé)"] = (
    df["Total Voyageurs + Non voyageurs 2023"] * (1 + df["Croissance annuelle"])
)

x = df[["Total Voyageurs + Non voyageurs 2023", "Croissance annuelle"]]
y = df["Total Voyageurs + Non voyageurs 2023"]

model = LinearRegression()
model.fit(x, y)

print("Coefficients : ", model.coef_)
print("Intercept : ", model.intercept_)

df["Prédiction 2024"] = model.predict(x)
print(df[["Nom de la gare", "Preédiction 2024"]])
