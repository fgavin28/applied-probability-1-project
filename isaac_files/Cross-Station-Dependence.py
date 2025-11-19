import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1) Load Data
# ------------------------------------------------------
url = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource/ef819959-8183-45d4-a0bb-8027cf8f0876/download/dublin-bikes_station_status_032025.csv"
df_station = pd.read_csv(url)

df_station["timestamp"] = pd.to_datetime(df_station["last_reported"])

# ------------------------------------------------------
# 2) Pivot to wide format
# ------------------------------------------------------
df_wide = df_station.pivot_table(
    index="timestamp",
    columns="station_id",
    values="num_bikes_available"
).sort_index()

# Drop columns with all NaN (rare but safe)
df_wide = df_wide.dropna(axis=1, how="all")

# ------------------------------------------------------
# 3) Correlation Matrix
# ------------------------------------------------------
corr_matrix = df_wide.corr()

# Flatten correlations excluding self-correlation
corr_values = corr_matrix.values
corr_flat = corr_values[np.triu_indices_from(corr_values, k=1)]

# ------------------------------------------------------
# 📌 PLOT 1: Heatmap
# ------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.title("Cross-Station Correlation of Available Bikes")
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 📌 PLOT 2: Histogram of Pairwise Correlations
# ------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.title("Distribution of Station-to-Station Correlations")
plt.hist(corr_flat, bins=20, edgecolor="black")
plt.xlabel("Correlation")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 📌 PLOT 3: Scatter Plots for Example Pairs
# ------------------------------------------------------
# Find one strong and one weak correlation pair
pairs = []
cols = corr_matrix.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
strong = pairs_sorted[0]   # highest |correlation|
weak = pairs_sorted[-1]    # lowest |correlation|

def scatter_two(st1, st2, title):
    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.scatter(df_wide[st1], df_wide[st2], s=8)
    plt.xlabel(f"Station {st1} Available Bikes")
    plt.ylabel(f"Station {st2} Available Bikes")
    plt.tight_layout()
    plt.show()

scatter_two(strong[0], strong[1], f"Strong correlation example r={strong[2]:.3f}")
scatter_two(weak[0], weak[1], f"Weak correlation example r={weak[2]:.3f}")

# ------------------------------------------------------
# Summary stats
# ------------------------------------------------------
mean_corr = np.nanmean(corr_flat)
median_corr = np.nanmedian(corr_flat)

print("\n----- Cross-Station Correlation Summary -----")
print(f"Mean correlation:   {mean_corr:.3f}")
print(f"Median correlation: {median_corr:.3f}")
print(f"Max correlation:    {np.nanmax(corr_flat):.3f}")
print(f"Min correlation:    {np.nanmin(corr_flat):.3f}")

print("\nInterpretation:")
print("• Heatmap shows clusters of stations with similar demand patterns.")
print("• Histogram shows whether correlations are generally weak, moderate, or strong.")
print("• Scatterplots illustrate examples of strong vs weak dependence.")
print("\nNote: Correlation suggests possible dependence, but does not prove it.")
