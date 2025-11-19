import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The previous code block (Load Data, Pivot, Correlation) is assumed to be run.
# We will use the original 'df_station' dataframe from step 1.
# ------------------------------------------------------
# 1) Load Data
# ------------------------------------------------------
url = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource/ef819959-8183-45d4-a0bb-8027cf8f0876/download/dublin-bikes_station_status_032025.csv"
df_station = pd.read_csv(url)

# Convert to datetime and sort (crucial for cumulative calculation)
df_station["timestamp"] = pd.to_datetime(df_station["last_reported"])
df_station = df_station.sort_values("timestamp")

# ------------------------------------------------------
# 4) Define 'low_availability' and Compute Running Average
# ------------------------------------------------------

# 📌 Step 4a: Define 'low_availability' (e.g., less than 2 available bikes)
LOW_BIKE_THRESHOLD = 2
df_station["low_availability"] = (
    df_station["num_bikes_available"] < LOW_BIKE_THRESHOLD
).astype(int)
print(
    f"Defined 'low_availability' as num_bikes_available < {LOW_BIKE_THRESHOLD}"
)

# 📌 Step 4b: Compute running average and true probability
# The cumulative mean represents the empirical probability P(low_availability) as n increases.
cumulative_low = df_station["low_availability"].expanding().mean()

# The overall mean of the full sample is the true probability approximation.
true_prob = df_station["low_availability"].mean()

# Compute the absolute error for later plotting
error = np.abs(cumulative_low - true_prob)

# Get the sample sizes n (the index of the series, starting from 1)
sample_size_n = np.arange(1, len(cumulative_low) + 1)

# ------------------------------------------------------
# 📌 PLOT 4: Law of Large Numbers Demo
# ------------------------------------------------------
plt.figure(figsize=(10, 8))

# --- Subplot 1: Convergence Plot ---
plt.subplot(2, 1, 1)
plt.plot(sample_size_n, cumulative_low, label="Running Average P(Low Availability)")
plt.axhline(
    true_prob,
    color="r",
    linestyle="--",
    label=f"True Probability ({true_prob:.4f})",
)
plt.title("Law of Large Numbers: Convergence of Empirical Probability")
plt.xlabel("Sample Size (n)")
plt.ylabel("Probability P(Low Availability)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.xlim(0, len(cumulative_low))

# --- Subplot 2: Error Shrinkage Plot ---
plt.subplot(2, 1, 2)
plt.plot(sample_size_n, error, label="|Error|", color="g")
plt.axhline(0, color="k", linewidth=0.5)
plt.title("|Error| Shrinkage: |Running Average - True Probability|")
plt.xlabel("Sample Size (n)")
plt.ylabel("|Error|")
plt.ylim(bottom=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xlim(0, len(error))

plt.tight_layout()
plt.show()

# ------------------------------------------------------
# Summary
# ------------------------------------------------------
print("\n----- Law of Large Numbers Summary -----")
print(
    f"True Probability P(Low Availability < {LOW_BIKE_THRESHOLD}): {true_prob:.4f}"
)
print(
    "Plot Interpretation:"
)
print(
    "• The top plot shows the running average (empirical probability) getting closer"
)
print(
    "  to the horizontal true probability line as the sample size increases."
)
print(
    "• The bottom plot confirms this convergence by showing the absolute error"
)
print(
    "  shrinking towards zero, which is the core principle of the Law of Large Numbers."
)