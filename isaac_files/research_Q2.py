import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource/ef819959-8183-45d4-a0bb-8027cf8f0876/download/dublin-bikes_station_status_032025.csv"
df = pd.read_csv(url)

df['timestamp'] = pd.to_datetime(df['last_reported'])

station_id = 42
df_station = df[df['station_id'] == station_id].copy()
df_station = df_station.sort_values("timestamp")

x = df_station["num_bikes_available"].values

# sampling interval is fixed 5 minutes
steps_per_hour = 12

# function for autocorrelation
def autocorr(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

# compute autocorrelation per hour
max_hours = 24  # analyze 24 hours
hour_lags = list(range(1, max_hours + 1))
autocorrs = [autocorr(x, lag * steps_per_hour) for lag in hour_lags]

# print lag-1 hour autocorrelation
print("Lag-1 hour autocorrelation:", autocorrs[0])

# plot
plt.figure(figsize=(8, 4))
plt.stem(hour_lags, autocorrs)
plt.xlabel("Lag (hours)")
plt.ylabel("Autocorrelation")
plt.title(f"Autocorrelation by hour: Station {station_id}")
plt.show()
