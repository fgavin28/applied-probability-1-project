"""
STU22004 Applied Probability 1 – Group Project
Question 1: Temporal Availability Modelling (Poisson / Binomial / Conditional Probability)

Group:
- Agastya
- Jakub
- Fionn
- Isaac
"""

# =========================
# Imports
# =========================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render to files, not windows
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import poisson, binom, chi2_contingency

plt.rcParams["figure.figsize"] = (12, 6)


# =============================================================================
# AGASTYA – DATA LOADING, SELECTION & BASIC PREPROCESSING
# =============================================================================

def preview_stations(status_csv_path: str, n: int = 20):
    """
    Quick peek at station ids/names/capacity so you can pick STATION_ID.
    """
    import pandas as pd
    df = pd.read_csv(status_csv_path)
    cols = df.columns.tolist()
    if {"station_id","name","capacity"}.issubset(cols):
        out = df[["station_id","name","capacity"]].drop_duplicates().sort_values("station_id").head(n)
    elif {"number","name","bike_stands"}.issubset(cols):
        out = df[["number","name","bike_stands"]].drop_duplicates().sort_values("number").head(n)
    else:
        print("Columns:", cols)
        raise RuntimeError("Unrecognised schema; cannot preview.")
    print(out.to_string(index=False))


def load_and_prepare_data(
    csv_path: str,
    station_id: int,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Load Dublin Bikes status CSV (GBFS-style preferred) and return a tidy DataFrame
    for one station with: timestamp, available_bikes, capacity, hour, day_of_week,
    is_weekend, availability_rate.

    Supports:
    - GBFS monthly CSVs (columns: station_id, last_reported, num_bikes_available, capacity, ...)
    - Legacy JCDecaux-style CSVs (columns: number, last_update, available_bikes, bike_stands/available_bike_stands)
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    # --- GBFS (new) schema branch ---
    if {"station_id","last_reported","num_bikes_available"}.issubset(cols):
        # parse timestamp (strings like "2025-08-01 00:05:00")
        ts = pd.to_datetime(df["last_reported"], errors="coerce", utc=True)
        # if tz-aware, convert to Europe/Dublin and drop tz for simplicity
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Europe/Dublin").dt.tz_localize(None)
        df["timestamp"] = ts

        df["available_bikes"] = df["num_bikes_available"]
        # capacity exists in GBFS historical CSVs; use it directly
        if "capacity" in df.columns:
            df["capacity_use"] = df["capacity"]
        else:
            # very unlikely on these files, but keep a fallback
            if "num_docks_available" in df.columns:
                df["capacity_use"] = df["num_bikes_available"] + df["num_docks_available"]
            else:
                raise KeyError("GBFS file missing 'capacity' and 'num_docks_available'.")

        df_station = df[df["station_id"] == station_id].copy()
        keep = ["timestamp","available_bikes","capacity_use"]
        if df_station.empty:
            raise ValueError(f"No rows found for station_id={station_id}. Use preview_stations() to pick a valid id.")
        df_station = df_station[keep].rename(columns={"capacity_use":"capacity"})

    # --- Legacy JCDecaux (old) schema branch ---
    elif {"number","last_update","available_bikes"}.issubset(cols):
        ts = pd.to_datetime(df["last_update"], errors="coerce", utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Europe/Dublin").dt.tz_localize(None)
        df["timestamp"] = ts
        # capacity: prefer 'bike_stands'; otherwise derive
        if "bike_stands" in df.columns:
            df["capacity"] = df["bike_stands"]
        elif {"available_bikes","available_bike_stands"}.issubset(cols):
            df["capacity"] = df["available_bikes"] + df["available_bike_stands"]
        else:
            raise KeyError("Legacy file missing capacity info.")

        df_station = df[df["number"] == station_id].copy()
        keep = ["timestamp","available_bikes","capacity"]
        if df_station.empty:
            raise ValueError(f"No rows found for number={station_id}. Use preview_stations() to pick a valid id.")
        df_station = df_station[keep]

    else:
        raise RuntimeError(f"Unrecognised schema. Columns found: {sorted(cols)}")

    # Clean/derive common fields
    df_station = df_station.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df_station["hour"] = df_station["timestamp"].dt.hour
    df_station["day_of_week"] = df_station["timestamp"].dt.dayofweek
    df_station["is_weekend"] = df_station["day_of_week"] >= 5
    df_station["availability_rate"] = df_station["available_bikes"] / df_station["capacity"]
    return df_station


# =============================================================================
# JAKUB – POISSON & BINOMIAL MODELLING
# =============================================================================

def poisson_analysis(df_station: pd.DataFrame, window: str = "1h") -> dict:

    """
    Test whether the number of 'events' per time window follows a Poisson distribution.

    Here we approximate an event as an absolute change in available_bikes between snapshots.

    Parameters
    ----------
    df_station : pd.DataFrame
        Data for a single station.
    window : str
        Pandas offset alias, e.g. "1H" for one hour.

    Returns
    -------
    results : dict
        Contains mean, variance, chi-square statistic, p-value.
    """
    print("\n=== Poisson modelling of bike changes per window ===")

    # Compute absolute change in available_bikes
    station = df_station.copy()
    station["bike_change"] = station["available_bikes"].diff().abs()
    station = station.dropna(subset=["bike_change"])

    changes = station.set_index("timestamp")["bike_change"]
    # Sum of absolute changes per window
    counts_per_window = changes.resample(window).sum().fillna(0)

    mean_count = counts_per_window.mean()
    var_count = counts_per_window.var()

    print(f"Time window: {window}")
    print(f"Mean events per window (λ̂): {mean_count:.3f}")
    print(f"Variance: {var_count:.3f}")
    print(f"Variance/Mean ratio: {var_count / mean_count:.3f}")

    # Build observed frequencies
    observed_counts = counts_per_window.value_counts().sort_index()
    max_k = int(observed_counts.index.max())
    total_n = len(counts_per_window)

    observed = []
    expected = []
    acc_obs = 0.0
    acc_exp = 0.0

    for k in range(max_k + 1):
        obs_k = observed_counts.get(k, 0)
        exp_k = poisson.pmf(k, mean_count) * total_n

        acc_obs += obs_k
        acc_exp += exp_k

        # Combine small expected bins to ensure expected >= 5
        if acc_exp >= 5 or k == max_k:
            observed.append(acc_obs)
            expected.append(acc_exp)
            acc_obs = 0.0
            acc_exp = 0.0

    observed = np.array(observed)
    expected = np.array(expected)

    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value > 0.05:
        print("Result: cannot reject Poisson assumption (data consistent with Poisson).")
    else:
        print("Result: reject Poisson assumption at 5% level (data deviate from Poisson).")

    # Plot histogram + Poisson PMF
    plt.figure()
    plt.hist(counts_per_window.values, bins=range(0, max_k + 2), density=True,
             alpha=0.6, edgecolor="black", align="left", label="Observed")
    x = np.arange(0, max_k + 1)
    pmf = poisson.pmf(x, mean_count)
    plt.plot(x, pmf, "ro-", label=f"Poisson(λ={mean_count:.2f})")
    plt.xlabel("Events per window")
    plt.ylabel("Probability / density")
    plt.title("Poisson fit to events per window")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("02_poisson_fit.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "mean": mean_count,
        "variance": var_count,
        "chi2": chi2_stat,
        "p_value": p_value
    }


def binomial_analysis(df_station: pd.DataFrame) -> dict:
    """
    Compare observed occupancy distribution to Binomial(C, p̂).
    Uses chi-square on a truncated support (expected >= 5) and rescales expected to match observed sum.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binom
    from scipy import stats

    print("\n=== Binomial modelling of occupancy ===")

    # capacity as mode (constant in your file)
    C = int(df_station["capacity"].mode()[0])
    df_const = df_station[df_station["capacity"] == C].copy()

    k_vals, counts = np.unique(df_const["available_bikes"], return_counts=True)
    total_n = counts.sum()

    p_hat = df_const["available_bikes"].mean() / C
    print(f"Capacity C: {C}")
    print(f"Estimated occupancy probability p̂: {p_hat:.3f}")

    obs_probs = counts / total_n
    binom_probs = binom.pmf(k_vals, C, p_hat)
    expected_counts = binom_probs * total_n

    # keep bins with expected >= 5
    mask = expected_counts >= 5
    if mask.sum() >= 2:
        obs = counts[mask].astype(float)
        exp = expected_counts[mask].astype(float)

        # --- IMPORTANT: rescale expected to match observed total on the truncated support
        scale = obs.sum() / exp.sum()
        exp *= scale

        chi2, p_value = stats.chisquare(f_obs=obs, f_exp=exp)
        print(f"Chi-square (binomial): {chi2:.4f}, p-value: {p_value:.4f}")
    else:
        chi2, p_value = np.nan, np.nan
        print("Too few bins with expected >= 5 for a reliable chi-square test.")

    # Plot observed vs Binomial probabilities
    plt.figure()
    plt.bar(k_vals, obs_probs, width=0.8, alpha=0.6, edgecolor="black", label="Observed")
    plt.plot(k_vals, binom_probs, "ro-", label="Binomial model")
    plt.xlabel("Number of bikes in station")
    plt.ylabel("Probability")
    plt.title("Observed occupancy vs Binomial(C, p̂) model")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("03_binomial_fit.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "capacity": C,
        "p_hat": p_hat,
        "chi2": chi2,
        "p_value": p_value
    }


# --- Quick descriptive summary + one simple plot ---
def descriptive_summary(df_station):
    import pandas as pd
    import matplotlib.pyplot as plt

    print("=== Descriptive summary for chosen station ===")
    print(f"Rows: {len(df_station)}")
    print(f"Time range: {df_station['timestamp'].min()}  ->  {df_station['timestamp'].max()}")
    print(df_station[["available_bikes", "capacity", "availability_rate"]].describe())

    # Histogram of availability rate
    plt.figure()
    plt.hist(df_station["availability_rate"].dropna(), bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Availability rate (available_bikes / capacity)")
    plt.ylabel("Frequency")
    plt.title("Histogram of station availability rate")
    plt.tight_layout()
    # plt.show()
    plt.savefig("01_hist_availability.png", dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# FIONN – CONDITIONAL PROBABILITIES & TEMPORAL PATTERNS
# =============================================================================

def add_low_availability_flags(df_station: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Add 'low_availability' and 'time_period' (Peak/Off-Peak) columns.

    Low availability = availability_rate < threshold.
    Peak hours = 7–9 and 17–19 (you can adjust).
    """
    df = df_station.copy()
    df["low_availability"] = df["availability_rate"] < threshold

    def time_category(h):
        if 7 <= h <= 9 or 17 <= h <= 19:
            return "Peak"
        else:
            return "Off-Peak"

    df["time_period"] = df["hour"].apply(time_category)
    df["day_type"] = df["is_weekend"].map({False: "Weekday", True: "Weekend"})

    return df


def conditional_probabilities(df_station_flags: pd.DataFrame) -> dict:
    """
    Compute conditional probabilities of low availability and make plots.

    Returns a dict with main probabilities you might quote in the report.
    """
    print("\n=== Conditional probabilities of low availability ===")

    df = df_station_flags

    p_low = df["low_availability"].mean()
    print(f"P(Low availability) overall: {p_low:.3f} ({p_low*100:.1f}%)")

    # Peak vs Off-Peak
    probs_time = df.groupby("time_period")["low_availability"].mean()
    print("\nBy time period (Peak / Off-Peak):")
    for period, val in probs_time.items():
        print(f"P(Low | {period}) = {val:.3f} ({val*100:.1f}%)")

    # Weekday vs Weekend
    probs_day = df.groupby("day_type")["low_availability"].mean()
    print("\nBy day type (Weekday / Weekend):")
    for day_type, val in probs_day.items():
        print(f"P(Low | {day_type}) = {val:.3f} ({val*100:.1f}%)")

    # Hour-of-day profile
    hourly = df.groupby("hour")["low_availability"].mean()

    # Plot 1: P(Low | hour)
    plt.figure()
    plt.bar(hourly.index, hourly.values, edgecolor="black")
    plt.axhline(p_low, color="red", linestyle="--", label=f"Overall ({p_low:.2f})")
    plt.xlabel("Hour of day")
    plt.ylabel("P(Low availability)")
    plt.title("Probability of low availability by hour")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout(); plt.savefig("04_hourly_low_prob.png", dpi=200, bbox_inches="tight"); plt.close()

    # Plot 2: Peak vs Off-Peak
    plt.figure()
    plt.bar(probs_time.index, probs_time.values, edgecolor="black")
    plt.ylabel("P(Low availability)")
    plt.title("Peak vs Off-Peak probability of low availability")
    plt.tight_layout(); plt.savefig("05_peak_offpeak.png", dpi=200, bbox_inches="tight"); plt.close()




    # Plot 3: Weekday vs Weekend
    plt.figure()
    plt.bar(probs_day.index, probs_day.values, edgecolor="black")
    plt.ylabel("P(Low availability)")
    plt.title("Weekday vs Weekend probability of low availability")
    plt.tight_layout(); plt.savefig("06_weekday_weekend.png", dpi=200, bbox_inches="tight"); plt.close()

    # Chi-square: low_availability vs time_period (independence)
    contingency = pd.crosstab(df["time_period"], df["low_availability"])
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    print("\nChi-square test: H0 = low availability independent of Peak/Off-Peak")
    print(f"Chi2: {chi2:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: reject H0 (time period matters).")
    else:
        print("Result: cannot reject H0 (no strong evidence of dependence).")

    return {
        "p_low": p_low,
        "p_low_peak": probs_time.get("Peak", np.nan),
        "p_low_offpeak": probs_time.get("Off-Peak", np.nan),
        "p_low_weekday": probs_day.get("Weekday", np.nan),
        "p_low_weekend": probs_day.get("Weekend", np.nan),
        "chi2_time": chi2,
        "p_value_time": p_value
    }


# =============================================================================
# ISAAC – INDEPENDENCE, AUTOCORRELATION & LAW OF LARGE NUMBERS
# =============================================================================

def autocorrelation_lag1(x: np.ndarray) -> float:
    """
    Quick lag-1 autocorrelation.
    """
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def independence_and_lln(df_station_flags: pd.DataFrame) -> None:
    """
    - Show temporal dependence via lag-1 autocorrelation.
    - Show Law of Large Numbers via running mean of low_availability.
    """
    print("\n=== Independence & Law of Large Numbers ===")

    df = df_station_flags.sort_values("timestamp").reset_index(drop=True)

    # Temporal dependence (autocorrelation of available_bikes)
    series = df["available_bikes"].values
    if len(series) > 2:
        rho1 = autocorrelation_lag1(series)
        print(f"Lag-1 autocorrelation of available_bikes: {rho1:.3f}")
    else:
        print("Not enough data for autocorrelation.")

    # Law of Large Numbers for low_availability
    low = df["low_availability"].astype(int)
    running_mean = low.expanding().mean()
    true_prob = low.mean()

    print(f"True P(Low availability) (full sample): {true_prob:.3f}")

    # Show a few sample points
    for n in [100, 500, 1000, len(df)]:
        if n <= len(df):
            approx = low.iloc[:n].mean()
            print(f"n={n:5d}: running estimate = {approx:.3f}, error = {abs(approx - true_prob):.3f}")

    # Plot convergence
    plt.figure()
    plt.plot(running_mean.values, label="Running mean of 1{Low}")
    plt.axhline(true_prob, color="red", linestyle="--", label=f"True probability ({true_prob:.3f})")
    plt.xlabel("Sample size n")
    plt.ylabel("Estimated P(Low availability)")
    plt.title("Law of Large Numbers: convergence of relative frequency")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("07_lln_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()


def rush_hour_probs(df_station):
    import pandas as pd
    d = df_station.copy()
    d["is_empty"] = (d["available_bikes"] == 0)
    d["weekday"]  = d["day_of_week"] < 5
    d["morning"]  = d["hour"].between(7, 9, inclusive="both")
    d["evening"]  = d["hour"].between(16, 19, inclusive="both")
    out = {}
    for label, mask in {
        "weekday_morning": d["weekday"] & d["morning"],
        "weekday_evening": d["weekday"] & d["evening"],
        "weekend_morning": ~d["weekday"] & d["morning"],
        "weekend_evening": ~d["weekday"] & d["evening"],
        "overall": pd.Series(True, index=d.index)
    }.items():
        subset = d[mask]
        if len(subset):
            out[label] = {
                "n": len(subset),
                "P_empty": subset["is_empty"].mean(),
                "P_low20": (subset["available_bikes"]/subset["capacity"] < 0.2).mean()
            }
    return out


# =============================================================================
# MAIN – PUTTING IT ALL TOGETHER
# =============================================================================

def main():
    DATA_PATH   = "/Users/agastyakataria/prob-project/dublinbikes.csv"   # your 08/2025 status file
    STATIONS    = "/Users/agastyakataria/prob-project/dublin.csv"        # optional (not strictly needed)
    STATION_ID  = 1   # e.g., CLARENDON ROW from preview_stations()
    START_DATE  = None
    END_DATE    = None
    LOW_THRESHOLD = 0.2

    df_station = load_and_prepare_data(
        csv_path=DATA_PATH,
        station_id=STATION_ID,
        start_date=START_DATE,
        end_date=END_DATE
    )

    descriptive_summary(df_station)
    poisson_results = poisson_analysis(df_station, window="1H")
    binom_results   = binomial_analysis(df_station)

    df_flags   = add_low_availability_flags(df_station, threshold=LOW_THRESHOLD)
    cond_res   = conditional_probabilities(df_flags)
    independence_and_lln(df_flags)
    
    print("\n--- RESULTS ---")
    print("Poisson:", poisson_results)
    print("Binomial:", binom_results)
    print("Conditionals:", cond_res)    
    print("Rush-hour probs:", rush_hour_probs(df_station))



if __name__ == "__main__":
    main()
