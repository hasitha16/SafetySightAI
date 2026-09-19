from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "safesight_incidents_clean.csv"


def load_safety_data():
    """Load the cleaned SafeSight historical incident dataset."""

    df = pd.read_csv(
        DATA_FILE,
        dtype={"industry_code": str},
        low_memory=False
    )

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce"
    )

    # Clean categorical text fields
    text_columns = [
        "city",
        "state",
        "industry_code",
        "injury_nature",
        "body_part",
        "event_type",
        "incident_source"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    return df


def get_top_incident_types(df, top_n=10):
    """Return the most frequently reported incident types."""

    return (
        df["event_type"]
        .fillna("Unknown")
        .value_counts()
        .head(top_n)
        .reset_index(name="count")
        .rename(columns={"event_type": "incident_type"})
    )


def get_top_locations(df, top_n=10):
    """Return locations with the highest number of reports."""

    return (
        df.groupby(["state", "city"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )


def get_top_injury_types(df, top_n=10):
    """Return the most common injury types."""

    return (
        df["injury_nature"]
        .fillna("Unknown")
        .value_counts()
        .head(top_n)
        .reset_index(name="count")
        .rename(columns={"injury_nature": "injury_type"})
    )


def get_top_sources(df, top_n=10):
    """Return the most common incident sources."""

    return (
        df["incident_source"]
        .fillna("Unknown")
        .value_counts()
        .head(top_n)
        .reset_index(name="count")
        .rename(columns={"incident_source": "source"})
    )


def get_state_patterns(df, top_n=10):
    """Return states with the highest number of reports."""

    return (
        df["state"]
        .fillna("Unknown")
        .value_counts()
        .head(top_n)
        .reset_index(name="count")
        .rename(columns={"state": "state"})
    )


def get_serious_outcomes(df):
    """Calculate serious outcome statistics."""

    total_reports = len(df)

    hospitalized = int(df["hospitalized"].sum())
    amputations = int(df["amputation"].sum())
    loss_of_eye = int(df["loss_of_eye"].sum())

    return {
        "total_reports": total_reports,
        "hospitalized": hospitalized,
        "amputations": amputations,
        "loss_of_eye": loss_of_eye,
        "hospitalization_rate": round(
            hospitalized / total_reports * 100, 2
        ) if total_reports else 0
    }


def get_yearly_trends(df):
    """Return report counts by year."""

    valid_dates = df.dropna(subset=["event_date"]).copy()

    valid_dates["year"] = valid_dates["event_date"].dt.year

    return (
        valid_dates.groupby("year")
        .size()
        .reset_index(name="report_count")
        .sort_values("year")
    )

def detect_emerging_incident_types(
    df,
    recent_months=12,
    previous_months=12,
    min_recent_reports=20,
    top_n=10
):
    """
    Detect incident types that increased in the most recent time window
    compared with the preceding equivalent time window.
    """

    data = df.dropna(
        subset=["event_date", "event_type"]
    ).copy()

    if data.empty:
        return pd.DataFrame()

    latest_date = data["event_date"].max()

    recent_start = latest_date - pd.DateOffset(
        months=recent_months
    )

    previous_start = recent_start - pd.DateOffset(
        months=previous_months
    )

    recent = data[
        (data["event_date"] > recent_start)
        & (data["event_date"] <= latest_date)
    ]

    previous = data[
        (data["event_date"] > previous_start)
        & (data["event_date"] <= recent_start)
    ]

    recent_counts = (
        recent["event_type"]
        .value_counts()
        .rename("recent_count")
    )

    previous_counts = (
        previous["event_type"]
        .value_counts()
        .rename("previous_count")
    )

    trends = pd.concat(
        [recent_counts, previous_counts],
        axis=1
    ).fillna(0)

    trends["recent_count"] = (
        trends["recent_count"].astype(int)
    )

    trends["previous_count"] = (
        trends["previous_count"].astype(int)
    )

    trends = trends[
        trends["recent_count"] >= min_recent_reports
    ].copy()

    # Difference in number of reports
    trends["change"] = (
        trends["recent_count"]
        - trends["previous_count"]
    )

    # Percentage increase
    trends["growth_percent"] = (
        trends["change"]
        / trends["previous_count"].replace(0, 1)
        * 100
    ).round(1)

    # Only keep increasing patterns
    trends = trends[
        trends["change"] > 0
    ]

    trends = (
        trends
        .sort_values(
            ["growth_percent", "recent_count"],
            ascending=False
        )
        .head(top_n)
        .reset_index()
        .rename(
            columns={"event_type": "incident_type"}
        )
    )

    return trends

def detect_recurring_patterns(df, top_n=10):
    """
    Run the main SafeSight cross-report pattern analysis.
    """

    

    return {
        "incident_types": get_top_incident_types(df, top_n),
        "locations": get_top_locations(df, top_n),
        "injury_types": get_top_injury_types(df, top_n),
        "incident_sources": get_top_sources(df, top_n),
        "states": get_state_patterns(df, top_n),
        "serious_outcomes": get_serious_outcomes(df),
        "yearly_trends": get_yearly_trends(df),
        "emerging_incidents": detect_emerging_incident_types(
            df,
            top_n=top_n
        ),
    }


if __name__ == "__main__":

    print("\n====================================")
    print("   SafeSight AI Pattern Detector")
    print("====================================")

    data = load_safety_data()

    print(f"\nLoaded {len(data):,} safety reports.")

    patterns = detect_recurring_patterns(data)

    print("\n--- Serious Outcomes ---")

    for key, value in patterns["serious_outcomes"].items():
        print(f"{key}: {value}")

    print("\n--- Top Incident Types ---")
    print(patterns["incident_types"].to_string(index=False))

    print("\n--- Top Locations ---")
    print(patterns["locations"].to_string(index=False))

    print("\n--- Top Injury Types ---")
    print(patterns["injury_types"].to_string(index=False))

    print("\n--- Top Incident Sources ---")
    print(patterns["incident_sources"].to_string(index=False))

    print("\n--- Top States ---")
    print(patterns["states"].to_string(index=False))

    print("\n--- Yearly Trends ---")
    print(patterns["yearly_trends"].to_string(index=False))

    print("\n--- Emerging Incident Precursors ---")
    print(
        patterns["emerging_incidents"]
        .to_string(index=False)
    )