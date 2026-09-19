import json
import time
from pathlib import Path

import pandas as pd

from pipeline import analyze_report


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "safesight_dev_1000.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "safesight_analyzed.csv"


# --------------------------------------------------
# Save one successful result immediately
# --------------------------------------------------

def save_result(result):

    result_df = pd.DataFrame([result])

    if OUTPUT_FILE.exists():
        result_df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        result_df.to_csv(
            OUTPUT_FILE,
            mode="w",
            header=True,
            index=False
        )


# --------------------------------------------------
# Load IDs that have already been processed
# --------------------------------------------------

def get_completed_ids():

    if not OUTPUT_FILE.exists():
        return set()

    existing_df = pd.read_csv(OUTPUT_FILE)

    return set(
        existing_df["report_id"]
        .astype(str)
        .tolist()
    )


# --------------------------------------------------
# Batch Processor
# --------------------------------------------------

def process_reports(limit=20):

    df = pd.read_csv(INPUT_FILE).head(limit)

    completed_ids = get_completed_ids()

    print("\n====================================")
    print("        SafeSight AI Batch")
    print("====================================")
    print(f"Reports selected: {len(df)}")
    print(f"Already completed: {len(completed_ids)}")
    print()

    successful = 0
    failed = 0

    for position, (_, row) in enumerate(df.iterrows(), start=1):

        report_id = str(row["report_id"])

        # Skip reports already saved
        if report_id in completed_ids:
            print(
                f"[{position}/{len(df)}] "
                f"Report {report_id} already processed — skipping."
            )
            continue

        print(
            f"[{position}/{len(df)}] "
            f"Processing report {report_id}..."
        )

        try:

            analysis = analyze_report(row["report_text"])

            parsed = analysis["parsed_report"]
            risk = analysis["risk_assessment"]

            result = {
                "report_id": report_id,
                "event_date": row["event_date"],
                "city": row["city"],
                "state": row["state"],
                "industry_code": row["industry_code"],
                "report_text": row["report_text"],

                "hazard_type": parsed["hazard_type"],

                "risk_factors": json.dumps(
                    parsed["risk_factors"]
                ),

                "location_context":
                    parsed["location_context"],

                "equipment_involved": json.dumps(
                    parsed["equipment_involved"]
                ),

                "people_at_risk": json.dumps(
                    parsed["people_at_risk"]
                ),

                "potential_consequence":
                    parsed["potential_consequence"],

                "risk_level":
                    risk["risk_level"],

                "confidence":
                    risk["confidence"],

                "reasoning":
                    risk["reasoning"],

                "recommended_action":
                    risk["recommended_action"]
            }

            # SAVE IMMEDIATELY
            save_result(result)

            completed_ids.add(report_id)

            successful += 1

            print(
                f"    ✓ {risk['risk_level']} "
                f"(confidence {risk['confidence']:.2f})"
            )

        except Exception as e:

            error_message = str(e)

            # Quota exhausted — stop instead of wasting requests
            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):
                print("\n⚠ Gemini quota exhausted.")
                print("Stopping batch safely.")
                print("All successful reports have already been saved.")
                break

            # Temporary server issue
            elif (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                print("    ⚠ Gemini temporarily unavailable.")
                print("    Waiting 10 seconds and trying once more...")

                time.sleep(10)

                try:

                    analysis = analyze_report(row["report_text"])

                    parsed = analysis["parsed_report"]
                    risk = analysis["risk_assessment"]

                    result = {
                        "report_id": report_id,
                        "event_date": row["event_date"],
                        "city": row["city"],
                        "state": row["state"],
                        "industry_code": row["industry_code"],
                        "report_text": row["report_text"],
                        "hazard_type": parsed["hazard_type"],
                        "risk_factors": json.dumps(
                            parsed["risk_factors"]
                        ),
                        "location_context":
                            parsed["location_context"],
                        "equipment_involved": json.dumps(
                            parsed["equipment_involved"]
                        ),
                        "people_at_risk": json.dumps(
                            parsed["people_at_risk"]
                        ),
                        "potential_consequence":
                            parsed["potential_consequence"],
                        "risk_level":
                            risk["risk_level"],
                        "confidence":
                            risk["confidence"],
                        "reasoning":
                            risk["reasoning"],
                        "recommended_action":
                            risk["recommended_action"]
                    }

                    save_result(result)
                    completed_ids.add(report_id)
                    successful += 1

                    print(
                        f"    ✓ Retry successful: "
                        f"{risk['risk_level']}"
                    )

                except Exception as retry_error:

                    retry_message = str(retry_error)

                    if (
                        "429" in retry_message
                        or "RESOURCE_EXHAUSTED" in retry_message
                    ):
                        print("\n⚠ Gemini quota exhausted during retry.")
                        print("Stopping safely.")
                        break

                    print(f"    ✗ Retry failed: {retry_error}")
                    failed += 1

            else:
                print(f"    ✗ ERROR: {e}")
                failed += 1

        # Avoid hammering the API
        time.sleep(2)

    print("\n====================================")
    print("          Batch Summary")
    print("====================================")
    print(f"New successful reports: {successful}")
    print(f"Failed reports: {failed}")

    if OUTPUT_FILE.exists():

        output_df = pd.read_csv(OUTPUT_FILE)

        print(f"Total cached reports: {len(output_df)}")

        if not output_df.empty:
            print("\nRisk distribution:")
            print(output_df["risk_level"].value_counts())


if __name__ == "__main__":
    process_reports(limit=20)