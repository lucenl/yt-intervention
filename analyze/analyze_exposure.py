#!/usr/bin/env python3
"""
analyze_recommendations.py

For each (puppet JSON + prediction CSV) pair it will:
  1) Save results_<label>.csv
  2) Combine into all_puppet_analysis_results.csv
  3) Build harmful_content_summary.xlsx with binary and multiclass stats
"""

import json
import pandas as pd


def analyze_puppet_recommendations(puppet, video_data, top_n=None):
    """
    Count harmful vs total and category distribution in homepage, upnext, and combined.
    - top_n is None → homepage[:25], upnext[:12]
    - top_n=8 or 4 → homepage[:top_n], upnext[:top_n]
    """
    pid = puppet["puppet_id"]
    intended = puppet["harmful_percentage"]

    raw_hp = puppet.get("homepage_recs", [])
    raw_un = puppet.get("upnext_recs",   [])

    if top_n is None:
        homepage = raw_hp[:25]
        upnext = raw_un[:12]
    else:
        homepage = raw_hp[:top_n]
        upnext = raw_un[:top_n]

    all_recs = homepage + upnext

    # Binary analysis (harmful/harmless)
    hp_h = sum(video_data.get(v, {}).get("binary_prediction", 0) for v in homepage)
    un_h = sum(video_data.get(v, {}).get("binary_prediction", 0) for v in upnext)
    all_h = hp_h + un_h

    hp_pct = round(hp_h / len(homepage) * 100, 2) if homepage else 0
    un_pct = round(un_h / len(upnext) * 100, 2) if upnext else 0
    all_pct = round(all_h / len(all_recs) * 100, 2) if all_recs else 0

    hp_miss = sum(1 for v in homepage if v not in video_data)
    un_miss = sum(1 for v in upnext if v not in video_data)

    # Multiclass analysis (category distribution among harmful videos)
    def get_category_dist(videos):
        categories = [video_data.get(v, {}).get("harmful_category") for v in videos if video_data.get(v, {}).get("binary_prediction", 0) == 1]
        if not categories:
            return {"total_harmful": 0, "categories": {}}
        total_harmful = len(categories)
        category_counts = pd.Series(categories).value_counts().to_dict()
        category_pcts = {cat: round(count / total_harmful * 100, 2) for cat, count in category_counts.items()}
        return {"total_harmful": total_harmful, "categories": category_pcts}

    hp_cat_dist = get_category_dist(homepage)
    un_cat_dist = get_category_dist(upnext)
    all_cat_dist = get_category_dist(all_recs)

    return {
        "puppet_id": pid,
        "intended_harmful_percentage": intended,
        "top_n": top_n,  # None, 8, or 4
        "homepage_total": len(homepage),
        "homepage_harmful": hp_h,
        "homepage_missing": hp_miss,
        "homepage_harmful_percentage": hp_pct,
        "upnext_total": len(upnext),
        "upnext_harmful": un_h,
        "upnext_missing": un_miss,
        "upnext_harmful_percentage": un_pct,
        "all_total": len(all_recs),
        "all_harmful": all_h,
        "all_missing": hp_miss + un_miss,
        "all_harmful_percentage": all_pct,
        "homepage_categories": hp_cat_dist["categories"],
        "homepage_total_harmful": hp_cat_dist["total_harmful"],
        "upnext_categories": un_cat_dist["categories"],
        "upnext_total_harmful": un_cat_dist["total_harmful"],
        "all_categories": all_cat_dist["categories"],
        "all_total_harmful": all_cat_dist["total_harmful"]
    }


def process_file_pair(label, json_path, csv_path, top_n_values):
    print(f"→ Processing {label}")
    df_pred = pd.read_csv(csv_path)
    # Create a dictionary mapping video_id to a dict of binary_prediction and harmful_category
    vh = {row["video_id"]: {"binary_prediction": row["binary_prediction"], "harmful_category": row["harmful_category"]}
          for row in df_pred[["video_id", "binary_prediction", "harmful_category"]].to_dict('records')}

    with open(json_path) as f:
        data = json.load(f)
    puppets = data if isinstance(data, list) else [data]

    rows = []
    for p in puppets:
        for n in top_n_values:
            rows.append(analyze_puppet_recommendations(p, vh, n))

    df = pd.DataFrame(rows)
    out_csv = f"results_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  saved intermediate → {out_csv}")
    return df


def generate_excel_table(results_df, output_file="harmful_content_summary_v2.xlsx"):
    # Labels for unified headers
    hp_label = {None: "Top 25", 8: "Top 8", 4: "Top 4"}
    un_label = {None: "Top 12", 8: "Top 8", 4: "Top 4"}
    all_label = {None: "Top 37", 8: "Top 16", 4: "Top 8"}

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        wb = writer.book

        hdr_fmt = wb.add_format({
            'bold': True, 'text_wrap': True,
            'valign': 'center', 'align': 'center',
            'bg_color': '#D9D9D9', 'border': 1
        })
        cell_fmt = wb.add_format({
            'valign': 'center', 'align': 'center', 'border': 1
        })

        # --- Unified sheet (Binary Harmful Percentage) ---
        ws_binary = wb.add_worksheet("Binary_Percentage")
        writer.sheets["Binary_Percentage"] = ws_binary

        # Write header row for binary percentages
        headers = ["Training %"]
        for n in [None, 8, 4]:
            headers.append(f"Homepage {hp_label[n]} %")
        for n in [None, 8, 4]:
            headers.append(f"Up-Next {un_label[n]} %")
        for n in [None, 8, 4]:
            headers.append(f"All {all_label[n]} %")
        ws_binary.write_row(0, 0, headers, hdr_fmt)

        # Write data rows for binary percentages
        row = 1
        for hp in sorted(results_df["intended_harmful_percentage"].unique()):
            ws_binary.write(int(row), 0, f"{hp}%", cell_fmt)
            col = 1
            for section, col_key in [
                ("homepage", "homepage_harmful_percentage"),
                ("upnext",   "upnext_harmful_percentage"),
                ("all",      "all_harmful_percentage")
            ]:
                for n in [None, 8, 4]:
                    if n is None:
                        mask = results_df["top_n"].isna()
                    else:
                        mask = results_df["top_n"] == n

                    sub = results_df[
                        (results_df["intended_harmful_percentage"] == hp) & mask
                    ]

                    if not sub.empty:
                        vals = sub[col_key]
                        m, mi, ma = vals.mean(), vals.min(), vals.max()
                        ws_binary.write(int(row), int(col), f"{m:.2f} ({mi:.2f}-{ma:.2f})", cell_fmt)
                    else:
                        ws_binary.write(int(row), int(col), "—", cell_fmt)
                    col += 1
            row += 1

        # --- Multiclass Category Distribution Sheet ---
        ws_category = wb.add_worksheet("Category_Distribution")
        writer.sheets["Category_Distribution"] = ws_category

        # Determine the top 3 unique category values (e.g., 0, 1, 2)
        all_categories = []
        for col in ["homepage_categories", "upnext_categories", "all_categories"]:
            for d in results_df[col]:
                if d:
                    all_categories.extend(d.keys())
        unique_categories = sorted(set(all_categories))[:3]  # Take top 3 categories

        # Write header row for category distribution
        headers = ["Training %"]
        for n in [None, 8, 4]:
            for cat in unique_categories:
                headers.append(f"Homepage {hp_label[n]} category {cat}")
        for n in [None, 8, 4]:
            for cat in unique_categories:
                headers.append(f"Up-Next {un_label[n]} category {cat}")
        for n in [None, 8, 4]:
            for cat in unique_categories:
                headers.append(f"All {all_label[n]} category {cat}")
        ws_category.write_row(0, 0, headers, hdr_fmt)

        # Write data rows for category distribution
        row = 1
        for hp in sorted(results_df["intended_harmful_percentage"].unique()):
            ws_category.write(int(row), 0, f"{hp}%", cell_fmt)
            col = 1
            for section, cat_col in [
                ("homepage", "homepage_categories"),
                ("upnext",   "upnext_categories"),
                ("all",      "all_categories")
            ]:
                for n in [None, 8, 4]:
                    if n is None:
                        mask = results_df["top_n"].isna()
                    else:
                        mask = results_df["top_n"] == n

                    sub = results_df[
                        (results_df["intended_harmful_percentage"] == hp) & mask
                    ]

                    if not sub.empty:
                        cat_pcts = {cat: [] for cat in unique_categories}
                        for _, row_data in sub.iterrows():
                            if row_data[cat_col]:
                                for cat in unique_categories:
                                    cat_pcts[cat].append(row_data[cat_col].get(cat, 0))

                        for cat in unique_categories:
                            if cat_pcts[cat]:
                                m = sum(cat_pcts[cat]) / len(cat_pcts[cat]) if len(cat_pcts[cat]) > 0 else 0
                                mi = min(cat_pcts[cat]) if cat_pcts[cat] else 0
                                ma = max(cat_pcts[cat]) if cat_pcts[cat] else 0
                                ws_category.write(int(row), int(col), f"{m:.2f} ({mi:.2f}-{ma:.2f})", cell_fmt)
                            else:
                                ws_category.write(int(row), int(col), "—", cell_fmt)
                            col += 1
                    else:
                        for _ in unique_categories:
                            ws_category.write(int(row), int(col), "—", cell_fmt)
                            col += 1
            row += 1

        # --- Detailed sheets per top_n ---
        for n in [None, 8, 4]:
            name = hp_label[n]
            if n is None:
                df_n = results_df[results_df["top_n"].isna()]
            else:
                df_n = results_df[results_df["top_n"] == n]

            summary = df_n.groupby("intended_harmful_percentage").agg({
                "homepage_harmful_percentage": ["mean", "min", "max", "count"],
                "upnext_harmful_percentage":   ["mean", "min", "max", "count"],
                "all_harmful_percentage":      ["mean", "min", "max", "count"]
            }).round(2)

            summary.to_excel(writer, sheet_name=name)

    print(f"Excel summary saved → {output_file}")
    return output_file


def main():
    # Update these three to your actual file locations:
    data_config = [
        ("all", "./success_puppets.json", "./results_details_all_puppets_with_predictions.csv"),
    ]
    top_n_values = [None, 8, 4]

    all_dfs = []
    for label, jpath, cpath in data_config:
        df = process_file_pair(label, jpath, cpath, top_n_values)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("all_puppet_analysis_results.csv", index=False)
    print("Combined CSV saved → all_puppet_analysis_results.csv")

    generate_excel_table(combined)


if __name__ == "__main__":
    main()