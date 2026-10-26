#!/usr/bin/env python
"""Collect the JSON files written by the test tools into a Markdown table.

Example:
    python tools/summarize_results.py work_dirs/results/*.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="summarise evaluation results")
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    rows = [json.loads(Path(p).read_text()) for p in sorted(args.results)]
    print("| Setting | Dice / F1 | ASSD / mAP | MCC | IoU |")
    print("|---|---|---|---|---|")
    for r in rows:
        m = r["metrics"]
        if "Dice" in m:
            print(f"| {r['setting']} | {m['Dice']:.2f} | {m['ASSD']:.2f} | - | - |")
        else:
            print(f"| {r['setting']} | {m['F1']:.2f} | {m['mAP']:.2f} | {m['MCC']:.2f} | {m['IoU']:.2f} |")


if __name__ == "__main__":
    main()
