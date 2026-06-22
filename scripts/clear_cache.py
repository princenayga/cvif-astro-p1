import pandas as pd
import re
import os
import shutil

df = pd.read_csv("transit_counts.csv")
errors = df[df["status"] != "ok"]

deleted = set()
for status in errors["status"]:
    # Extract Windows-style paths from error messages
    matches = re.findall(r"C:[/\\][^\s\n]+", status)
    for m in matches:
        # Normalize slashes
        m2 = m.replace("\\\\", os.sep).replace("/", os.sep)
        parent = os.path.dirname(m2)
        if parent and parent not in deleted and os.path.exists(parent):
            deleted.add(parent)
            shutil.rmtree(parent, ignore_errors=True)
            print(f"Deleted: {parent}")

print(f"\nDeleted {len(deleted)} corrupt cache directories")

# Remove error rows so count_transits.py will re-process them
ok_rows = df[df["status"] == "ok"]
ok_rows.to_csv("transit_counts.csv", index=False)
print(f"Removed {len(errors)} error rows from transit_counts.csv")
print(f"Remaining (ok): {len(ok_rows)} planets")
