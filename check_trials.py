import sqlite3
from pathlib import Path

for db in sorted(Path("results/studies").glob("lstm_*.db")):
    con = sqlite3.connect(db)
    completed = con.execute(
        "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'"
    ).fetchone()[0]
    total = con.execute(
        "SELECT COUNT(*) FROM trials"
    ).fetchone()[0]
    print(f"{db.name}: {completed}/50 COMPLETE, {total} total")
    con.close()
