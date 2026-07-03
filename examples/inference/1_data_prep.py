"""Step 1 — Convert your database to RelBench format.

Reads the tables + schema from ``config.py``, attaches primary-key / foreign-key /
time-column metadata, and writes a RelBench database to disk. Nothing here is
task-specific; you're just describing the shape of your data.

    pixi run python examples/inference/1_data_prep.py
"""

import config
import pipeline


def main():
    print(f"[step 1] reading tables from {config.SQL_URI}")
    raw = pipeline.read_tables(config.SQL_URI, list(config.TABLES))
    db = pipeline.build_database(config.TABLES, raw)
    out = pipeline.save_database(config.DB_NAME, db)

    print(f"[step 1] wrote RelBench database '{config.DB_NAME}' -> {out}")
    for name, table in db.table_dict.items():
        print(
            f"   - {name}: {len(table.df)} rows | pkey={table.pkey_col} | "
            f"fkeys={table.fkey_col_to_pkey_table} | time={table.time_col}"
        )
    print("\nNext: define the prediction task with 2_task_prep.py")


if __name__ == "__main__":
    main()
