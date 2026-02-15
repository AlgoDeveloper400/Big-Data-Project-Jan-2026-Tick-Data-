from trino.dbapi import connect
import time

# ----------------------------
# CONFIG
# ----------------------------
TRINO_HOST       = "localhost"
TRINO_PORT       = 8081
TRINO_USER       = "trino"

# ----------------------------
# CONNECTION
# We re-open the connection after rollback to force a fresh catalog view.
# Trino caches the current snapshot pointer — a new connection clears that.
# ----------------------------
def get_connection():
    for i in range(10):
        try:
            conn = connect(host=TRINO_HOST, port=TRINO_PORT, user=TRINO_USER,
                           catalog="iceberg", schema="gold")
            return conn
        except Exception as e:
            if i < 9:
                print(f"[INFO] Retrying... ({i+1}/10)")
                time.sleep(5)
            else:
                raise e

def fresh_cursor():
    """Always call this after a rollback to get a clean Trino connection."""
    conn = get_connection()
    return conn, conn.cursor()

# ----------------------------
# GET TABLES
# ----------------------------
def get_gold_tables(cursor):
    cursor.execute("SHOW TABLES FROM iceberg.gold")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"[INFO] Trino gold tables: {tables}")
    return tables

# ----------------------------
# SNAPSHOTS
# element_at() safely returns NULL if key is missing in the summary map
# ----------------------------
def get_snapshots(cursor, table):
    cursor.execute(f"""
        SELECT
            snapshot_id,
            committed_at,
            operation,
            element_at(summary, 'added-records')   as added,
            element_at(summary, 'deleted-records')  as deleted,
            element_at(summary, 'total-records')    as total
        FROM iceberg.gold."{table}$snapshots"
        ORDER BY committed_at ASC
    """)
    return cursor.fetchall()

def get_current_snapshot(cursor, table):
    """Returns the snapshot_id Trino is currently pointing at."""
    cursor.execute(f"""
        SELECT snapshot_id
        FROM iceberg.gold."{table}$snapshots"
        ORDER BY committed_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    return row[0] if row else None

def print_snapshots(snapshots, current_id=None):
    print(f"\n  {'#':<4} {'Snapshot ID':<22} {'Committed At':<26} {'Operation':<12} "
          f"{'Added':>10} {'Deleted':>10} {'Total':>12}")
    print(f"  {'-'*102}")
    for i, snap in enumerate(snapshots):
        snap_id   = snap[0]
        committed = snap[1]
        operation = snap[2] or "unknown"
        added     = snap[3] or "0"
        deleted   = snap[4] or "0"
        total     = snap[5] or "?"

        markers = []
        if i == len(snapshots) - 1:
            markers.append("latest")
        if snap_id == current_id:
            markers.append("CURRENT")
        marker_str = "  <- " + ", ".join(markers) if markers else ""

        print(f"  {i:<4} {str(snap_id):<22} {str(committed):<26} {operation:<12} "
              f"{added:>10} {deleted:>10} {total:>12}{marker_str}")

# ----------------------------
# PARTITION VIEW
# Shows row count per year — the core "what data do I have" check
# Can query at a specific snapshot or current state
# ----------------------------
def show_partitions(cursor, table, snapshot_id=None):
    if snapshot_id:
        print(f"\n  [PARTITIONS] '{table}' at snapshot {snapshot_id}:")
        sql = f"""
            SELECT year(datetime) as yr, COUNT(*) as rows
            FROM iceberg.gold."{table}"
            FOR VERSION AS OF {snapshot_id}
            GROUP BY year(datetime)
            ORDER BY yr
        """
    else:
        print(f"\n  [PARTITIONS] '{table}' current state:")
        sql = f"""
            SELECT year(datetime) as yr, COUNT(*) as rows
            FROM iceberg.gold."{table}"
            GROUP BY year(datetime)
            ORDER BY yr
        """
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        if rows:
            total = sum(r[1] for r in rows)
            print(f"  {'Year':<8} {'Rows':>14}")
            print(f"  {'-'*24}")
            for row in rows:
                print(f"  {str(row[0]):<8} {row[1]:>14,}")
            print(f"  {'-'*24}")
            print(f"  {'TOTAL':<8} {total:>14,}")
        else:
            print("  No data found")
    except Exception as e:
        print(f"  [ERROR] Partition check failed: {e}")

# ----------------------------
# QUERY AT SNAPSHOT
# ----------------------------
def query_at_snapshot(cursor, table, snapshot_id, limit=10):
    print(f"\n  [QUERY] '{table}' at snapshot {snapshot_id} (first {limit} rows):")
    try:
        cursor.execute(f"""
            SELECT * FROM iceberg.gold."{table}"
            FOR VERSION AS OF {snapshot_id}
            LIMIT {limit}
        """)
        rows = cursor.fetchall()
        if rows:
            print(f"  {'datetime':<28} {'bid':>12} {'ask':>12}")
            print(f"  {'-'*54}")
            for row in rows:
                print(f"  {str(row[0]):<28} {row[1]:>12.5f} {row[2]:>12.5f}")
        else:
            print("  No rows found at this snapshot")
        return rows
    except Exception as e:
        print(f"  [ERROR] Query failed: {e}")
        return []

# ----------------------------
# QUERY AT TIMESTAMP
# ----------------------------
def query_at_timestamp(cursor, table, ts_str, limit=10):
    print(f"\n  [QUERY] '{table}' as of {ts_str} (first {limit} rows):")
    try:
        cursor.execute(f"""
            SELECT * FROM iceberg.gold."{table}"
            FOR TIMESTAMP AS OF TIMESTAMP '{ts_str}'
            LIMIT {limit}
        """)
        rows = cursor.fetchall()
        if rows:
            print(f"  {'datetime':<28} {'bid':>12} {'ask':>12}")
            print(f"  {'-'*54}")
            for row in rows:
                print(f"  {str(row[0]):<28} {row[1]:>12.5f} {row[2]:>12.5f}")
        else:
            print("  No rows found at this timestamp")
        return rows
    except Exception as e:
        print(f"  [ERROR] Query failed: {e}")
        return []

# ----------------------------
# DIFF TWO SNAPSHOTS
# ----------------------------
def diff_snapshots(cursor, table, snap_id_a, snap_id_b):
    print(f"\n  [DIFF] Comparing snapshot {snap_id_a} -> {snap_id_b}")
    try:
        cursor.execute(f'SELECT COUNT(*) FROM iceberg.gold."{table}" FOR VERSION AS OF {snap_id_a}')
        count_a = cursor.fetchone()[0]
        cursor.execute(f'SELECT COUNT(*) FROM iceberg.gold."{table}" FOR VERSION AS OF {snap_id_b}')
        count_b = cursor.fetchone()[0]
        delta   = count_b - count_a
        sign    = "+" if delta >= 0 else ""
        print(f"  Snapshot {snap_id_a} : {count_a:>12,} rows")
        print(f"  Snapshot {snap_id_b} : {count_b:>12,} rows")
        print(f"  Delta              : {sign}{delta:,} rows")
    except Exception as e:
        print(f"  [ERROR] Diff failed: {e}")

# ----------------------------
# ROLLBACK TO SNAPSHOT
# Step-by-step:
#   1. Show current partitions BEFORE rollback
#   2. Execute rollback_to_snapshot — Iceberg writes a new snapshot
#      pointing the table HEAD back to the target snapshot's data files
#   3. Close and reopen connection — forces Trino to re-read the catalog
#      pointer instead of using the cached current snapshot
#   4. Show partitions AFTER rollback so you can confirm the data changed
# ----------------------------
def rollback_to_snapshot(conn, cursor, table, snapshot_id):
    print(f"\n  [ROLLBACK] Target snapshot: {snapshot_id}")

    # Step 1 — show what we have NOW before touching anything
    print("\n  State BEFORE rollback:")
    show_partitions(cursor, table)

    # Step 2 — confirm
    confirm = input("\n  WARNING: Table will point to the state at that snapshot. Confirm? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("  Cancelled.")
        return conn, cursor

    # Step 3 — execute rollback
    try:
        cursor.execute(f"""
            CALL iceberg.system.rollback_to_snapshot('gold', '{table}', {snapshot_id})
        """)
        print(f"  [SUCCESS] Rollback executed for snapshot {snapshot_id}")
    except Exception as e:
        print(f"  [ERROR] Rollback failed: {e}")
        return conn, cursor

    # Step 4 — close old connection and open a fresh one
    # This is critical: Trino caches the snapshot pointer on the connection.
    # Without a fresh connection the partition query will still show old data.
    print("  [INFO] Refreshing Trino connection to apply rollback...")
    try:
        cursor.close()
        conn.close()
    except Exception:
        pass
    time.sleep(2)  # brief pause to let Iceberg finish writing the new snapshot metadata

    conn, cursor = fresh_cursor()
    print("  [INFO] Connection refreshed")

    # Step 5 — show partitions after rollback so user can see the difference
    print("\n  State AFTER rollback:")
    show_partitions(cursor, table)

    # Step 6 — show updated snapshot list so user can see the new rollback entry
    snapshots = get_snapshots(cursor, table)
    current   = get_current_snapshot(cursor, table)
    print(f"\n  Updated snapshot history (newest rollback entry at bottom):")
    print_snapshots(snapshots, current_id=current)

    return conn, cursor

# ----------------------------
# INTERACTIVE HELPERS
# ----------------------------
def pick_from_list(label, items):
    for i, item in enumerate(items):
        print(f"  {i}) {item}")
    while True:
        try:
            idx = int(input(f"\n  Select {label} (0-{len(items)-1}): "))
            if 0 <= idx < len(items):
                return items[idx]
            print("  Invalid selection, try again")
        except ValueError:
            print("  Please enter a number")

def pick_snapshot_index(snapshots, label):
    while True:
        try:
            idx = int(input(f"\n  Select snapshot # for {label} (0-{len(snapshots)-1}): "))
            if 0 <= idx < len(snapshots):
                return snapshots[idx]
            print("  Invalid selection, try again")
        except ValueError:
            print("  Please enter a number")

# ----------------------------
# MENU
# ----------------------------
def menu(conn, cursor, table, snapshots):
    while True:
        current = get_current_snapshot(cursor, table)
        print(f"""
{'='*60}
  TABLE   : {table}
  CURRENT : snapshot {current}
  HISTORY : {len(snapshots)} snapshots
{'='*60}
  1) List snapshots
  2) Show current partitions
  3) Query at snapshot
  4) Query at timestamp
  5) Diff two snapshots
  6) Rollback to snapshot
  7) Switch table
  0) Exit
""")
        choice = input("  Choice: ").strip()

        if choice == "1":
            print_snapshots(snapshots, current_id=current)

        elif choice == "2":
            show_partitions(cursor, table)

        elif choice == "3":
            print_snapshots(snapshots, current_id=current)
            snap  = pick_snapshot_index(snapshots, "query")
            limit = input("  Row limit (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            query_at_snapshot(cursor, table, snap[0], limit)

        elif choice == "4":
            ts    = input("  Enter timestamp (YYYY-MM-DD HH:MM:SS): ").strip()
            limit = input("  Row limit (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            query_at_timestamp(cursor, table, ts, limit)

        elif choice == "5":
            print_snapshots(snapshots, current_id=current)
            snap_a = pick_snapshot_index(snapshots, "snapshot A (older)")
            snap_b = pick_snapshot_index(snapshots, "snapshot B (newer)")
            diff_snapshots(cursor, table, snap_a[0], snap_b[0])

        elif choice == "6":
            print_snapshots(snapshots, current_id=current)
            snap = pick_snapshot_index(snapshots, "rollback target")
            conn, cursor = rollback_to_snapshot(conn, cursor, table, snap[0])
            snapshots    = get_snapshots(cursor, table)  # refresh list

        elif choice == "7":
            return "switch", conn, cursor

        elif choice == "0":
            return "exit", conn, cursor

        else:
            print("  Invalid choice")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    conn, cursor = fresh_cursor()
    print("[INFO] Connected to Trino")

    tables = get_gold_tables(cursor)
    if not tables:
        print("[FATAL] No tables found in gold schema. Exiting.")
        conn.close()
        exit(1)

    while True:
        print(f"\n{'='*60}")
        print("  Available tables:")
        table = pick_from_list("table", tables)

        snapshots = get_snapshots(cursor, table)
        if not snapshots:
            print(f"[WARNING] No snapshots found for '{table}'")
            continue

        current = get_current_snapshot(cursor, table)
        print_snapshots(snapshots, current_id=current)
        show_partitions(cursor, table)

        result, conn, cursor = menu(conn, cursor, table, snapshots)
        if result == "exit":
            break

    cursor.close()
    conn.close()
    print("\n[DONE] Goodbye!")