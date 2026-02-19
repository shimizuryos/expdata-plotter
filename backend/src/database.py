import sqlite3
import os
from contextlib import contextmanager

# Default database path (relative to the project root or backend dir as needed)
# In this project, let's keep it in backend/data/expdata.db or similar
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "../data/expdata.db")

def get_db_path() -> str:
    """
    Returns the database path, allowing override via environment variable for testing.
    """
    return os.environ.get("EXPDATA_DB_PATH", DEFAULT_DB_PATH)

def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.
    Enables foreign keys and row factory.
    """
    if db_path is None:
        db_path = get_db_path()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(db_path: str = None):
    """
    Initializes the database schema.
    Creates tables if they do not exist.
    """
    if db_path is None:
        db_path = get_db_path()

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # --- Samples Table ---
    # stores hierarchical structure in 'device_groups' JSON
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        device_type TEXT NOT NULL DEFAULT 'three_terminal_hanle',
        structures  TEXT NOT NULL,      -- JSON list of layers
        device_groups TEXT NOT NULL,    -- JSON list of device groups
        note        TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now'))
    );
    """)

    # --- Measurements Table ---
    # stores measurement data and metadata in JSON columns
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        id               TEXT PRIMARY KEY,
        sample_id        TEXT NOT NULL,
        device_id        TEXT NOT NULL,
        measurement_type TEXT NOT NULL,
        metadata         TEXT NOT NULL,     -- JSON
        data             TEXT DEFAULT NULL, -- JSON
        derived          TEXT DEFAULT NULL, -- JSON
        file_ref         TEXT DEFAULT NULL,
        measured_at      TEXT DEFAULT NULL,
        created_at       TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (sample_id) REFERENCES samples(id)
    );
    """)

    # --- Schema migrations (safe to re-run) ---
    try:
        cursor.execute("ALTER TABLE samples ADD COLUMN r_parasitic REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Indices for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_sample ON measurements(sample_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_device ON measurements(device_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_type   ON measurements(measurement_type);")

    conn.commit()
    conn.close()

@contextmanager
def get_db_cursor(commit=False, db_path: str = None):
    """
    Context manager for database cursor.
    """
    if db_path is None:
        db_path = get_db_path()

    conn = get_db_connection(db_path)
    try:
        yield conn.cursor()
        if commit:
            conn.commit()
    finally:
        conn.close()
