import sqlite3
import time
import os
from src.graph_pruner import DatabaseGraphPruner

def setup_dummy_bird_db(db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Table A: students
    cursor.execute("""
    CREATE TABLE students (
        student_id INTEGER PRIMARY KEY,
        student_name TEXT,
        age INTEGER
    );
    """)

    # Table B: bridge table (enrollments) - missing explicit FOREIGN KEY clause!
    cursor.execute("""
    CREATE TABLE enrollments (
        enrollment_id INTEGER PRIMARY KEY,
        student_id INTEGER,
        course_id INTEGER,
        semester TEXT
    );
    """)

    # Table C: courses
    cursor.execute("""
    CREATE TABLE courses (
        course_id INTEGER PRIMARY KEY,
        course_name TEXT,
        credits INTEGER
    );
    """)

    conn.commit()
    conn.close()

def test_pruner():
    db_path = "test_bird_temp.db"
    setup_dummy_bird_db(db_path)

    try:
        pruner = DatabaseGraphPruner(db_path)

        # Question targets students and courses, skipping explicit mention of 'enrollments'
        question = "List the student names who are taking the Database Systems course."
        evidence = "course_name refers to courses."

        start_time = time.perf_counter()
        active_tables, ddl = pruner.prune(question, evidence)
        duration_ms = (time.perf_counter() - start_time) * 1000

        print(f"\nPruning Latency: {duration_ms:.3f} ms")
        print(f"Active Tables Discovered: {active_tables}")
        print("\n=== Reconstituted DDL ===")
        print(ddl)
        print("=========================")

        # Assertions
        assert "students" in active_tables, "students table should be detected as a seed."
        assert "courses" in active_tables, "courses table should be detected as a seed."
        assert "enrollments" in active_tables, "enrollments MUST be pulled in as a bridge table via implicit FK."
        assert duration_ms < 10.0, f"Pruning took {duration_ms:.2f} ms, expected < 10 ms."

        print("\n✅ Milestone 2 Verification Passed: Bridge table resolved and latency < 5-10ms!")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_pruner()