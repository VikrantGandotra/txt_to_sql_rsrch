import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.graph_pruner import DatabaseGraphPruner


class GraphPrunerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="mars_pruner_")
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "schema.sqlite"

    def create_database(self, ddl):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.executescript(ddl)
            connection.commit()

    def test_shared_id_names_do_not_create_undeclared_join_edges(self):
        self.create_database("""
            CREATE TABLE students (student_id INTEGER PRIMARY KEY, student_name TEXT, age INTEGER);
            CREATE TABLE enrollments (
                enrollment_id INTEGER PRIMARY KEY, student_id INTEGER,
                course_id INTEGER, semester TEXT
            );
            CREATE TABLE courses (course_id INTEGER PRIMARY KEY, course_name TEXT, credits INTEGER);
        """)
        pruner = DatabaseGraphPruner(str(self.database))
        active_tables, ddl = pruner.prune(
            "List the student names taking the Database Systems course.",
            "student_name refers to students; course_name refers to courses.",
        )
        self.assertEqual(active_tables, {"students", "courses"})
        self.assertEqual(pruner.graph.number_of_edges(), 0)
        self.assertIn('"student_name" TEXT', ddl)
        self.assertIn('"age" INTEGER', ddl)
        self.assertIn('"credits" INTEGER', ddl)
        self.assertIn('PRIMARY KEY ("student_id")', ddl)
        self.assertNotIn("FOREIGN KEY", ddl)
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.executescript(ddl)
            self.assertEqual(rendered.execute("PRAGMA foreign_key_list(students)").fetchall(), [])

    def test_declared_fk_path_and_one_hop_neighbor_retain_every_column(self):
        self.create_database("""
            CREATE TABLE students (
                id INTEGER PRIMARY KEY, name TEXT, age INTEGER, unused BLOB
            );
            CREATE TABLE enrollments (
                id INTEGER PRIMARY KEY, student_id INTEGER, course_id INTEGER,
                semester TEXT, FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (course_id) REFERENCES courses(id)
            );
            CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, credits INTEGER);
            CREATE TABLE unrelated (student_id INTEGER PRIMARY KEY, note TEXT);
        """)
        pruner = DatabaseGraphPruner(str(self.database))
        active, ddl = pruner.prune("Who is the oldest student?", "")
        self.assertEqual(active, {"students", "enrollments"})
        self.assertNotIn("unrelated", active)
        self.assertIn('"age" INTEGER', ddl)
        self.assertIn('"unused" BLOB', ddl)
        self.assertIn('"semester" TEXT', ddl)
        self.assertIn('FOREIGN KEY ("student_id") REFERENCES "students" ("id")', ddl)
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.execute("PRAGMA foreign_keys=ON")
            rendered.executescript(ddl)
            self.assertEqual(len(rendered.execute("PRAGMA foreign_key_list(enrollments)").fetchall()), 1)

        active, ddl = pruner.prune("student name in course title", "")
        self.assertEqual(active, {"students", "enrollments", "courses"})
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.executescript(ddl)
            self.assertEqual(len(rendered.execute("PRAGMA foreign_key_list(enrollments)").fetchall()), 2)

    def test_declared_but_unusable_foreign_key_does_not_create_join_edge(self):
        self.create_database("""
            CREATE TABLE parent (id INTEGER PRIMARY KEY, non_key TEXT);
            CREATE TABLE child (
                id INTEGER PRIMARY KEY, parent_value TEXT,
                FOREIGN KEY (parent_value) REFERENCES parent(non_key)
            );
        """)
        pruner = DatabaseGraphPruner(str(self.database))
        self.assertFalse(pruner.graph.has_edge("child", "parent"))
        active, ddl = pruner.prune("child", "")
        self.assertEqual(active, {"child"})
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.executescript(ddl)

    def test_composite_multiple_self_and_implicit_target_foreign_keys_round_trip(self):
        self.create_database("""
            CREATE TABLE child (
                child_id INTEGER PRIMARY KEY,
                first_a INTEGER, first_b INTEGER,
                second_a INTEGER, second_b INTEGER,
                parent_child INTEGER REFERENCES child(child_id),
                label TEXT,
                FOREIGN KEY (first_b, first_a) REFERENCES parent ON DELETE CASCADE,
                FOREIGN KEY (second_b, second_a) REFERENCES parent(b, a) ON UPDATE CASCADE
            );
            CREATE TABLE parent (
                a INTEGER, b INTEGER, name VARCHAR(40) NOT NULL,
                PRIMARY KEY (b, a)
            );
        """)
        pruner = DatabaseGraphPruner(str(self.database))
        _, ddl = pruner.prune("unmatched", "")
        self.assertIn('PRIMARY KEY ("b", "a")', ddl)
        self.assertIn('"name" VARCHAR(40) NOT NULL', ddl)
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.execute("PRAGMA foreign_keys=ON")
            rendered.executescript(ddl)
            rendered.execute("INSERT INTO parent VALUES (11, 22, 'P')")
            rendered.execute("INSERT INTO child VALUES (1, 11, 22, 11, 22, 1, 'C')")
            self.assertEqual(rendered.execute("PRAGMA foreign_key_check").fetchall(), [])
            foreign_keys = rendered.execute("PRAGMA foreign_key_list(child)").fetchall()
            self.assertEqual(len({row[0] for row in foreign_keys}), 3)
            self.assertEqual(len(foreign_keys), 5)
            self.assertEqual(
                {(row[2], row[3], row[4], row[5], row[6]) for row in foreign_keys},
                {
                    ("parent", "first_b", "b", "NO ACTION", "CASCADE"),
                    ("parent", "first_a", "a", "NO ACTION", "CASCADE"),
                    ("parent", "second_b", "b", "CASCADE", "NO ACTION"),
                    ("parent", "second_a", "a", "CASCADE", "NO ACTION"),
                    ("child", "parent_child", "child_id", "NO ACTION", "NO ACTION"),
                },
            )

    def test_pruning_retains_referenced_non_primary_keys_and_declared_unique_key(self):
        self.create_database("""
            CREATE TABLE schools (
                school_id INTEGER PRIMARY KEY,
                code CHAR(4) UNIQUE,
                title TEXT, unused BLOB
            );
            CREATE TABLE students (
                student_id INTEGER PRIMARY KEY, school_code CHAR(4), name TEXT,
                FOREIGN KEY (school_code) REFERENCES schools(code)
            );
        """)
        pruner = DatabaseGraphPruner(str(self.database))
        _, ddl = pruner.prune("students name schools title", "")
        self.assertIn('"code" CHAR(4)', ddl)
        self.assertIn('UNIQUE ("code")', ddl)
        self.assertIn('"unused" BLOB', ddl)
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.execute("PRAGMA foreign_keys=ON")
            rendered.executescript(ddl)
            rendered.execute("INSERT INTO schools VALUES (1, 'ABCD', 'S', NULL)")
            rendered.execute("INSERT INTO students VALUES (1, 'ABCD', 'A')")
            self.assertEqual(rendered.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_quoted_identifiers_and_expression_defaults_are_valid_sql(self):
        self.create_database('''
            CREATE TABLE "strange`table" (
                "key" INTEGER PRIMARY KEY,
                "full""name" TEXT NOT NULL DEFAULT 'anonymous',
                "created" TEXT DEFAULT (datetime('now'))
            );
        ''')
        _, ddl = DatabaseGraphPruner(str(self.database)).prune("unmatched", "")
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.executescript(ddl)
            rendered.execute('INSERT INTO "strange`table" ("key") VALUES (1)')
            row = rendered.execute('SELECT * FROM "strange`table"').fetchone()
            self.assertEqual(row[:2], (1, "anonymous"))
            self.assertIsNotNone(row[2])

    def test_generated_column_is_not_silently_dropped(self):
        self.create_database("""
            CREATE TABLE measurements (
                id INTEGER PRIMARY KEY, width INTEGER, height INTEGER,
                area INTEGER GENERATED ALWAYS AS (width * height) STORED
            );
        """)
        active, ddl = DatabaseGraphPruner(str(self.database)).prune("measurements", "")
        self.assertEqual(active, {"measurements"})
        self.assertIn("area INTEGER GENERATED ALWAYS", ddl)
        with closing(sqlite3.connect(":memory:")) as rendered:
            rendered.executescript(ddl)
            rendered.execute("INSERT INTO measurements(id, width, height) VALUES (1, 3, 4)")
            self.assertEqual(rendered.execute("SELECT area FROM measurements").fetchone(), (12,))

    def test_missing_database_is_not_created(self):
        with self.assertRaises(sqlite3.OperationalError):
            DatabaseGraphPruner(str(self.database))
        self.assertFalse(self.database.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
