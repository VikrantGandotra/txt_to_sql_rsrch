import sqlite3
import re
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import difflib

class DatabaseGraphPruner:
    """
    Deterministic table-level pruner over declared SQLite foreign keys.

    Join paths are guaranteed to follow declared FKs within each connected
    component. Disconnected seed tables have no guaranteed join path.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.graph = nx.Graph()
        self.table_columns: Dict[str, List[str]] = {}
        self.primary_keys: Dict[str, Set[str]] = {}
        self.foreign_keys: Dict[str, Set[str]] = {}
        self.column_metadata: Dict[str, Dict[str, tuple]] = {}
        self.primary_key_order: Dict[str, List[str]] = {}
        self.unique_keys: Dict[str, List[List[str]]] = {}
        # A simple graph stores connectivity only. Constraints need their own
        # list to preserve composite keys, self references and parallel FKs.
        self.foreign_key_constraints: Dict[str, List[dict]] = {}
        self.original_ddl: Dict[str, str] = {}
        self.has_hidden_columns: Set[str] = set()
        self._build_schema_graph()

    @staticmethod
    def _quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _build_schema_graph(self):
        # Schema inspection must never create an empty database on a typo.
        database_uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(database_uri, uri=True)) as conn:
            cursor = conn.cursor()

            # 1. Fetch tables
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
            table_rows = cursor.fetchall()
            tables = [row[0] for row in table_rows]
            self.original_ddl = {name: sql for name, sql in table_rows if sql}

            # Referenced tables may appear later in SQLite's creation order.
            for tbl in tables:
                self.graph.add_node(tbl)
                self.table_columns[tbl] = []
                self.primary_keys[tbl] = set()
                self.foreign_keys[tbl] = set()
                self.column_metadata[tbl] = {}
                self.primary_key_order[tbl] = []
                self.unique_keys[tbl] = []
                self.foreign_key_constraints[tbl] = []

            for tbl in tables:
                # Column metadata & primary keys
                cursor.execute(f"PRAGMA table_info({self._quote(tbl)});")
                columns = cursor.fetchall()
                cursor.execute(f"PRAGMA table_xinfo({self._quote(tbl)});")
                if any(row[6] for row in cursor.fetchall()):
                    self.has_hidden_columns.add(tbl)
                for col in columns:
                    # col: (cid, name, type, notnull, dflt_value, pk)
                    col_name = col[1]
                    is_pk = bool(col[5])
                    self.table_columns[tbl].append(col_name)
                    self.column_metadata[tbl][col_name] = col
                    if is_pk:
                        self.primary_keys[tbl].add(col_name)
                self.primary_key_order[tbl] = [
                    col[1] for col in sorted(columns, key=lambda col: col[5]) if col[5]
                ]
                # Referenced non-PK keys must remain valid FK targets. Partial
                # and expression indexes do not define a table-wide key.
                cursor.execute(f"PRAGMA index_list({self._quote(tbl)});")
                for index in cursor.fetchall():
                    if not index[2] or index[3] == "pk" or index[4]:
                        continue
                    cursor.execute(f"PRAGMA index_info({self._quote(index[1])});")
                    index_columns = [row[2] for row in cursor.fetchall()]
                    if index_columns and all(column is not None for column in index_columns):
                        self.unique_keys[tbl].append(index_columns)

            # Read all columns before resolving omitted target PK names.
            table_names = {name.casefold(): name for name in tables}
            for tbl in tables:
                # Explicit Foreign Keys
                cursor.execute(f"PRAGMA foreign_key_list({self._quote(tbl)});")
                constraints = {}
                for fk in sorted(cursor.fetchall(), key=lambda row: (row[0], row[1])):
                    # fk: (id, seq, target_table, from_col, to_col, on_update, on_delete, match)
                    target_table = table_names.get(fk[2].casefold(), fk[2])
                    from_col = next(
                        (name for name in self.table_columns[tbl]
                         if name.casefold() == fk[3].casefold()), fk[3]
                    )
                    to_col = fk[4]
                    self.foreign_keys[tbl].add(from_col)
                    if to_col is None and target_table in self.primary_key_order:
                        target_keys = self.primary_key_order[target_table]
                        to_col = target_keys[fk[1]] if fk[1] < len(target_keys) else None
                    if target_table in tables and to_col is not None:
                        to_col = next(
                            (name for name in self.table_columns[target_table]
                             if name.casefold() == to_col.casefold()), to_col
                        )
                        self.foreign_keys[target_table].add(to_col)
                    constraint = constraints.setdefault(fk[0], {
                        "target": target_table, "columns": [], "target_columns": [],
                        "on_update": fk[5], "on_delete": fk[6], "match": fk[7],
                    })
                    constraint["columns"].append(from_col)
                    constraint["target_columns"].append(to_col)
                self.foreign_key_constraints[tbl] = list(constraints.values())
                for constraint in self.foreign_key_constraints[tbl]:
                    target = constraint["target"]
                    columns = constraint["columns"]
                    target_columns = constraint["target_columns"]
                    if target == tbl or target not in self.table_columns:
                        continue
                    if (not columns or any(col not in self.column_metadata[tbl] for col in columns)
                        or any(col not in self.column_metadata[target] for col in target_columns)):
                        continue
                    # SQLite can declare a foreign key pointing at a non-key
                    # column; it raises "foreign key mismatch" only on use.
                    # Such a declaration is not a usable join edge.
                    target_keys = [self.primary_key_order[target], *self.unique_keys[target]]
                    if target_columns not in target_keys:
                        continue
                    relationship = tuple(
                        (tbl, source, target, destination)
                        for source, destination in zip(columns, target_columns)
                    )
                    if self.graph.has_edge(tbl, target):
                        self.graph[tbl][target]["relationships"].append(relationship)
                    else:
                        self.graph.add_edge(tbl, target, weight=1.0,
                                            relationships=[relationship])


    def extract_seeds(self, question: str, evidence: str) -> Tuple[Set[str], Set[str]]:
        """
        Stage 2: Lexical and Fuzzy matching against table and column names.
        Handles plurals and slight typos (e.g., "student" -> "students") without needing an LLM.
        """
        text = f"{question} {evidence}".lower()
        tokens = set(re.findall(r"\b[a-z0-9_]+\b", text))
        
        # Filter out tiny stop words ('the', 'who', 'are') to avoid false positive fuzzy matches
        meaningful_tokens = {t for t in tokens if len(t) > 3}

        seed_tables = set()
        seed_columns = set()

        for tbl, cols in self.table_columns.items():
            tbl_clean = tbl.lower().replace("_", "")
            
            # Exact match OR Fuzzy Match (Similarity > 85%)
            if tbl.lower() in tokens or tbl_clean in tokens or any(difflib.SequenceMatcher(None, t, tbl_clean).ratio() > 0.85 for t in meaningful_tokens):
                seed_tables.add(tbl)

            for col in cols:
                col_clean = col.lower().replace("_", "")

                # A mention of "student" must not fuzzy-match an unrelated
                # table's student_id. Key-like columns need their full name.
                key_like = col.lower().endswith(("_id", "id", "_code", "code"))
                if (col.lower() in tokens or col_clean in tokens or
                    (not key_like and any(
                        difflib.SequenceMatcher(None, t, col_clean).ratio() > 0.85
                        for t in meaningful_tokens
                    ))):
                    seed_tables.add(tbl)
                    seed_columns.add(f"{tbl}.{col}")

        return seed_tables, seed_columns

    def prune(self, question: str, evidence: str) -> Tuple[Set[str], str]:
        """
        Retain declared-FK paths between lexical seeds plus their one-hop neighbors.
        """
        seed_tables, _ = self.extract_seeds(question, evidence)

        # Fallback: If no seeds matched, retain all nodes
        if not seed_tables:
            active_tables = set(self.graph.nodes)
        elif len(seed_tables) == 1:
            active_tables = set(seed_tables)
        else:
            active_tables = set()
            # Group seeds by connected component to prevent disconnection errors
            connected_components = list(nx.connected_components(self.graph))
            for comp in connected_components:
                terminals = seed_tables.intersection(comp)
                if not terminals:
                    continue
                if len(terminals) == 1:
                    active_tables.update(terminals)
                else:
                    # Kou-Markowsky-Berman (KMB) Steiner Tree approximation
                    subgraph = self.graph.subgraph(comp)
                    st = steiner_tree(subgraph, sorted(terminals), weight="weight")
                    active_tables.update(st.nodes)

        # One hop is measured from lexical seeds and FK path bridges only. Do
        # not repeatedly expand new neighbors into the entire component.
        relevant = set(active_tables)
        for table in relevant:
            active_tables.update(self.graph.neighbors(table))

        pruned_ddl = self._generate_pruned_ddl(active_tables)
        return active_tables, pruned_ddl

    def _generate_pruned_ddl(self, active_tables: Set[str]) -> str:
        ddl_parts = []
        for tbl in sorted(active_tables):
            if tbl in self.has_hidden_columns:
                # table_info omits generated/virtual hidden columns. Preserve
                # the source DDL instead of silently dropping those columns.
                ddl_parts.append(self.original_ddl[tbl].rstrip().rstrip(";") + ";")
                continue
            definitions = []
            for col in self.table_columns[tbl]:
                metadata = self.column_metadata[tbl][col]
                definition = self._quote(col)
                if metadata[2]:
                    definition += f" {metadata[2]}"
                if metadata[3]:
                    definition += " NOT NULL"
                if metadata[4] is not None:
                    definition += f" DEFAULT ({metadata[4]})"
                definitions.append(definition)
            primary_key = self.primary_key_order[tbl]
            if primary_key:
                definitions.append(
                    "PRIMARY KEY (" + ", ".join(map(self._quote, primary_key)) + ")"
                )
            for unique_key in self.unique_keys[tbl]:
                definitions.append(
                    "UNIQUE (" + ", ".join(map(self._quote, unique_key)) + ")"
                )
            for constraint in self.foreign_key_constraints[tbl]:
                if constraint["target"] not in active_tables:
                    continue
                definition = (
                    "FOREIGN KEY (" + ", ".join(map(self._quote, constraint["columns"]))
                    + ") REFERENCES " + self._quote(constraint["target"])
                )
                if all(col is not None for col in constraint["target_columns"]):
                    definition += " (" + ", ".join(
                        map(self._quote, constraint["target_columns"])
                    ) + ")"
                definition += f" ON UPDATE {constraint['on_update']} ON DELETE {constraint['on_delete']}"
                definitions.append(definition)
            formatted_cols = ",\n  ".join(definitions)
            ddl_parts.append(f"CREATE TABLE {self._quote(tbl)} (\n  {formatted_cols}\n);")

        return "\n\n".join(ddl_parts)
