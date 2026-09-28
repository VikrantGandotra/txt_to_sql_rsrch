import sqlite3
import re
from contextlib import closing
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import difflib

class DatabaseGraphPruner:
    """
    Deterministic CPU-based Graph Pruner for SQLite schemas.
    Replaces RL Grounding agents by discovering Steiner subgraphs over
    explicit and implicit foreign-key relations.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.graph = nx.Graph()
        self.table_columns: Dict[str, List[str]] = {}
        self.primary_keys: Dict[str, Set[str]] = {}
        self.foreign_keys: Dict[str, Set[str]] = {}
        self._build_schema_graph()

    def _build_schema_graph(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()

            # 1. Fetch tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]

            # Referenced tables may appear later in SQLite's creation order.
            for tbl in tables:
                self.graph.add_node(tbl)
                self.table_columns[tbl] = []
                self.primary_keys[tbl] = set()
                self.foreign_keys[tbl] = set()

            for tbl in tables:
                # Column metadata & primary keys
                cursor.execute(f"PRAGMA table_info(`{tbl}`);")
                for col in cursor.fetchall():
                    # col: (cid, name, type, notnull, dflt_value, pk)
                    col_name = col[1]
                    is_pk = bool(col[5])
                    self.table_columns[tbl].append(col_name)
                    if is_pk:
                        self.primary_keys[tbl].add(col_name)

                # Explicit Foreign Keys
                cursor.execute(f"PRAGMA foreign_key_list(`{tbl}`);")
                for fk in cursor.fetchall():
                    # fk: (id, seq, target_table, from_col, to_col, on_update, on_delete, match)
                    target_table = fk[2]
                    from_col = fk[3]
                    to_col = fk[4]
                    if target_table in tables and target_table != tbl:
                        self.foreign_keys[tbl].add(from_col)
                        self.foreign_keys[target_table].add(to_col)
                        self.graph.add_edge(
                            tbl,
                            target_table,
                            weight=1.0,
                            relationship=f"{tbl}.{from_col} = {target_table}.{to_col}"
                        )

        # 2. Add implicit foreign key edges for missing constraints
        self._infer_implicit_foreign_keys(tables)

    def _infer_implicit_foreign_keys(self, tables: List[str]):
        """
        Catches the 5-10% of BIRD schemas lacking explicit foreign key constraints.
        Connects tables sharing columns ending in 'id', '_id', or identical PK names.
        """
        table_count = len(tables)
        for i in range(table_count):
            for j in range(i + 1, table_count):
                tbl_a = tables[i]
                tbl_b = tables[j]
                
                # If an explicit edge already exists, continue
                if self.graph.has_edge(tbl_a, tbl_b):
                    continue

                cols_a = {c.lower(): c for c in self.table_columns[tbl_a]}
                cols_b = {c.lower(): c for c in self.table_columns[tbl_b]}

                shared_keys = set(cols_a.keys()) & set(cols_b.keys())
                for key in shared_keys:
                    # Match keys like 'student_id', 'school_code', 'id' or identical primary keys
                    is_id_like = key.endswith("_id") or key.endswith("id") or key.endswith("code")
                    is_pk_match = (
                        cols_a[key] in self.primary_keys[tbl_a] or 
                        cols_b[key] in self.primary_keys[tbl_b]
                    )

                    if is_id_like or is_pk_match:
                        self.graph.add_edge(
                            tbl_a,
                            tbl_b,
                            weight=1.2,  # Slight penalty over explicit constraints
                            relationship=f"{tbl_a}.{cols_a[key]} = {tbl_b}.{cols_b[key]}"
                        )
                        self.foreign_keys[tbl_a].add(cols_a[key])
                        self.foreign_keys[tbl_b].add(cols_b[key])


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
                
                # Exact match OR Fuzzy Match (Similarity > 85%)
                if col.lower() in tokens or col_clean in tokens or any(difflib.SequenceMatcher(None, t, col_clean).ratio() > 0.85 for t in meaningful_tokens):
                    seed_tables.add(tbl)
                    seed_columns.add(f"{tbl}.{col}")

        return seed_tables, seed_columns

    def prune(self, question: str, evidence: str) -> Tuple[Set[str], str]:
        """
        Runs Steiner Tree approximation to retain minimal subgraphs and produces DDL.
        """
        seed_tables, seed_columns = self.extract_seeds(question, evidence)

        # Fallback: If no seeds matched, retain all nodes
        if not seed_tables:
            active_tables = set(self.graph.nodes)
        elif len(seed_tables) == 1:
            active_tables = seed_tables
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
                    st = steiner_tree(subgraph, list(terminals), weight="weight")
                    active_tables.update(st.nodes)

        # Generate minimal DDL projection
        pruned_ddl = self._generate_pruned_ddl(active_tables, seed_columns)
        return active_tables, pruned_ddl

    def _generate_pruned_ddl(self, active_tables: Set[str], seed_columns: Set[str]) -> str:
        ddl_parts = []
        for tbl in sorted(active_tables):
            cols_to_keep = []
            for col in self.table_columns[tbl]:
                col_ref = f"{tbl}.{col}"
                # Retain column if: matched seed, primary key, or foreign key join key
                if (col_ref in seed_columns or 
                    col in self.primary_keys[tbl] or 
                    col in self.foreign_keys[tbl] or 
                    len(seed_columns) == 0):
                    cols_to_keep.append(col)

            # Fallback: if table is a bridge table with no seed columns, keep PKs & FKs
            if not cols_to_keep:
                cols_to_keep = list(self.primary_keys[tbl].union(self.foreign_keys[tbl]))
                if not cols_to_keep:
                    cols_to_keep = self.table_columns[tbl][:3]

            formatted_cols = ",\n  ".join([f"`{c}`" for c in cols_to_keep])
            ddl_parts.append(f"CREATE TABLE `{tbl}` (\n  {formatted_cols}\n);")

        return "\n\n".join(ddl_parts)
