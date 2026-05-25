#!/usr/bin/env python3
"""
=======================================================================
PANDAIBESY - COGNITIVE CLI ENGINE
nalar_bridge.py v1.0
=======================================================================
Offline-first, local-first Cognitive OS for terminal agents.
Solves AI amnesia across sessions via persistent local memory.

Usage:
  python nalar_bridge.py capture "your thought" [project]
  python nalar_bridge.py query "search term"
  python nalar_bridge.py mcp-pull "active context"
  python nalar_bridge.py list [project]
  python nalar_bridge.py export-logs
  python nalar_bridge.py stats
=======================================================================
"""

import os
import re
import sys
import json
import time
import sqlite3
import uuid
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────

VERSION      = "1.0.0"
DB_PATH      = Path.home() / ".pandaibesy" / "memories.db"
SCORE_FLOOR  = 0.05   # minimum Jaccard score to count as a match
TOP_K        = 5      # max results returned per query
SCAN_LIMIT   = 500    # max rows scanned per search (battery safety)

STOPWORDS = {
    "a","an","the","is","it","in","on","for","of","and","to","was",
    "are","be","with","as","at","by","from","or","that","this","i",
    "my","we","you","your","me","do","did","not","so","its","if",
    "but","about","have","has","had","can","will","would","should",
    "what","how","when","where","why","which","who","ke","di","dan",
    "yang","ini","itu","untuk","dari","dengan","pada","adalah","ada",
    "juga","bisa","akan","tidak","lebih","saya","kita","kami","apa"
}


# ── DATABASE ──────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    """Open (or create) the database. Returns a ready connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    conn.execute("PRAGMA synchronous=NORMAL") # faster writes on mobile
    conn.execute("PRAGMA temp_store=MEMORY")  # avoid temp file I/O
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection):
    """Create tables if they don't exist. Safe to call every startup."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            content      TEXT    NOT NULL,
            source       TEXT    NOT NULL DEFAULT 'capture',
            tokens       TEXT    NOT NULL DEFAULT '[]',
            embedding    BLOB,
            project_tag  TEXT,
            session_id   TEXT    NOT NULL DEFAULT '',
            recall_score REAL    NOT NULL DEFAULT 0.0,
            created_at   INTEGER NOT NULL,
            updated_at   INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text   TEXT    NOT NULL,
            source       TEXT    NOT NULL DEFAULT 'manual_query',
            matched_ids  TEXT    NOT NULL DEFAULT '[]',
            result_count INTEGER NOT NULL DEFAULT 0,
            latency_ms   INTEGER,
            logged_at    INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_mem_created
            ON memories(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_mem_source
            ON memories(source);
        CREATE INDEX IF NOT EXISTS idx_mem_project
            ON memories(project_tag);
        CREATE INDEX IF NOT EXISTS idx_log_source
            ON query_log(source);
    """)
    conn.commit()


# ── TOKENIZER ─────────────────────────────────────────────────────────

def tokenize(text: str) -> set:
    """
    Clean and tokenize text into a set of meaningful words.
    Removes stopwords, punctuation, and short tokens.
    Supports Bahasa Indonesia and English.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)   # replace punctuation with space
    text = re.sub(r'\d+', '', text)         # remove pure numbers
    tokens = {
        w for w in text.split()
        if w not in STOPWORDS and len(w) > 2
    }
    return tokens


# ── SEARCH ────────────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    """Standard Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def search(conn: sqlite3.Connection, query: str,
           source_label: str = "manual_query",
           project: str = None) -> list:
    """
    Search memories by Jaccard similarity.
    Logs the query silently for retention tracking.
    Returns top-K results sorted by score descending.
    """
    t0 = int(time.time() * 1000)
    q_tokens = tokenize(query)

    if not q_tokens:
        return []

    # Build query — optionally filter by project
    if project:
        rows = conn.execute(
            "SELECT id, content, source, tokens, project_tag "
            "FROM memories "
            "WHERE project_tag = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (project, SCAN_LIMIT)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, content, source, tokens, project_tag "
            "FROM memories "
            "ORDER BY created_at DESC LIMIT ?",
            (SCAN_LIMIT,)
        ).fetchall()

    # Score each memory
    scored = []
    for row in rows:
        mem_tokens = set(json.loads(row[3]))
        score = jaccard(q_tokens, mem_tokens)
        if score >= SCORE_FLOOR:
            scored.append({
                "id":          row[0],
                "content":     row[1],
                "source":      row[2],
                "project_tag": row[4],
                "score":       score
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:TOP_K]

    # Silent retention log
    latency = int(time.time() * 1000) - t0
    conn.execute(
        "INSERT INTO query_log "
        "(query_text, source, matched_ids, result_count, latency_ms, logged_at) "
        "VALUES (?,?,?,?,?,?)",
        (
            query,
            source_label,
            json.dumps([r["id"] for r in results]),
            len(results),
            latency,
            int(time.time() * 1000)
        )
    )
    conn.commit()
    return results


# ── CAPTURE ───────────────────────────────────────────────────────────

def capture(conn: sqlite3.Connection, content: str,
            source: str = "capture",
            project: str = None) -> int:
    """
    Store a new memory. Returns the new row ID.
    Tokenizes at write time so search is fast at read time.
    """
    now    = int(time.time() * 1000)
    tokens = json.dumps(list(tokenize(content)))
    cur    = conn.execute(
        "INSERT INTO memories "
        "(content, source, tokens, project_tag, session_id, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (content, source, tokens, project, str(uuid.uuid4()), now, now)
    )
    conn.commit()
    return cur.lastrowid


# ── DISPLAY ───────────────────────────────────────────────────────────

def print_results(results: list, label: str = "Memory"):
    """Pretty-print search results to terminal."""
    if not results:
        print("[!] No matching memories found.")
        return
    for i, r in enumerate(results, 1):
        tag = f" | {r['project_tag']}" if r['project_tag'] else ""
        print(f"\n[{i}] Score: {r['score']*100:.1f}% | {r['source']}{tag}")
        print("─" * 52)
        # Truncate long content in terminal for readability
        content = r['content']
        if len(content) > 400:
            content = content[:400] + "... [truncated]"
        print(content)
    print("─" * 52)


def print_header():
    print("=" * 52)
    print("   PANDAIBESY — Cognitive CLI  v" + VERSION)
    print("   Offline. Local. Persistent.")
    print("=" * 52)


# ── CLI ───────────────────────────────────────────────────────────────

def cmd_capture(conn, args):
    if len(args) < 1:
        print("Usage: capture 'text' [project_tag]")
        sys.exit(1)
    content = args[0]
    project = args[1] if len(args) > 1 else None
    mem_id  = capture(conn, content, source="capture", project=project)
    tag_str = f" [{project}]" if project else ""
    print(f"[✓] Memory #{mem_id} captured{tag_str}")


def cmd_query(conn, args):
    if len(args) < 1:
        print("Usage: query 'search term' [project_tag]")
        sys.exit(1)
    query   = args[0]
    project = args[1] if len(args) > 1 else None
    results = search(conn, query, source_label="manual_query", project=project)
    print_results(results)


def cmd_mcp_pull(conn, args):
    if len(args) < 1:
        print("Usage: mcp-pull 'active context'")
        sys.exit(1)
    results = search(conn, args[0], source_label="mcp_pull")
    if not results:
        print("[MCP] No relevant context found.")
        return
    # MCP output: clean XML block for agent injection
    print("<pandaibesy_context>")
    for r in results:
        tag = r['project_tag'] or 'global'
        print(f"[source:{r['source']} | project:{tag} | score:{r['score']:.2f}]")
        print(r['content'])
        print()
    print("</pandaibesy_context>")


def cmd_list(conn, args):
    project = args[0] if args else None
    if project:
        rows = conn.execute(
            "SELECT id, source, project_tag, created_at, "
            "SUBSTR(content, 1, 60) FROM memories "
            "WHERE project_tag = ? ORDER BY created_at DESC LIMIT 20",
            (project,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, source, project_tag, created_at, "
            "SUBSTR(content, 1, 60) FROM memories "
            "ORDER BY created_at DESC LIMIT 20"
        ).fetchall()

    if not rows:
        print("[!] No memories found.")
        return
    print(f"\n{'ID':<5} {'Source':<12} {'Project':<12} {'Preview'}")
    print("─" * 60)
    for row in rows:
        pid  = row[0]
        src  = row[1][:11]
        tag  = (row[2] or "─")[:11]
        prev = row[4].replace("\n", " ")
        print(f"{pid:<5} {src:<12} {tag:<12} {prev}")


def cmd_export_logs(conn, args):
    rows = conn.execute(
        "SELECT query_text, source, matched_ids, "
        "result_count, latency_ms, logged_at "
        "FROM query_log ORDER BY id ASC"
    ).fetchall()
    logs = [
        {
            "query":        r[0],
            "source":       r[1],
            "matched_ids":  json.loads(r[2]),
            "result_count": r[3],
            "latency_ms":   r[4],
            "logged_at":    r[5]
        }
        for r in rows
    ]
    print(json.dumps(logs, indent=2, ensure_ascii=False))


def cmd_stats(conn, args):
    total_mem = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    total_log = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    manual_q  = conn.execute(
        "SELECT COUNT(*) FROM query_log WHERE source='manual_query'"
    ).fetchone()[0]
    mcp_pulls = conn.execute(
        "SELECT COUNT(*) FROM query_log WHERE source='mcp_pull'"
    ).fetchone()[0]
    avg_lat   = conn.execute(
        "SELECT AVG(latency_ms) FROM query_log"
    ).fetchone()[0] or 0
    projects  = conn.execute(
        "SELECT project_tag, COUNT(*) FROM memories "
        "WHERE project_tag IS NOT NULL "
        "GROUP BY project_tag ORDER BY COUNT(*) DESC"
    ).fetchall()

    print("\n── Pandaibesy Stats ──────────────────────────────")
    print(f"  Total memories  : {total_mem}")
    print(f"  Total queries   : {total_log}")
    print(f"   ↳ Manual       : {manual_q}")
    print(f"   ↳ MCP pulls    : {mcp_pulls}")
    print(f"  Avg latency     : {avg_lat:.1f}ms")
    if projects:
        print(f"\n  Projects:")
        for p in projects:
            print(f"   ↳ {p[0]:<20} {p[1]} memories")
    print("──────────────────────────────────────────────────")


# ── INTERACTIVE MODE ──────────────────────────────────────────────────

def interactive_mode(conn):
    print_header()
    print(f"  DB: {DB_PATH}\n")

    while True:
        print("\n  1. Capture memory")
        print("  2. Query memories")
        print("  3. List memories")
        print("  4. Simulate MCP pull")
        print("  5. Stats")
        print("  6. Export logs")
        print("  7. Exit")
        choice = input("\nPilih (1-7): ").strip()

        if choice == "1":
            content = input("Tulis memori:\n> ").strip()
            if not content:
                continue
            project = input("Project tag (Enter untuk skip): ").strip() or None
            mem_id  = capture(conn, content, source="capture", project=project)
            print(f"[✓] Memory #{mem_id} disimpan")

        elif choice == "2":
            query = input("Query: ").strip()
            if not query:
                continue
            results = search(conn, query, source_label="manual_query")
            print_results(results)

        elif choice == "3":
            project = input("Filter project (Enter untuk semua): ").strip() or None
            cmd_list(conn, [project] if project else [])

        elif choice == "4":
            context = input("Active context: ").strip()
            if not context:
                continue
            results = search(conn, context, source_label="mcp_pull")
            print_results(results, label="MCP Context")

        elif choice == "5":
            cmd_stats(conn, [])

        elif choice == "6":
            cmd_export_logs(conn, [])

        elif choice == "7":
            print("\nPandaibesy dimatikan. Sampai jumpa.")
            break

        else:
            print("[!] Pilihan tidak valid.")


# ── ENTRYPOINT ────────────────────────────────────────────────────────

COMMANDS = {
    "capture":     cmd_capture,
    "query":       cmd_query,
    "mcp-pull":    cmd_mcp_pull,
    "list":        cmd_list,
    "export-logs": cmd_export_logs,
    "stats":       cmd_stats,
}


def main():
    conn = get_conn()

    if len(sys.argv) < 2:
        interactive_mode(conn)
        conn.close()
        return

    cmd  = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd == "--version" or cmd == "-v":
        print(f"Pandaibesy v{VERSION}")
        return

    if cmd == "--help" or cmd == "-h":
        print(__doc__)
        return

    if cmd not in COMMANDS:
        print(f"[!] Command tidak dikenal: '{cmd}'")
        print(f"    Tersedia: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    try:
        COMMANDS[cmd](conn, args)
    except KeyboardInterrupt:
        print("\n[!] Dibatalkan.")
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
