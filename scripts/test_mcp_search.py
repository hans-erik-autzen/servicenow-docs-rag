#!/usr/bin/env python3
"""
Quick smoke test for the MCP server — imports the tools directly
without launching a subprocess, so it works without a running server.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "mcp_server"))

from snow_docs_mcp import servicenow_docs_search

QUERY = "What are the limits for Now Assist prompt characters?"

print(f"Query: {QUERY}\n")
results = servicenow_docs_search(query=QUERY, max_results=3)

if not results:
    print("No results returned.")
    sys.exit(1)

for i, r in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"section_heading : {r['section_heading']}")
    print(f"product_area    : {r['product_area']}")
    print(f"source_url      : {r['source_url']}")
    print(f"text preview    : {r['text'][:300].strip()}…")
    print()
