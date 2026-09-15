from core.rag_engine import StandardsRAGEngine

engine = StandardsRAGEngine()

queries = [
    "solder ball causes insufficient solder paste excessive solder paste stencil printing PCB assembly",
    "poor solder joint causes insufficient solder wetting reflow temperature contamination PCB assembly",
]

for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = engine.search_standards(query, top_k=10)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('source')} | Page {result.get('page')} | RRF={result.get('rrf_score', 0):.6f}")
        print(result.get("text", "")[:700])