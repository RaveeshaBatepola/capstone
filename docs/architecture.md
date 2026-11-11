# System Architecture

```mermaid
flowchart TD
    subgraph OFFLINE["Offline: knowledge base construction"]
        A["FinQA reports: text + tables"] --> B["Parse and chunk: sentences and table rows as text"]
        B --> C[("BM25 keyword index")]
        B --> D["Embed chunks: all-MiniLM-L6-v2"]
        D --> E[("FAISS dense index")]
    end

    subgraph ONLINE["Online: Hybrid RAG pipeline"]
        Q["Question + report page"] --> R1["BM25 retrieval"]
        Q --> R2["Dense retrieval"]
        C -.-> R1
        E -.-> R2
        R1 --> F["Reciprocal Rank Fusion: top 10 chunks"]
        R2 --> F
        F --> G{"Evidence check: best dense similarity at least 0.45?"}
        G -- "No" --> H["Answer: insufficient evidence"]
        G -- "Yes" --> I["LLM gpt-oss-120b: answer only from the evidence"]
        I --> J["Answer + cited passages: chunk id, text, scores"]
        H --> J
    end
```

## Compared configurations

| Configuration | Retrieval | Evidence check | LLM |
|---|---|---|---|
| LLM only | none | none | same model |
| Dense RAG | dense, top 10 | none | same model |
| Hybrid RAG (proposed) | BM25 + dense with RRF, top 10 | threshold 0.45 | same model |