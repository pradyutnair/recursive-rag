# Force-Hard No-Critic Architecture

This branch freezes the base method used for future experiments. It always runs the hard multi-agent DAG path and does not include a router, easy lane, critic, GEPA, GRPO, oracle labels, or W&B instrumentation.

```mermaid
flowchart TD
    Q["Input question"] --> P["Profile classifier"]
    P --> PL["DAG planner<br/>Qwen3-14B no-think"]
    PL --> DAG["Plan JSON<br/>nodes + dependencies + final node"]
    DAG --> LAYERS["Layered DAG executor<br/>parallel within each layer"]

    subgraph NODE["Per-node investigator loop"]
        R["Retrieve top-k chunks<br/>E5/wiki18 retriever on node408:8003"] --> X["Extract grounded answer span<br/>Qwen3-14B no-think"]
        X --> C{"confidence >= 0.65?"}
        C -- yes --> F["Store finding<br/>answer + chunk_id + confidence"]
        C -- no --> RW["Rewrite retrieval query"]
        RW --> R
        C -. "retry up to max_searches=5" .-> R
    end

    LAYERS --> NODE
    F --> TAG["Resolve child-node tags<br/><A1.1> parent answers"]
    TAG --> LAYERS
    F --> TRACE["DAG trace + retrieved chunk store"]
    TRACE --> S["Synthesizer<br/>one final answer span + support ids"]
    S --> G{"Citation gate accepts support?"}
    G -- yes --> A["Generic answer cleanup"]
    G -- no --> B["Fallback to best grounded finding"]
    B --> A
    A --> OUT["Final answer"]

    subgraph REMOVED["Explicitly absent in this base"]
        N1["No router"]
        N2["No easy lane"]
        N3["No critic"]
        N4["No GEPA/GRPO"]
        N5["No oracle signal at runtime"]
    end
```
