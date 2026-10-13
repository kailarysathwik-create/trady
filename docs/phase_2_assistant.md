# TRADY Phase 2: Local Quantitative AI Research Assistant

## 1. Overview & Mission

TRADY Phase 2 introduces a **strictly local conversational quantitative research assistant** powered by Ollama. It enables quantitative researchers to interactively inspect datasets, query causal feature metadata, analyze serialized ML model metrics, compare experiment runs, and explain backtest performance without leaving their local research workflow.

### Core Guarantees:
1. **Zero Cloud API Keys**: Fully private, offline-capable local inference via Ollama.
2. **Grounded Quantitative Facts**: The LLM is strictly decoupled from numerical computation. All statistics, returns, and metadata are queried directly from real Parquet files, DuckDB tables, and model metadata via allowlisted Python tools.
3. **Safety & Firewalls Preserved**: The AI assistant has **no shell execution privileges**, **no file write/modification access**, and **no authority to route live orders or bypass risk limits**.

---

## 2. Architecture & Service Layer

```
trady/
├── src/trady/
│   ├── assistant/
│   │   ├── __init__.py          # Package exports (OllamaClient, AssistantService, AssistantResponse)
│   │   ├── client.py            # Local HTTP client communicating with Ollama REST API
│   │   ├── tools.py             # 6 allowlisted research tools and tool schema definitions
│   │   ├── service.py           # Coordinator enforcing grounding, system prompt, and tool loop
│   │   └── cli.py               # Assistant CLI subcommands (`trade9 assistant ...`)
│   ├── config/settings.py       # AssistantConfig (model_name, endpoint, timeout, temperature)
│   └── studio/
│       ├── server.py            # Assistant endpoints (/api/assistant/status, /api/assistant/chat)
│       └── static/index.html    # Claymorphic Studio AI Chat Interface
└── tests/
    └── test_assistant.py        # Comprehensive unit & integration tests with mocked Ollama
```

---

## 3. Allowlisted Research Tools

The assistant can only interact with the TRADY workspace via 6 strictly allowlisted, read-only Python functions:

| Tool Name | Scope | Purpose |
| :--- | :--- | :--- |
| `inspect_datasets` | `data/processed`, `data/features`, `data/targets` | Reads Parquet schema, row counts, and column names. |
| `get_feature_metadata` | `FeatureRegistry` | Retrieves lookback windows, formulas, and causal point-in-time flags for registered features. |
| `get_experiment_results` | `models/*.joblib` | Reads trained model metadata, hyperparams, and out-of-sample test metrics. |
| `compare_experiments` | `models/*.joblib` | Computes side-by-side metric differences ($\Delta = A - B$) between two experiments. |
| `get_backtest_metrics` | `reports/*.html`, `reports/*_trades.parquet` | Audits backtest reports, simulated trade counts, and performance logs. |
| `explain_research_report` | Multi-source aggregation | Collects grounded evidence across models, backtests, and datasets for structured report explanation. |

---

## 4. Configuration

The AI Assistant is configured via `trady.config.settings.AssistantConfig` or environment variables:

```yaml
assistant:
  ollama_endpoint: "http://127.0.0.1:11434"
  model_name: "qwen3:4b"
  timeout_seconds: 120.0
  temperature: 0.2
```

### Supported Environment Variables:
- `TRADY_ASSISTANT_OLLAMA_ENDPOINT`: Custom Ollama URL (default: `http://127.0.0.1:11434`).
- `TRADY_ASSISTANT_MODEL`: Model tag to use (default: `qwen3:4b`).
- `TRADY_ASSISTANT_TIMEOUT_SECONDS`: Request timeout in seconds (default: `120.0`).
- `TRADY_ASSISTANT_TEMPERATURE`: Sampling temperature (default: `0.2` for deterministic, grounded outputs).

---

## 5. Usage & Verification

### CLI Commands:
```powershell
# 1. Check local Ollama daemon status & installed models
trade9 assistant status

# 2. Ask a one-off quantitative research question
trade9 assistant ask "What datasets are currently available in the workspace?"

# 3. Launch an interactive multi-turn research session
trade9 assistant chat
```

### Web Studio Interface:
Launch the Claymorphic TRADY Web Studio:
```powershell
trade9 studio
# or: uv run python studio.py
```
Open `http://127.0.0.1:8501/` in your browser. The AI Assistant panel is located on the right side of the interface, providing:
- Real-time model connection badge (e.g. `qwen3:4b (Ready)` or `Offline`).
- Multi-turn conversational chat with automatic message scrolling.
- Real-time tool execution status badges (e.g., `🔧 inspect_datasets (success)`).
- Quick suggestion chips for common research inquiries.

---

## 6. Test Results & Quality Audit

The Phase 2 AI Assistant was verified with comprehensive automated tests:

- **Total Test Suite**: 104 passed in 20.97s (`pytest`).
- **Code Style & Linting**: 100% compliant with PEP 8 and Ruff (`line-length = 88`).
- **Mocked Ollama Tests**: Verified offline handling, missing model recovery, tool execution loop, and unauthorized tool blocking.

---

## 7. Known Limitations & Operating Advice

1. **Local Compute & Cold Start**: Initial inference on CPU may take 30-60 seconds to load weights into memory. GPU acceleration (e.g. NVIDIA RTX 3050 with CUDA) significantly accelerates response times.
2. **Model Availability**: If the target model (`qwen3:4b`) has not been downloaded, run `ollama pull qwen3:4b` in your terminal.
3. **No Autonomous Execution**: The assistant is strictly consultative and cannot autonomously execute trades, modify code, or run live market orders.
