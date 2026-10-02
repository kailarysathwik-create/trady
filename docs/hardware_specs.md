# TRADY Hardware Specifications & Compute Guidelines

## 1. Target Machine Profile

* **System Model**: Lenovo LOQ
* **Processor (CPU)**: Intel Core i5-12450HX
  - 8 Cores (4 Performance Cores + 4 Efficient Cores)
  - 12 Execution Threads
  - Max Turbo Frequency: ~4.40 GHz
* **Graphics Processor (GPU)**: NVIDIA GeForce RTX 3050 Laptop GPU
  - Dedicated VRAM: 4 GB / 6 GB GDDR6
  - CUDA / Tensor Cores for parallel tensor operations
* **System Memory (RAM)**: 16 GB DDR5
* **Operating System**: Microsoft Windows 11 / Windows 10
* **Storage**: NVMe PCIe SSD

---

## 2. Distributed Development Workflow

TRADY is architected to support a multi-machine development lifecycle:

```
+------------------------------------+          +------------------------------------+
|        Main Coding Machine         |          |       Lenovo LOQ Workstation       |
|                                    |          |                                    |
| • Architecture & Code Design       |  GitHub  | • Heavy Data Processing            |
| • Unit Testing & Documentation     | <======> | • Model Training & Validation      |
| • Git Feature Branches & Reviews   |          | • Historical Backtesting           |
|                                    |          | • Real-time Paper Trading          |
+------------------------------------+          +------------------------------------+
```

### Git Synchronization Protocol
1. **Never commit data or large artifacts**: Datasets in `data/`, model weights in `models/`, and generated reports in `reports/` are excluded by `.gitignore`.
2. **Environment Reproducibility**: Both machines should run Python 3.11+ (Python 3.12 recommended) and manage environments via `uv` or `venv` using `pyproject.toml`.
3. **Configurations**: Machine-specific hardware overrides (e.g. `TRADY_MAX_WORKERS`, `TRADY_DEVICE`) are configured via `.env` or custom config files rather than modifying shared defaults.

---

## 3. Resource Management Guidelines for Lenovo LOQ

### CPU & Worker Allocation
- Intel i5-12450HX features 12 logical threads.
- **Guideline**: Default `max_workers` is set to `4` (leaving ample threads for OS, IDE, background disk I/O, and thermal headroom). Do not set `max_workers` > 8 to avoid thermal throttling.

### RAM (16 GB)
- Out of 16 GB, the Windows OS and background services utilize ~4–5 GB.
- **Guideline**: Target active in-memory datasets to $\le 6 \text{ GB}$.
- Use chunked reading or disk-backed formats (Parquet/Arrow/Memory-mapped files) for large tick or high-frequency series.

### GPU & VRAM (RTX 3050)
- The RTX 3050 laptop GPU has 4 GB or 6 GB VRAM.
- **Guideline**: Keep batch sizes conservative (`batch_size: 64` or `32` for sequence models like GRU/LSTM/Transformers).
- If training PyTorch models, ensure mixed precision (`torch.cuda.amp.autocast`) and call `torch.cuda.empty_cache()` between folds.
- If CUDA is not installed or configured, TRADY defaults smoothly to CPU execution.
