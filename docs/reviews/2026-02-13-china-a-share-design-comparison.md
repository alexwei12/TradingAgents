# Design Comparison: China A-Share Support

**Documents Evaluated**:
1. `docs/plans/2026-02-13-china-a-share-design-v2.md`
2. `docs/plans/2026-02-13-china-a-share-design-v2.1.md`

**Review Date**: 2026-02-13

---

## 🏆 Recommendation: Adopt Design v2.1

**Design v2.1 is the superior solution.** It addresses the critical routing limitations identified in the initial review with a robust architectural pattern (Thread-Local Context), whereas v2 applies a fragile "patch" that leaves some features broken.

### Key Differentiator: The "Invisible Ticker" Problem

The core challenge identified in the review was: **How do we route tools that don't take a ticker argument (specifically `get_global_news`) to the correct market source?**

| Feature | Design v2 (Args Inspection) | Design v2.1 (TradingContext) |
| :--- | :--- | :--- |
| **Routing Mechanism** | Inspects `*args` in `route_to_vendor` to find ticker. | Uses `contextvars` to store `current_ticker` globally for the request. |
| **`get_global_news`** | ❌ **Broken/Out of Scope**. Cannot route to China macro news because the tool has no ticker argument. Agents analyzing A-shares will still get US Fed news. | ✅ **Solved**. The context knows we are analyzing a China stock, so `get_global_news` can smartly route to `ak.news_cctv`. |
| **Extensibility** | Fragile. Relies on hardcoded index positions in `TICKER_TOOLS`. If tool signatures change, routing breaks. | Robust. Decouples tool signatures from routing logic. |
| **Invasiveness** | Low (local changes). | Medium (requires setting context in `propagate`), but cleaner long-term. |

### Other Improvements in v2.1

1.  **Explicit Date Conversion**: v2.1 provides concrete helper functions (`_to_akshare_date`) for the YYYYMMDD format issue, rather than just mentioning it.
2.  **Error Handling**: v2.1 introduces a specific `AKShareError` class, allowing for better error granularity than generic Exceptions.
3.  **Signature Alignment**: Both designs align signatures, but v2.1 explicitly maps them to the expected `interface.py` standards in a clearer way.

### Summary of Coverage

| Review Issue | v2 Solution | v2.1 Solution | Verdict |
| :--- | :--- | :--- | :--- |
| **1. Route by Ticker** | Partial (`*args` parsing) | **Complete** (`TradingContext`) | **v2.1 Wins** |
| **2. Function Sig** | Aligned | Aligned | Tie |
| **3. Stockstats** | Independent impl | Independent impl | Tie |
| **4. Registration** | Added | Added | Tie |
| **5. Fallback** | Documented | Documented + Custom Error | Tie |
| **6. Global News** | **Given Up** | **Solved** | **v2.1 Wins** |

## Implementation Roadmap (based on v2.1)

1.  **Foundation**: Create `tradingagents/dataflows/trading_context.py`.
2.  **Utils**: Create `ticker_utils.py` for market detection.
3.  **Graph Integration**: Update `tradingagents/graph/propagation.py` (or `trading_graph.py`) to set/clear `TradingContext` during execution.
4.  **Data Layer**: Implement `akshare_data.py`, `akshare_news.py`, `akshare_indicators.py`.
5.  **Routing**: Update `interface.py` to use `TradingContext` for routing logic.
6.  **Config**: Update `pyproject.toml` with `akshare` dependency.

## Conclusion

Proceed with **Design v2.1**. The Thread-Local Context pattern (`contextvars`) is the standard Pythonic way to handle request-scoped state (like "current user" or "current tenant") and fits perfectly for "current ticker" in this agent architecture. It solves the routing problem without forcing awkward signature changes on tools.
