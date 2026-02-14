# Review: China A-Share Implementation Plan

**Document:** `docs/plans/2026-02-14-china-a-share-implementation.md`  
**Reviewer:** AI Code Review  
**Date:** 2026-02-14  
**Related Docs:**
- `docs/plans/2026-02-13-china-a-share-design-v2.md` (Design v2)
- `docs/reviews/2026-02-13-china-a-share-design-comparison.md` (Design comparison, chose v2.1)

---

## Overall Assessment

**Rating: 🟡 需要修改后方可执行 (Needs Revisions Before Execution)**

这是一份**结构清晰、分步合理**的实施计划，成功地将 v2.1 设计方案（`TradingContext` + `contextvars`）转化为具体的编码任务。整体思路正确，但存在若干实现细节问题、与设计文档的不一致之处、以及一些遗漏。以下逐一评审。

---

## ✅ 优点 (Strengths)

### 1. 任务拆分合理，粒度适中
10 个 Task 形成了清晰的依赖链：基础设施 → 数据模块 → 集成 → 测试 → 文档。每个 Task 都有明确的"文件-步骤-测试-提交"结构，适合逐步执行。

### 2. 与现有架构对齐
- `route_to_vendor` 的修改正确地在现有路由逻辑**之前**插入了 China stock 检测，确保零侵入。
- 函数签名与 `y_finance.py` 中的现有实现保持一致（已验证 `get_YFin_data_online`, `get_fundamentals`, `get_balance_sheet` 等签名）。
- AKShare 作为"一级路由"而非 fallback chain 的一员，设计正确——yfinance/alpha_vantage 确实不支持 A 股。

### 3. TradingContext 设计简洁有效
使用 `contextvars.ContextVar` 实现线程安全的请求级状态管理，是 Python 标准做法。在 `propagate()` 中 set/clear 的模式清晰，且用 `try/finally` 保证清理。

### 4. 测试策略分层合理
- 纯逻辑测试（ticker_utils, TradingContext）不需要网络
- 网络测试用 `@pytest.mark.network` 标记，可选跳过
- E2E 测试用 `@pytest.mark.skip` 标记，手动运行

---

## 🔴 严重问题 (Critical Issues)

### Issue 1: `TradingContext` 与 `ticker_utils` 功能重复且不一致

**问题描述：**  
文档同时创建了两个模块来判断 China stock：
- `TradingContext.is_china_stock()` — 只检查 `.SH` 和 `.SZ`
- `ticker_utils.is_china_stock()` — 同样只检查 `.SH` 和 `.SZ`

但 v2 设计文档中的 `ticker_utils.py` 明确包含了 `.BJ`（北交所）：

```python
# v2 设计文档中
CHINA_SUFFIXES = {".SH", ".SZ", ".BJ"}  # 上海、深圳、北交所
```

而实施计划中两个模块**都遗漏了 `.BJ`**。更重要的是，`TradingContext.is_china_stock()` 与 `ticker_utils.is_china_stock()` 是**功能完全重复**的。

**建议：**  
- `TradingContext.is_china_stock()` 应**委托给** `ticker_utils.is_china_stock()`，而非自行实现。
- 在 `ticker_utils.py` 中使用常量集合 `CHINA_SUFFIXES = {".SH", ".SZ", ".BJ"}`，所有判断统一走这个常量。

```python
# TradingContext 应该这样改：
from .ticker_utils import is_china_stock as _is_china

@staticmethod
def is_china_stock() -> bool:
    ticker = _current_ticker.get()
    if not ticker:
        return False
    return _is_china(ticker)
```

### Issue 2: `AKShareError` 在两个模块中重复定义

**问题描述：**  
`AKShareError` 异常类在 `akshare_data.py`（行 359-361）和 `akshare_news.py`（行 818-820）中**各定义了一次**。

**影响：**
- `except AKShareError` 只能捕获同一模块内的版本，跨模块捕获会失败。
- 违反 DRY 原则。

**建议：**  
在 `akshare_data.py` 中定义一次，在 `akshare_news.py` 中 import 使用：

```python
# akshare_news.py
from .akshare_data import AKShareError
```

或者更好的做法是单独创建 `akshare_common.py` 放通用异常和工具函数。

### Issue 3: `route_to_vendor` 中 China stock 路由逻辑缺少 `get_global_news` 的考虑

**问题描述：**  
Task 6 Step 4 的 `route_to_vendor` 修改（行 1149-1173）使用 `TradingContext.is_china_stock()` 来路由**所有**方法到 akshare。然而：

1. 当 `method = "get_global_news"` 且 `TradingContext.is_china_stock()` 为 True 时，代码会走到 `VENDOR_METHODS["get_global_news"]["akshare"]`，即 `get_akshare_global_news`。
2. `get_akshare_global_news` 实际上是 `akshare_news.get_global_news`（行 926-940），它**内部再次检查** `TradingContext.is_china_stock()` 来决定调用 `_get_china_macro_news`。

这意味着 `TradingContext.is_china_stock()` 的检查做了**两次**——一次在路由层，一次在实现内部。虽然功能上不会出错，但逻辑冗余，且两者的语义不一致：
- 路由层：检测到 China stock → 强制走 akshare vendor
- 实现层：再次检测 → 走 china macro vs global macro

**建议：**  
`akshare_news.get_global_news` 既然已经通过路由层确认是 akshare vendor，就不需要再检查 `TradingContext.is_china_stock()`。直接调用 `_get_china_macro_news` 即可，或者将 `get_global_news` 重命名为 `get_china_macro_news` 使语义更清晰。

### Issue 4: `_get_china_macro_news` 使用 `ak.news_cctv()` 只能获取单天新闻

**问题描述：**  
`ak.news_cctv(date="YYYYMMDD")` 接受的是**单个日期**参数，返回该天的 CCTV 新闻。但 `_get_china_macro_news` 传入的是 `start_dt`（行 885）：

```python
df = ak.news_cctv(date=start_dt.strftime("%Y%m%d"))
```

当 `look_back_days=7` 时，这只能获取 7 天前那**一天**的新闻，而非整个 7 天范围内的新闻。

**建议：**  
需要循环调用 `ak.news_cctv(date=...)` 遍历 `look_back_days` 天的每一天，或者使用其他 AKShare 宏观新闻 API（如 `ak.news_economic_baidu()`）。

```python
all_news = []
for i in range(look_back_days + 1):
    day = curr_dt - timedelta(days=i)
    try:
        df = ak.news_cctv(date=day.strftime("%Y%m%d"))
        if not df.empty:
            all_news.append(df)
    except:
        continue
df = pd.concat(all_news) if all_news else pd.DataFrame()
```

---

## 🟡 中等问题 (Medium Issues)

### Issue 5: `detect_market` 中 US 股票启发式判断过于简单

**问题描述（Task 3, 行 193-194）：**

```python
elif ticker.isalpha() or (len(ticker) <= 5 and ticker.isalnum()):
    return "US"
```

这个启发式会将以下非 US ticker 错误归类为"US"：
- `"BABA"` — 虽然在 NYSE 上市，但也可能让人误以为是通用判断
- `"BTC"` — 加密货币
- `"A1234"` — 任意 5 位字母数字

**建议：**  
这个函数目前在实施计划中**没有被任何模块使用**。如果只是为了提供辅助功能，建议：
1. 在文档注释中明确标注这是一个"尽力而为"的启发式函数
2. 或者移除这个函数，避免给人误导

### Issue 6: 财务报表函数（Task 4）的 `freq` 参数未被实际使用

**问题描述：**  
`get_balance_sheet`, `get_cashflow`, `get_income_statement` 都有 `freq` 参数（"annual" or "quarterly"），但实现中只是用 `df.head(4 if freq == "quarterly" else 2)` 来截取记录数。

AKShare 的 `stock_balance_sheet_by_report_em()` 等 API 返回的数据本身是**按报告期排列**的，其中既有年报也有季报混合。仅用 `head(N)` 截取不能真正区分年报和季报。

**建议：**  
应该根据"报告期"列（通常叫 `REPORT_DATE_NAME` 或 `报告期`）来过滤：
- 年报：只保留 12-31 结尾的报告期
- 季报：保留所有或最近 N 期

```python
if freq == "annual":
    df = df[df['报告期'].str.endswith("1231")]
df = df.head(4)
```

### Issue 7: `get_indicators` 中遍历日期的方式效率低下

**问题描述（Task 4, 行 560-575）：**

```python
current_dt = curr_date_dt
while current_dt >= before:
    date_str = current_dt.strftime('%Y-%m-%d')
    matching = stock_df[stock_df['date'].dt.strftime('%Y-%m-%d') == date_str]
    ...
    current_dt = current_dt - relativedelta(days=1)
```

这个实现逐天遍历（包括周末和假期），每次都对整个 DataFrame 做字符串匹配。非交易日会输出大量 "N/A: Not a trading day" 记录。

**建议：**  
参考 v2 设计文档中 `akshare_indicators.py` 的做法——直接按日期范围过滤 DataFrame，只遍历实际有数据的交易日：

```python
df_filtered = stock_df[stock_df['date'] >= before]
for _, row in df_filtered.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    value = row[indicator]
    ...
```

这既省去了非交易日的无效遍历，也避免了重复的 `strftime` 比较。

### Issue 8: `get_fundamentals` 返回信息过少

**问题描述（Task 4, 行 611-617）：**

目前只映射了 5 个字段（股票简称、公司名称、行业、总股本、流通股），遗漏了很多有价值的信息。`ak.stock_individual_info_em()` 返回的字段还包括：
- 上市日期
- 总市值
- 流通市值
- 市盈率（动态/静态）
- 市净率

**建议：**  
保留更多字段，或者干脆输出全部 `item: value` 对（如 v2 设计中的做法），让 LLM 自行判断哪些信息有用：

```python
for _, row in df.iterrows():
    lines.append(f"{row['item']}: {row['value']}")
```

### Issue 9: 测试用例中使用未来日期

**问题描述：**  
多处测试使用 `"2026-01-01"` 到 `"2026-01-15"` 的日期（行 758, 765 等）。这些日期作为测试来说是合理的（测试是写给当前日期的），但如果 AKShare 在这些日期没有数据（比如 2026 年是将来），测试可能会得到"No data found"而非真正的数据验证。

**建议：**  
集成测试建议使用**确定存在数据的历史日期**，例如 `"2024-12-01"` 到 `"2024-12-10"`，以确保测试的确定性。

---

## 🔵 小问题 (Minor Issues)

### Issue 10: Smoke test 创建后立刻删除（Task 9）

Task 9 Step 3 要求删除 `test_akshare_smoke.py`，但 git commit 信息是"test: verify AKShare smoke tests pass"却 add 了 `tests/test_china_a_share_integration.py`（这已经在 Task 8 中提交过了）。这个 commit 实际上是空的。

**建议：** Smoke test 如果要保留，放到 `tests/` 或 `scripts/` 目录下；如果不保留则 Task 9 的 commit 步骤应修改或移除。

### Issue 11: 缺少 `__init__.py` 导出

`tests/` 目录作为新创建的目录，需要确认是否需要 `__init__.py`。虽然 pytest 可以自动发现测试，但如果使用 `python -m pytest` 从根目录运行且 `tests/` 不在 Python path 中，可能需要配置。

### Issue 12: `akshare_news.py` 中 `get_insider_transactions` 放置位置不合理

Insider transactions（股东持股变动）在语义上更接近"基本面数据"而非"新闻"。将其放在 `akshare_news.py` 中与设计文档不一致（v2 设计将其放在 `akshare_data.py`）。

**建议：** 将 `get_insider_transactions` 移到 `akshare_data.py`。

### Issue 13: 文档中 Markdown 嵌套代码块未正确关闭

Task 10 的文档（行 1573）出现了一个多余的 ` ``` ` 闭合标记，会导致 Markdown 渲染异常：

```markdown
```python
# Midea Group (美的集团)
state, decision = ta.propagate("000333.SZ", "2025-01-15")
```           ← 这里正确关闭
```           ← 这个多余的闭合标记会破坏渲染
```

### Issue 14: `parse_ticker` 返回类型不一致

- `ticker_utils.parse_ticker("AAPL")` 返回 `("AAPL", None)` 
- 但 v2 设计中返回 `("AAPL", "")`（空字符串）

下游代码中有 `if not exchange:` 的检查（如 `akshare_data.py` 行 389），两种设计在布尔判断上行为一致（`None` 和 `""` 都是 falsy），但类型不一致可能导致后续维护混乱。

**建议：** 统一返回 `None` 或 `""`，并在文档中明确说明。

---

## 📋 与设计文档的一致性检查

| 设计要求 | 实施计划 | 一致性 |
|---------|---------|-------|
| TradingContext 使用 contextvars | ✅ Task 2 | ✅ |
| ticker_utils 独立模块 | ✅ Task 3 | ✅ |
| 独立 AKShare indicator 实现（不用 yfinance） | ✅ Task 4 内嵌在 akshare_data.py | ⚠️ v2 设计是独立文件 `akshare_indicators.py` |
| VENDOR_METHODS 注册 | ✅ Task 6 | ✅ |
| route_to_vendor ticker-aware 路由 | ✅ Task 6 Step 4 | ✅ |
| propagate 设置 TradingContext | ✅ Task 7 | ✅ |
| get_global_news 走 CCTV 新闻 | ✅ Task 5 | ⚠️ 实现有缺陷（单天 API） |
| 北交所 `.BJ` 支持 | ❌ 未包含 | ❌ 遗漏 |
| AKShareError 自定义异常 | ✅ 但重复定义 | ⚠️ |

**关于 `akshare_indicators.py` 独立文件：** v2 设计文档明确将技术指标放在独立的 `akshare_indicators.py` 文件中，但实施计划将其合并进了 `akshare_data.py`。考虑到 `akshare_data.py` 已经包含 OHLCV、基本面、财务报表等多种功能（行 343-718，约 375 行），再加入技术指标会使文件过于庞大。建议按设计文档拆分。

---

## 📝 修改建议优先级总结

| 优先级 | Issue | 修改量 |
|-------|-------|-------|
| 🔴 P0 | #1: TradingContext/ticker_utils 重复 + 缺 .BJ | 小 |
| 🔴 P0 | #2: AKShareError 重复定义 | 小 |
| 🔴 P0 | #4: CCTV 新闻单天 API 问题 | 中 |
| 🟡 P1 | #3: route_to_vendor 冗余检查 | 小 |
| 🟡 P1 | #6: freq 参数未真正使用 | 中 |
| 🟡 P1 | #7: 指标遍历效率 | 小 |
| 🟡 P1 | #8: fundamentals 信息不足 | 小 |
| 🟡 P1 | #12: insider_transactions 位置 | 小 |
| 🔵 P2 | #5, #9, #10, #11, #13, #14 | 小 |

---

## 结论

这份实施计划**方向正确、结构完整**，较好地落地了 v2.1 设计方案。**推荐先修复 P0 问题（尤其是 #1 和 #4）后再开始执行**。P1 问题可以在实施过程中逐步改进。具体来说：

1. **先修改文档**：修复 `TradingContext` 委托、`AKShareError` 统一、CCTV 新闻多天获取逻辑
2. **分离 indicators**：保持 `akshare_indicators.py` 独立文件，与 v2 设计一致
3. **然后按 Task 1-10 顺序执行**
