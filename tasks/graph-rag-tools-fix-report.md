# Graph RAG Tools 修復報告

## 問題背景

在測試 CodebaseRAG MCP 專案中最後三個註冊的 Graph RAG 工具時，發現了多個初始化和參數不匹配的問題。這些工具是：

1. **trace_function_chain_tool** - 追蹤函數鏈結流程
2. **find_function_path_tool** - 尋找函數間路徑
3. **analyze_project_chains_tool** - 分析專案鏈結

## 發現的問題

### 1. 服務初始化錯誤
**問題描述：**
所有 Graph RAG 工具都遇到相同的初始化錯誤：
```
Error: qdrant_service and embedding_service are required for first initialization
```

**根本原因：**
- `get_graph_rag_service()` 函數是單例模式，首次調用時需要 `qdrant_service` 和 `embedding_service` 參數
- 工具內部調用 `get_implementation_chain_service()` 時，該服務內部又調用 `get_graph_rag_service()` 但沒有提供所需參數
- 形成了服務依賴鏈中的初始化死循環

### 2. 參數不匹配錯誤
**問題描述：**
`analyze_project_chains_tool` 出現參數數量錯誤：
```
Error: analyze_project_chains() takes from 1 to 11 positional arguments but 12 were given
```

**根本原因：**
- `registry.py` 中 `analyze_project_chains_tool` 的函數調用參數與 `project_chain_analysis.py` 中實際實現的函數簽名不匹配
- Registry 傳遞了12個參數，但實際函數只接受11個參數

## 修復措施

### 1. 修復服務初始化問題

**修改檔案：**
- `src/tools/graph_rag/function_chain_analysis.py`
- `src/tools/graph_rag/function_path_finding.py`
- `src/tools/graph_rag/project_chain_analysis.py`

**修復內容：**
在每個工具中添加了正確的服務初始化邏輯：

```python
# Initialize services
from src.services.qdrant_service import QdrantService
from src.services.embedding_service import EmbeddingService
from src.services.graph_rag_service import get_graph_rag_service
from src.services.hybrid_search_service import get_hybrid_search_service

breadcrumb_resolver = BreadcrumbResolver()
qdrant_service = QdrantService()
embedding_service = EmbeddingService()

# Initialize Graph RAG and Hybrid Search services with required dependencies
graph_rag_service = get_graph_rag_service(qdrant_service, embedding_service)
hybrid_search_service = get_hybrid_search_service()

implementation_chain_service = get_implementation_chain_service(
    graph_rag_service=graph_rag_service,
    hybrid_search_service=hybrid_search_service
)
```

### 2. 修復參數不匹配問題

**修改檔案：**
- `src/tools/registry.py`

**修復內容：**
修正了 `analyze_project_chains_tool` 的函數調用，將參數從12個改為11個，並按照實際函數簽名順序排列：

```python
return await analyze_project_chains(
    project_name,
    analysis_types,
    "*",  # scope_pattern
    complexity_weights,
    None,  # chain_types
    complexity_threshold,
    max_functions_per_chain,
    include_refactoring_suggestions,
    output_format,
    True,  # performance_monitoring
    50,  # batch_size
)
```

## 修復位置詳細列表

### 檔案修改摘要

1. **`src/tools/graph_rag/function_chain_analysis.py`**
   - 行號：87-104
   - 修復：添加完整的服務初始化邏輯

2. **`src/tools/graph_rag/function_path_finding.py`**
   - 行號：171-188
   - 修復：添加完整的服務初始化邏輯

3. **`src/tools/graph_rag/project_chain_analysis.py`**
   - 行號：218-235
   - 修復：添加完整的服務初始化邏輯

4. **`src/tools/registry.py`**
   - 行號：1144-1156
   - 修復：修正函數調用參數

## 預期結果

修復完成後，應該能夠：

### 1. 成功執行 trace_function_chain_tool
```bash
# 測試指令
mcp__codebase-rag-mcp__trace_function_chain_tool(
    entry_point="search",
    project_name="function_chain_mcp_tools_wave",
    direction="forward",
    max_depth=3
)
```

**預期：**
- 不再出現 "qdrant_service and embedding_service are required" 錯誤
- 返回包含函數鏈結追蹤結果的 JSON 響應
- 包含 arrow 格式的輸出和執行統計

### 2. 成功執行 find_function_path_tool
```bash
# 測試指令
mcp__codebase-rag-mcp__find_function_path_tool(
    start_function="search",
    end_function="index_directory",
    project_name="function_chain_mcp_tools_wave"
)
```

**預期：**
- 不再出現初始化錯誤
- 返回兩個函數間的路徑分析結果
- 包含路徑品質指標和建議

### 3. 成功執行 analyze_project_chains_tool
```bash
# 測試指令
mcp__codebase-rag-mcp__analyze_project_chains_tool(
    project_name="function_chain_mcp_tools_wave",
    output_format="summary"
)
```

**預期：**
- 不再出現 "takes from 1 to 11 positional arguments but 12 were given" 錯誤
- 返回專案範圍的鏈結分析結果
- 包含複雜度分析和熱點識別

## 驗證步驟

下個 session 請依序執行以下測試：

1. **基礎健康檢查**
   ```bash
   mcp__codebase-rag-mcp__health_check_tool()
   ```

2. **測試 trace_function_chain_tool**
   ```bash
   mcp__codebase-rag-mcp__trace_function_chain_tool(
       entry_point="search",
       project_name="function_chain_mcp_tools_wave",
       direction="forward",
       max_depth=3,
       output_format="arrow"
   )
   ```

3. **測試 find_function_path_tool**
   ```bash
   mcp__codebase-rag-mcp__find_function_path_tool(
       start_function="search",
       end_function="index_directory",
       project_name="function_chain_mcp_tools_wave",
       strategy="optimal"
   )
   ```

4. **測試 analyze_project_chains_tool**
   ```bash
   mcp__codebase-rag-mcp__analyze_project_chains_tool(
       project_name="function_chain_mcp_tools_wave",
       output_format="summary",
       max_functions_per_chain=10
   )
   ```

## 注意事項

1. **服務依賴順序**：修復確保了正確的服務初始化順序，避免循環依賴

2. **性能考量**：每次工具調用都會創建新的服務實例，可能影響性能，但確保了穩定性

3. **參數映射**：registry.py 中的工具參數與實際實現之間需要保持一致

4. **錯誤處理**：修復主要解決初始化問題，運行時錯誤處理邏輯保持不變

## 後續優化建議

1. **服務單例改進**：考慮改進服務單例模式，避免重複初始化
2. **參數驗證**：添加工具參數與實現函數的自動化驗證
3. **錯誤信息**：改善錯誤信息，提供更具體的問題定位指導

---

**建立時間：** 2025-07-18
**修復人員：** Claude Code
**狀態：** 待驗證
