# 全面路徑匹配邏輯審計報告

## 摘要

針對所有 MCP 工具進行路徑匹配邏輯審計，發現 2 個需要修復的函數和 4 個已經正確實現的工具。

## 審計結果

### ✅ 已正確實現（無需修復）

#### 1. **delete_file_chunks()** - `src/tools/project/project_utils.py:599`
- **狀態**: ✅ 已修復（雙重路徑匹配邏輯）
- **實現**: 先嘗試絕對路徑，失敗後嘗試相對路徑
- **影響工具**:
  - `clear_file_metadata_tool`
  - `reindex_file_tool`

#### 2. **clear_file_metadata()** - `src/tools/project/file_tools.py:112`
- **狀態**: ✅ 間接修復
- **實現**: 內部調用 `delete_file_chunks()`
- **受益**: 自動獲得雙重路徑匹配邏輯

#### 3. **reindex_file()** - `src/tools/project/file_tools.py:136`
- **狀態**: ✅ 間接修復
- **實現**: 內部調用 `delete_file_chunks()`
- **受益**: 自動獲得雙重路徑匹配邏輯

#### 4. **快取管理工具**
- **manual_invalidate_file_cache()** - `src/tools/cache/cache_management.py:25`
- **invalidate_chunks()** - `src/tools/cache/cache_management.py:1754`
- **狀態**: ✅ 無需修復
- **原因**: 不直接查詢資料庫，調用快取失效服務

### ✅ 已修復的函數

#### 1. **get_file_metadata()** - `src/tools/project/file_tools.py:21`

**狀態**: ✅ 已實施雙重路徑匹配修復 (2025-07-15)

**修復位置**: 第 51-65 行 (路徑準備) + 第 72-90 行 (查詢邏輯)

**已實施修復**:
```python
# 雙重路徑過濾器準備
filter_condition_abs = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=str(abs_path)))])

filter_condition_rel = None
try:
    from tools.project.project_utils import get_current_project
    current_project = get_current_project(str(abs_path.parent))
    if current_project and current_project.get("root"):
        project_root = Path(current_project["root"])
        relative_path = abs_path.relative_to(project_root)
        filter_condition_rel = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=str(relative_path)))])
except (ValueError, Exception):
    pass

# 雙重路徑查詢邏輯
count_response = client.count(collection_name=collection_name, count_filter=filter_condition_abs, exact=True)
chunk_count = count_response.count
filter_to_use = filter_condition_abs

# 如果絕對路徑無結果，嘗試相對路徑
if chunk_count == 0 and filter_condition_rel is not None:
    count_response = client.count(collection_name=collection_name, count_filter=filter_condition_rel, exact=True)
    chunk_count = count_response.count
    filter_to_use = filter_condition_rel
```

**影響工具**: `get_file_metadata_tool`

**修復前測試證據**:
```bash
mcp__codebase-rag-mcp__get_file_metadata_tool(file_path="src/main.py")
# 回傳: {"indexed": false, "total_chunks": 0, "message": "File is not indexed"}
```

**修復狀態**: ⏳ 需要 MCP 重啟後測試驗證

### ❌ 仍需修復的函數

#### 2. **_get_context_chunks()** - `src/tools/indexing/search_tools.py:169`

**狀態**: ❌ 待修復

**問題位置**: 第 212 行
```python
file_filter = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))])
```

**問題**: 直接使用搜尋結果中的 `file_path`，沒有路徑規範化

**影響功能**: 搜尋結果的上下文擴展

**潛在後果**: 上下文資訊缺失，搜尋結果不完整

**測試證據**: 搜尋 "FastMCP" 時只返回配置檔案，未找到 Python 代碼集合中的內容

## 修復進度更新 (2025-07-15)

### ✅ 已完成修復

#### 修復 1: get_file_metadata() 函數

**檔案**: `src/tools/project/file_tools.py`
**修復狀態**: ✅ 完成
**修復時間**: 2025-07-15

**實施的修復策略**:
1. 雙重路徑過濾器準備 (第 51-65 行)
2. 智能路徑回退查詢邏輯 (第 72-90 行)
3. 使用與 `delete_file_chunks()` 相同的路徑匹配邏輯

**修復詳情**:
- 準備絕對路徑和相對路徑兩個過濾器
- 先嘗試絕對路徑查詢
- 如果沒有結果，自動回退到相對路徑查詢
- 保持向後兼容性

### ✅ 已修復的函數 (2025-07-15 第二次修復)

#### 修復 2: _get_context_chunks() 函數

**檔案**: `src/tools/indexing/search_tools.py`
**修復狀態**: ✅ 已完成 (2025-07-15)
**修復位置**: 第 210-248 行 (_expand_search_context 函數內)

**實施的修復策略**:
1. ✅ 路徑規範化邏輯 - 將相對路徑轉換為絕對路徑
2. ✅ 雙重路徑匹配機制 - 先試絕對路徑，失敗後試相對路徑
3. ✅ 錯誤處理 - 確保相對路徑計算失敗時的穩健性

**修復詳情**:
```python
# 原始問題代碼 (第 212 行)
file_filter = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))])

# 修復後代碼 (第 210-248 行)
from pathlib import Path

# 標準化路徑處理
abs_path = Path(file_path).resolve() if Path(file_path).is_absolute() else Path.cwd() / file_path

# 雙重過濾器準備
file_filter_abs = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=str(abs_path)))])
file_filter_rel = None

# 相對路徑回退邏輯
try:
    from tools.project.project_utils import get_current_project
    current_project = get_current_project(str(abs_path.parent))
    if current_project and current_project.get("root"):
        project_root = Path(current_project["root"])
        relative_path = abs_path.relative_to(project_root)
        file_filter_rel = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=str(relative_path)))])
except (ValueError, Exception):
    pass

# 智能路徑匹配查詢
context_results = qdrant_client.search(..., query_filter=file_filter_abs, ...)

# 如果無結果且有相對路徑過濾器，嘗試相對路徑
if not context_results and file_filter_rel is not None:
    context_results = qdrant_client.search(..., query_filter=file_filter_rel, ...)
```

**影響功能**: 搜尋結果的上下文擴展 (include_context=true 時)
**修復複雜度**: 中等 (與 get_file_metadata() 類似的雙重路徑匹配邏輯)

## 當前狀態總結 (更新至 2025-07-15 第二次修復)

### ✅ 已解決問題 (3/3)
1. **delete_file_chunks()** - ✅ 已在前次修復中解決
2. **get_file_metadata()** - ✅ 第一次修復已完成
3. **_get_context_chunks()** - ✅ 第二次修復已完成

### ❌ 待解決問題 (0/3)
**所有路徑匹配問題已修復完成！**

### 測試發現 (2025-07-15)
- **Python 代碼集合存在**: `project_Agentic_RAG_code` (56,218 個資料點)
- **搜尋問題**: 搜尋 "FastMCP" 只返回配置檔案，未查詢代碼集合
- **可能原因**: 需要 MCP 重啟以載入修復後的代碼，或搜尋工具集合選擇邏輯問題

## 下次 Session 測試計劃 (所有修復完成，需驗證)

### ⏳ 需要 MCP 重啟後測試

#### 測試案例 1: get_file_metadata_tool 修復驗證
```bash
# 修復前 (已確認失敗)
mcp__codebase-rag-mcp__get_file_metadata_tool(file_path="src/main.py")
# 實際結果: {"indexed": false, "total_chunks": 0}

# 修復後 (MCP 重啟後測試)
mcp__codebase-rag-mcp__get_file_metadata_tool(file_path="src/main.py")
# 預期結果: {"indexed": true, "total_chunks": N, "collections": {"project_Agentic_RAG_code": {...}}}
```

#### 測試案例 2: 上下文搜尋功能驗證 (新修復)
```bash
# 測試 _get_context_chunks() 修復效果
mcp__codebase-rag-mcp__search(query="FastMCP", include_context=true, context_chunks=2)
# 預期: 如果找到結果，應該包含 context_before 和 context_after

# 確認搜尋是否能找到 Python 代碼
mcp__codebase-rag-mcp__search(query="FastMCP register_tools")
# 預期: 應該返回 src/main.py 中的相關代碼，而非只有配置檔案
```

#### 測試案例 3: 相對路徑匹配驗證
```bash
# 測試相對路徑是否能正確匹配
mcp__codebase-rag-mcp__get_file_metadata_tool(file_path="src/main.py")
mcp__codebase-rag-mcp__get_file_metadata_tool(file_path="/Users/jeff/Documents/personal/Agentic-RAG/src/main.py")
# 預期: 兩者都應該返回相同的結果
```

## 代碼品質改善建議

### 建議 1: 統一路徑處理工具函數
創建共用的路徑匹配工具函數，避免重複實現：

```python
# src/utils/path_utils.py
def create_robust_file_filter(file_path: str | Path) -> list[Filter]:
    \"\"\"Create file filters for both absolute and relative path matching.\"\"\"
    # 統一實現邏輯
    pass
```

### 建議 2: 路徑儲存標準化
考慮在索引階段統一路徑格式，或在查詢階段標準化路徑處理。

## 風險評估

### 低風險修復
- **get_file_metadata()**: 純查詢操作，不會影響資料完整性
- **_get_context_chunks()**: 僅影響搜尋結果豐富度

### 迴歸測試需求
- 確保修復不影響正常的絕對路徑查詢
- 驗證相對路徑和絕對路徑混合場景
- 測試不存在檔案的錯誤處理

## 下次 Session 行動清單

### 立即行動
1. **✅ 已完成: get_file_metadata() 修復**
2. **🔄 MCP 重啟**: 載入修復後的代碼
3. **🧪 執行測試計劃**: 驗證 get_file_metadata_tool 修復效果
4. **🔍 調查搜尋問題**: 確認為何搜尋只返回配置檔案集合

### 後續行動
5. **🔧 實施 _get_context_chunks() 修復**
6. **🛠️ 建立統一的路徑處理工具函數**
7. **📋 完整測試驗證**: 包含上下文搜尋功能

## 總結 (更新至 2025-07-15 第二次修復完成)

### 進度狀態 ✅ 全部完成
- **發現問題**: 2 個函數需要修復路徑匹配邏輯
- **✅ 已修復**: 2 個函數 (`get_file_metadata()`, `_get_context_chunks()`)
- **✅ 已解決**: 3 個工具通過間接修復已經正確 (早期修復)

### 修復詳情總結
1. **delete_file_chunks()** - ✅ 早期已修復 (雙重路徑匹配邏輯)
2. **get_file_metadata()** - ✅ 第一次修復 (2025-07-15)
3. **_get_context_chunks()** - ✅ 第二次修復 (2025-07-15)

### 影響範圍 ✅ 全面覆蓋
- **檔案元資料查詢**: ✅ 已修復，需 MCP 重啟驗證
- **搜尋上下文擴展**: ✅ 已修復，需 MCP 重啟驗證
- **檔案清理和重新索引**: ✅ 早期已修復且正常運作

### 修復複雜度評估
- **get_file_metadata()**: ✅ 已完成（中等複雜度）
- **_get_context_chunks()**: ✅ 已完成（中等複雜度）
- **搜尋集合問題**: ❓ 可能需要 MCP 重啟解決

### 下次 Session 重點
1. **🔄 MCP 重啟**: 載入所有修復後的代碼
2. **🧪 全面驗證**: 測試所有三個修復案例
3. **🔍 搜尋調查**: 如果重啟後仍有問題，調查搜尋集合選擇邏輯

### 預期結果
- 所有路徑匹配問題應該解決
- 相對路徑和絕對路徑都能正確查詢
- 搜尋功能應該能查詢到 Python 代碼集合
- 上下文擴展功能應該正常運作

---

**最後更新**: 2025-07-15 | **整體進度**: 100% (3/3 已解決) | **狀態**: ✅ 修復完成，需 MCP 重啟驗證
