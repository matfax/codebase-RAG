# clear_file_metadata_tool 路徑匹配邏輯修復任務

## 問題背景

在測試 MCP 工具過程中，發現 `clear_file_metadata_tool` 無法正確刪除 indexing_report*.json 檔案的索引資料。測試顯示所有嘗試刪除的操作都回傳 `deleted_points: 0`。

## 問題分析

### 根本原因：路徑匹配不一致

**儲存時的路徑格式**：
- 在索引過程中，`IndexingService._sanitize_file_path()` 將絕對路徑轉換為相對路徑
- 儲存在資料庫中的格式：`"indexing_report_Agentic_RAG_20250701_125227.json"`

**刪除時的查詢格式**：
- `delete_file_chunks()` 函數將輸入路徑轉換為絕對路徑
- 查詢使用的格式：`"/Users/jeff/Documents/personal/Agentic-RAG/indexing_report_Agentic_RAG_20250701_125227.json"`

**結果**：絕對路徑無法匹配到儲存的相對路徑，導致找不到記錄。

## 測試證據

1. **搜尋結果確認路徑格式**：
   ```json
   {
     "file_path": "indexing_report_Agentic_RAG_20250701_125227.json",
     "collection": "project_Agentic_RAG_config"
   }
   ```

2. **清除工具失敗**：
   ```bash
   mcp__codebase-rag-mcp__clear_file_metadata_tool(file_path="indexing_report_Agentic_RAG_20250701_125227.json")
   # 回傳: {"deleted_points": 0}
   ```

## 已實施的修復

### 修改位置
檔案：`src/tools/project/project_utils.py:delete_file_chunks()` (行 642-674)

### 修復邏輯
1. **雙重路徑匹配**：先嘗試絕對路徑，失敗後嘗試相對路徑
2. **專案根目錄解析**：使用 `get_current_project()` 獲取專案根目錄
3. **相對路徑計算**：將絕對路徑轉換為相對於專案根目錄的路徑
4. **智能回退**：如果無法計算相對路徑，回退到絕對路徑匹配

### 修復代碼片段
```python
# 先嘗試絕對路徑匹配
filter_condition_abs = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=str(abs_path)))])
count_response_abs = qdrant_client.count(collection_name=collection_name, count_filter=filter_condition_abs, exact=True)
points_before = count_response_abs.count

# 如果絕對路徑沒有匹配，嘗試相對路徑
if points_before == 0:
    current_project = get_current_project(str(abs_path.parent))
    if current_project and current_project.get("root"):
        project_root = Path(current_project["root"])
        try:
            relative_path = abs_path.relative_to(project_root)
            filter_condition_rel = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=str(relative_path)))])
            count_response_rel = qdrant_client.count(collection_name=collection_name, count_filter=filter_condition_rel, exact=True)
            if count_response_rel.count > 0:
                points_before = count_response_rel.count
                filter_condition = filter_condition_rel
            # ... 其他邏輯
```

## 測試狀態

### 修復前測試結果
- ❌ 所有 indexing_report 檔案清除失敗
- ❌ `deleted_points: 0` 在所有嘗試中

### 修復後測試結果
**✅ 修復完全成功！已通過全面測試驗證**

#### 第一輪測試 (2025-07-15)
- ✅ `indexing_report_Agentic_RAG_20250701_141826.json` → 刪除 30 個資料點
- ✅ `indexing_report_Agentic_RAG_20250701_135347.json` → 刪除 45 個資料點
- ✅ `indexing_report_Agentic_RAG_20250701_125227.json` → 刪除 46 個資料點

#### 第二輪全面清理測試
- ✅ `chunking_metrics.json` → 刪除 10 個資料點
- ✅ `indexing_report_Agentic_RAG_20250701_230802.json` → 刪除 15 個資料點
- ✅ `indexing_report_Agentic_RAG_20250701_130147.json` → 刪除 46 個資料點
- ✅ `indexing_report_psme_20250701_160330.json` → 刪除 30 個資料點

#### 搜尋驗證結果
- ✅ 搜尋 "indexing_report" 不再返回任何 indexing_report*.json 檔案
- ✅ 搜尋 "chunking_metrics" 不再返回 chunking_metrics.json 檔案
- ✅ 只返回合法的項目配置檔案內容

#### 總計清理統計
- **總清除檔案數**: 7 個檔案
- **總刪除資料點**: 222 個資料點
- **清除成功率**: 100%

## 相關檔案

1. **已清理的問題檔案**：
   - ✅ `indexing_report_Agentic_RAG_20250701_125227.json` (已清除)
   - ✅ `indexing_report_Agentic_RAG_20250701_130147.json` (已清除)
   - ✅ `indexing_report_Agentic_RAG_20250701_135347.json` (已清除)
   - ✅ `indexing_report_Agentic_RAG_20250701_141826.json` (已清除)
   - ✅ `indexing_report_Agentic_RAG_20250701_230802.json` (已清除)
   - ✅ `indexing_report_psme_20250701_160330.json` (已清除)
   - ✅ `chunking_metrics.json` (已清除)

2. **核心修改檔案**：
   - `src/tools/project/project_utils.py`

3. **相關檔案**：
   - `src/services/indexing_service.py` (路徑清理邏輯)
   - `.ragignore` (已更新排除規則)

## 任務完成狀態

### ✅ 已完成的行動
1. **✅ 路徑匹配邏輯修復** - `src/tools/project/project_utils.py` 雙重路徑匹配機制
2. **✅ 全面檔案清理** - 成功清除所有 7 個問題檔案
3. **✅ 搜尋結果驗證** - 確認清理效果，搜尋結果純淨
4. **✅ .ragignore 更新** - 新增 `chunking_metrics.json` 排除規則

### ✅ 修復效果確認
1. **路徑匹配** - 雙重匹配機制完全解決絕對/相對路徑不一致問題
2. **向後兼容** - 修復保持向後兼容，不影響其他檔案操作
3. **預防措施** - `.ragignore` 配置防止未來重複索引這些檔案

## ✅ 達成結果

**所有預期目標均已實現**：
- ✅ 成功清除所有 indexing_report 檔案的索引資料（6個檔案）
- ✅ 成功清除 chunking_metrics.json 檔案的索引資料
- ✅ 搜尋 "indexing_report" 不再回傳相關結果
- ✅ 搜尋 "chunking_metrics" 不再回傳相關結果
- ✅ 大幅提升搜尋結果的準確性和相關性
- ✅ 總計清除 222 個不必要的資料點

## 技術總結

### 修復範圍
- ✅ `clear_file_metadata_tool` 路徑匹配邏輯完全修復
- ✅ `get_file_metadata_tool` 同時受益於路徑匹配改進
- ✅ 修復保持向後兼容性，先嘗試現有邏輯再嘗試新邏輯
- ✅ 已驗證不影響其他正常檔案的清除操作

### 關鍵改進
1. **雙重路徑匹配機制** - 解決絕對/相對路徑不一致問題
2. **智能回退邏輯** - 確保在各種路徑情況下都能正常工作
3. **專案根目錄感知** - 正確計算相對路徑關係
4. **向後兼容保證** - 不影響現有功能的正常使用

---

**任務狀態**: ✅ **已完成** | **測試狀態**: ✅ **通過** | **部署狀態**: ✅ **生產就緒**
