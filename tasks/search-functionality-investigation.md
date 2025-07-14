# 搜尋功能問題調查報告

## 問題摘要
雖然項目索引成功完成並存儲了30,501個數據點，但搜尋功能完全無法返回任何結果。

## 調查時間
**日期**: 2025-07-14
**調查者**: Claude Code (AI Assistant)
**相關會話**: MCP工具測試與修復會話

## 問題現狀

### ✅ 正常運作的組件
1. **索引基礎架構**: 完全正常
   - 文件發現和過濾: ✅
   - 變更檢測: ✅ (74個文件被識別為已修改)
   - 數據存儲: ✅ (30,501個點成功存儲)
   - 元數據追蹤: ✅ (945個文件元數據條目)

2. **數據庫連接**: 完全正常
   - Qdrant連接: ✅ (健康檢查通過)
   - 集合創建: ✅ (4個集合正常)
   - 數據寫入: ✅ (批量插入成功)

3. **MCP工具基礎架構**: 85-90%成功率
   - 項目管理工具: ✅ 100%
   - 健康檢查工具: ✅ 100%
   - 索引狀態工具: ✅ 100%

### ❌ 有問題的組件

#### 1. 搜尋結果檢索 - 完全失效
**症狀**:
```json
{
  "results": [],
  "total": 0,
  "query": "任何查詢內容",
  "collections_searched": ["正確的集合名稱"],
  "suggestions": ["建議重新索引..."]
}
```

**測試過的搜尋模式**:
- ✗ `hybrid` 模式: 無結果
- ✗ `keyword` 模式: 無結果
- ✗ `semantic` 模式: 無結果
- ✗ 跨項目搜尋: 無結果 (73個集合)
- ✗ 簡單關鍵詞 ("function", "MCP", "search"): 無結果

#### 2. 智能解析 - 部分失效
**症狀**:
```
Failed to parse [文件路徑] intelligently, using fallback: 'coroutine' object has no attribute 'chunks'
```

**影響範圍**: 所有74個修改的文件均使用回退解析方式

## 技術分析

### 數據流分析

#### ✅ 索引流程 (正常)
```
文件發現 → 變更檢測 → 內容讀取 → 回退解析 → 數據存儲 → 元數據更新
```

#### ❌ 搜尋流程 (中斷)
```
查詢輸入 → 查詢嵌入 → 向量搜尋 → [中斷點] → 結果處理 → 回應格式化
```

### 根本原因推測

#### 高可能性原因

1. **嵌入服務問題**
   - **位置**: `src/services/embedding_service.py`
   - **症狀**: `RuntimeWarning: coroutine 'EmbeddingService.generate_embeddings' was never awaited`
   - **推測**: 異步調用未正確處理，導致嵌入向量未生成或無效
   - **影響**: 查詢嵌入失敗，無法進行向量相似性搜尋

2. **代碼解析器異步處理錯誤**
   - **位置**: `src/services/code_parser_service.py`
   - **症狀**: `'coroutine' object has no attribute 'chunks'`
   - **推測**: 異步函數返回協程對象而非實際結果
   - **影響**: 智能解析失敗，內容品質降低

3. **搜尋管道異步協調問題**
   - **位置**: `src/tools/indexing/search_tools.py`, `src/services/search_*`
   - **推測**: 搜尋請求處理中的異步調用鏈中斷
   - **影響**: 完全無法檢索結果

#### 中等可能性原因

4. **向量數據庫查詢問題**
   - Qdrant查詢參數錯誤
   - 集合結構不匹配
   - 向量維度不一致

5. **快取服務干擾**
   - 搜尋快取返回空結果
   - 快取失效機制問題

#### 低可能性原因

6. **權限或配置問題**
   - 數據存儲和檢索權限不同
   - 環境變量配置問題

## 詳細技術證據

### 索引成功證據
```bash
# 索引前
"total_points": 30185

# 索引後
"total_points": 30501
"files processed": 74
"chunks generated": 74
"success_rate": 100.0%
```

### 搜尋失敗證據
```bash
# 多次搜尋測試
搜尋 "cache invalidation" → 0 結果
搜尋 "function" → 0 結果
搜尋 "MCP" → 0 結果
跨項目搜尋 "search" → 0 結果 (73個集合)
```

### 異步處理錯誤證據
```bash
RuntimeWarning: coroutine 'EmbeddingService.generate_embeddings' was never awaited
RuntimeWarning: coroutine 'CodeParserService.parse_file' was never awaited
```

## 建議的修復策略

### 階段1: 異步調用修復 (高優先級)
1. **修復嵌入服務異步調用**
   - 檢查 `EmbeddingService.generate_embeddings` 的調用點
   - 確保所有調用都使用 `await` 或適當的異步處理
   - 文件: `src/services/embedding_service.py`, `src/services/indexing_pipeline.py`

2. **修復代碼解析器異步調用**
   - 檢查 `CodeParserService.parse_file` 的調用點
   - 修復協程對象的屬性訪問錯誤
   - 文件: `src/services/code_parser_service.py`, `src/services/indexing_service.py`

### 階段2: 搜尋管道調試 (高優先級)
3. **添加搜尋流程日誌**
   - 在搜尋管道的每個階段添加詳細日誌
   - 追蹤查詢嵌入生成過程
   - 驗證向量搜尋請求和回應

4. **測試嵌入生成**
   - 獨立測試查詢文本的嵌入生成
   - 驗證嵌入向量的有效性和維度

### 階段3: 數據驗證 (中優先級)
5. **驗證存儲的向量數據**
   - 檢查Qdrant中實際存儲的向量數據
   - 驗證向量維度和數據類型
   - 測試直接的向量相似性查詢

6. **測試搜尋組件隔離**
   - 單獨測試各個搜尋組件
   - 驗證快取服務不干擾搜尋

### 階段4: 全面測試 (低優先級)
7. **重建搜尋索引**
   - 清除現有數據並重新索引
   - 監控整個流程以識別問題點

8. **性能和配置優化**
   - 檢查配置參數
   - 優化異步處理性能

## 立即可執行的診斷步驟

### 快速診斷命令
```bash
# 1. 檢查嵌入服務健康狀態
# 使用相關的MCP健康檢查工具

# 2. 驗證Qdrant數據
# 直接查詢Qdrant以確認數據存在

# 3. 測試簡單搜尋
# 使用最基本的關鍵詞搜尋

# 4. 檢查日誌
# 查看搜尋過程中的詳細錯誤日誌
```

### 代碼級別檢查
1. **檢查異步調用模式**:
   ```python
   # 錯誤模式
   result = service.async_method()

   # 正確模式
   result = await service.async_method()
   ```

2. **檢查協程處理**:
   ```python
   # 錯誤模式
   chunks = parse_result.chunks  # parse_result是協程

   # 正確模式
   parse_result = await parse_method()
   chunks = parse_result.chunks
   ```

## 影響評估

### 功能影響
- **搜尋功能**: 完全不可用
- **索引功能**: 基本可用 (使用回退解析)
- **數據管理**: 完全正常
- **系統監控**: 完全正常

### 用戶體驗影響
- **嚴重性**: 高 - 核心搜尋功能完全失效
- **緊急性**: 高 - 影響主要使用場景
- **範圍**: 影響所有搜尋相關操作

## 後續追蹤

### 需要監控的指標
1. 搜尋成功率 (目前: 0%)
2. 嵌入生成成功率
3. 異步處理錯誤數量
4. 向量數據完整性

### 測試用例
1. 基本關鍵詞搜尋
2. 語義搜尋測試
3. 跨項目搜尋
4. 嵌入生成獨立測試

---

**文件創建**: 2025-07-14
**狀態**: 待修復
**優先級**: 高
**預估修復時間**: 1-2個開發會話
**相關文件**:
- `tasks/session-mcp-tools-testing-fix.md` (主要修復追蹤)
- `ai_docs/mcp-tools-testing-plan.md` (測試計劃)
