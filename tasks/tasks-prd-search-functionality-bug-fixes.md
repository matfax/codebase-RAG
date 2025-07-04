## Relevant Files

- `src/tools/indexing/search_tools.py` - Main search implementation containing the n_results multiplication bug and content filtering logic
- `src/tools/indexing/search_tools.test.py` - Unit tests for search functionality (to be created)
- `src/tools/core/error_utils.py` - Error handling utilities for search operations
- `src/tools/core/error_utils.test.py` - Unit tests for error handling utilities
- `src/services/qdrant_service.py` - Vector database service that may need content validation
- `src/services/qdrant_service.test.py` - Unit tests for Qdrant service
- `src/tests/test_search_bug_fixes.py` - Integration tests for search bug fixes (to be created)
- `src/utils/search_diagnostics.py` - Diagnostic utilities for search issues (to be created)
- `src/utils/search_diagnostics.test.py` - Unit tests for search diagnostics (to be created)

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `search_tools.py` and `search_tools.test.py` in the same directory).
- Use `npx jest [optional/path/to/test/file]` to run tests. Running without a path executes all tests found by the Jest configuration.

## Tasks

- [ ] 1.0 修復 n_results 參數倍數問題
  - [ ] 1.1 分析當前 `_perform_hybrid_search` 函數中的結果限制邏輯
  - [ ] 1.2 修改 `src/tools/indexing/search_tools.py:346` 行的結果限制邏輯
  - [ ] 1.3 實現正確的跨集合結果聚合和排序
  - [ ] 1.4 測試修復後的 n_results 行為在不同集合數量下的表現
  - [ ] 1.5 驗證修復不影響現有的搜索模式 (semantic, keyword, hybrid)

- [ ] 2.0 解決搜索結果空內容問題
  - [ ] 2.1 分析向量資料庫中空內容的根本原因
  - [ ] 2.2 在搜索結果中添加內容過濾機制
  - [ ] 2.3 創建內容完整性驗證工具
  - [ ] 2.4 實現有問題文件的重新索引機制
  - [ ] 2.5 添加空內容檢測的日志記錄

- [ ] 3.0 增強錯誤處理和診斷功能
  - [ ] 3.1 創建搜索診斷工具模組 `src/utils/search_diagnostics.py`
  - [ ] 3.2 增強搜索錯誤的日志記錄和報告
  - [ ] 3.3 實現向量資料庫一致性檢查功能
  - [ ] 3.4 添加搜索結果質量驗證機制
  - [ ] 3.5 創建搜索操作的詳細調試信息

- [ ] 4.0 建立完整的測試覆蓋
  - [ ] 4.1 為 `search_tools.py` 創建單元測試
  - [ ] 4.2 創建 n_results 參數行為的專門測試
  - [ ] 4.3 創建空內容檢測和處理的測試
  - [ ] 4.4 創建多集合搜索場景的集成測試
  - [ ] 4.5 創建邊界條件和錯誤情況的測試

- [ ] 5.0 驗證修復效果和效能影響
  - [ ] 5.1 執行所有新建的測試確保修復正確
  - [ ] 5.2 進行效能基準測試確保無顯著影響
  - [ ] 5.3 使用真實資料測試修復後的搜索功能
  - [ ] 5.4 驗證向後相容性確保現有功能不受影響
  - [ ] 5.5 記錄修復前後的搜索行為差異和改進效果
