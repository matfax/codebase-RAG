## Relevant Files

- `src/services/code_parser_service.py` - 新建的 CodeParser 服務，負責 Tree-sitter 整合和智慧分塊核心邏輯
- `src/services/indexing_service.py` - 需要修改 `_process_single_file` 方法整合 CodeParser
- `src/mcp_tools.py` - 更新 `index_directory` 工具，新增 `project_name` 參數
- `src/models/code_chunk.py` - 新建 CodeChunk 資料模型定義
- `pyproject.toml` - 新增 Tree-sitter 相關依賴包
- `src/utils/ast_utils.py` - 新建 AST 操作和語法分析工具函數
- `src/utils/language_detector.py` - 增強的語言檢測和解析器管理
- `tests/test_code_parser_service.py` - CodeParser 服務的單元測試
- `tests/test_intelligent_chunking.py` - 智慧分塊功能的整合測試
- `tests/fixtures/sample_code/` - 測試用的範例程式碼檔案（各種語言）
- `manual_indexing.py` - 更新以支援新的分塊機制和錯誤報告
- `README.md` - 更新文件說明新的智慧分塊功能
- `CLAUDE.md` - 更新架構說明和開發指令

### Notes

- 此實作將完全替換現有的整檔案 embedding 策略，不提供向後相容
- 需要預先安裝所有 Tree-sitter 語言解析器依賴
- 大型專案建議使用 `manual_indexing.py` 進行離線索引
- 錯誤處理採用降級策略：智慧分塊失敗時自動回退到整檔案處理
- 使用 `.venv/bin/pytest tests/` 執行所有測試

## Tasks

- [x] 1.0 建立 CodeParser 服務基礎架構
  - [x] 1.1 在 pyproject.toml 中新增 Tree-sitter 相關依賴包
  - [x] 1.2 建立 CodeChunk 資料模型和 ChunkType 枚舉定義
  - [x] 1.3 建立 CodeParserService 服務框架和基礎介面
  - [x] 1.4 實作 Tree-sitter 解析器載入和快取機制
  - [x] 1.5 建立語言檢測和解析器管理系統
  - [x] 1.6 實作基礎的 AST 遍歷和節點識別功能

- [x] 2.0 實作核心智慧分塊邏輯
  - [x] 2.1 實作 Python 語言的函數和類別分塊邏輯
  - [x] 2.2 實作 JavaScript 語言的函數和物件分塊邏輯
  - [x] 2.3 實作 TypeScript 語言的介面、型別和類別分塊邏輯
  - [x] 2.4 實作頂層常數和變數的智慧分塊判斷
  - [x] 2.5 建立通用的元資料提取和豐富化機制
  - [x] 2.6 實作文檔字串 (docstring) 提取和關聯功能

- [x] 3.0 整合現有系統並更新 MCP 工具
  - [x] 3.1 修改 IndexingService._process_single_file 整合 CodeParser
  - [x] 3.2 實作智慧分塊到現有 Chunk 格式的轉換機制
  - [x] 3.3 更新 index_directory MCP 工具，新增 project_name 參數
  - [x] 3.4 實作檔案內上下文增強（前後 5 行程式碼）
  - [x] 3.5 更新搜尋結果顯示，包含麵包屑導覽資訊
  - [x] 3.6 確保現有 MCP 工具的向下相容性

- [x] 4.0 實作錯誤處理和容錯機制
  - [x] 4.1 實作 Tree-sitter ERROR 節點的智慧處理邏輯
  - [x] 4.2 建立語法錯誤統計和分類系統
  - [x] 4.3 實作智慧分塊失敗時的降級策略（回退到整檔案）
  - [x] 4.4 在 manual_indexing.py 中新增錯誤報告功能
  - [x] 4.5 實作錯誤附近正確程式碼的保留機制
  - [x] 4.6 建立全面的錯誤記錄和除錯資訊系統

- [x] 5.0 語言支援擴展和進階功能
  - [x] 5.1 擴展支援 Go 語言的智慧分塊
  - [x] 5.2 擴展支援 Rust 語言的智慧分塊
  - [x] 5.3 擴展支援 Java 語言的智慧分塊
  - [x] 5.4 實作 JSON/YAML 檔案的結構化分塊
  - [x] 5.5 實作 Markdown 檔案的標題層次分塊
  - [x] 5.6 實作進階元資料功能（chunk_id, content_hash, embedding_text）

- [ ] 6.0 測試和品質保證
  - [x] 6.1 建立 CodeParser 服務的全面單元測試
  - [x] 6.2 建立各種語言的測試範例程式碼檔案
  - [ ] 6.3 實作智慧分塊功能的整合測試
  - [ ] 6.4 建立語法錯誤處理的專項測試
  - [ ] 6.5 實作效能基準測試和記憶體使用監控
  - [ ] 6.6 建立端到端的索引和搜尋流程測試

- [ ] 7.0 文件更新和部署準備
  - [x] 7.1 更新 README.md 說明新的智慧分塊功能
  - [x] 7.2 更新 CLAUDE.md 的架構說明和開發指令
  - [x] 7.3 建立智慧分塊的使用指南和最佳實踐文件
  - [ ] 7.4 更新 manual_indexing.py 的使用說明和範例
  - [ ] 7.5 建立故障排除指南和常見問題解答
  - [ ] 7.6 準備版本發布說明和遷移指引