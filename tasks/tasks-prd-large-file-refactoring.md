## Relevant Files

- `src/services/code_parser_service.py` - 主要重構目標，將被拆分。
- `src/services/rag_search_strategy.py` - 主要重構目標，將被拆分。
- `src/mcp_tools.py` - 主要重構目標，將被移除並取代為工具套件。
- `manual_indexing.py` - 主要重構目標，將被拆分。
- `src/tools/` - 新工具模組的存放位置。
- `src/services/language_support_service.py` - 管理 Tree-sitter parsers 和語言相關配置的服務。
- `src/services/ast_extraction_service.py` - 封裝 AST 節點查詢、遍歷和屬性提取邏輯的服務。
- `src/services/chunking_strategies.py` - 定義抽象基礎類別和語言特定的程式碼分塊策略。
- `src/services/search_strategies.py` - 定義抽象基礎類別和具體的搜尋策略實作。
- `src/services/` - 新服務模組的存放位置。
- `tests/` - 需要為所有新建立的模組新增單元測試。

### Notes

- 重構的核心是將大型類別和檔案分解為更小、更專注的元件。
- 廣泛應用策略模式和註冊中心模式。
- 確保所有新模組都有對應的測試。

## Tasks

- [x] 1.0 重構 `code_parser_service.py`
  - [x] 1.1 建立 `src/services/language_support_service.py` 並實作 `LanguageSupportService` 類別，用於管理 Tree-sitter parsers。
  - [x] 1.2 建立 `src/services/ast_extraction_service.py` 並實作 `AstExtractionService` 類別，用於封裝 AST 節點查詢邏輯。
  - [x] 1.3 建立 `src/services/chunking_strategies.py`，在其中定義 `BaseChunkingStrategy` 抽象基礎類別。
  - [x] 1.4 在 `chunking_strategies.py` 中為至少兩種主要語言 (例如 Python, JavaScript) 建立具體的策略類別。
  - [x] 1.5 重構 `code_parser_service.py`，使其成為一個協調者，使用新建立的服務和策略。

- [ ] 2.0 重構 `rag_search_strategy.py`
  - [x] 2.1 建立 `src/services/search_strategies.py`，在其中定義 `BaseSearchStrategy` 抽象基礎類別。
  - [x] 2.2 在 `search_strategies.py` 中建立 `SemanticSearchStrategy`, `KeywordSearchStrategy`, 和 `HybridSearchStrategy` 具體策略類別。
  - [ ] 2.3 建立 `src/services/query_builder_service.py` 並實作 `QueryBuilderService`，用於建構 Qdrant 查詢。
  - [ ] 2.4 建立 `src/services/result_processing_service.py` 並實作 `ResultProcessingService`，用於格式化搜尋結果。
  - [ ] 2.5 重構 `rag_search_strategy.py`，使其成為一個協調者，調用策略和輔助服務。

- [ ] 3.0 重構 `mcp_tools.py`
  - [ ] 3.1 根據功能將 `mcp_tools.py` 中的工具函式移動到 `src/tools/` 下的對應子模組 (`indexing`, `project`, `core` 等)。
  - [ ] 3.2 在 `src/tools/registry.py` 中實作 `@register_tool` 裝飾器和 `TOOL_REGISTRY`。
  - [ ] 3.3 將所有被移動的工具函式應用 `@register_tool` 裝飾器。
  - [ ] 3.4 在 `src/tools/__init__.py` 中匯入所有工具模組以觸發註冊。
  - [ ] 3.5 全域搜尋並取代所有對 `mcp_tools.py` 的引用，改為從 `src.tools.registry` 取得工具。
  - [ ] 3.6 確認沒有任何引用後，刪除 `mcp_tools.py` 檔案。

- [ ] 4.0 重構 `manual_indexing.py`
  - [ ] 4.1 建立 `src/services/file_discovery_service.py` 並實作 `FileDiscoveryService`。
  - [ ] 4.2 建立 `src/services/indexing_pipeline.py` 並實作 `IndexingPipeline` 類別，用於協調整個索引流程。
  - [ ] 4.3 建立 `src/services/indexing_reporter.py` 並實作 `IndexingReporter` 類別。
  - [ ] 4.4 重構 `manual_indexing.py` 腳本，使其主要負責初始化並執行 `IndexingPipeline`。

- [ ] 5.0 建立單元測試並驗證重構
  - [ ] 5.1 為 `LanguageSupportService` 和 `AstExtractionService` 撰寫單元測試。
  - [ ] 5.2 為每個 `ChunkingStrategy` 和 `SearchStrategy` 撰寫單元測試。
  - [ ] 5.3 為 `QueryBuilderService` 和 `ResultProcessingService` 撰寫單元測試。
  - [ ] 5.4 為 `IndexingPipeline` 和其元件撰寫單元測試。
  - [ ] 5.5 確保所有與重構相關的現有測試在修改後依然能通過。