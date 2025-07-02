# PRD: 大型 Python 檔案重構

## 1. 簡介/概觀

此計畫旨在重構專案中幾個超過 800 行的 Python 檔案。這些大型檔案難以維護、理解，並且降低了開發人員和 AI Agent 的協作效率。透過將它們分解為更小、更專注、遵循單一職責原則的模組，我們旨在提高程式碼庫的整體品質、可維護性和開發速度。

## 2. 目標

*   將 `code_parser_service.py`, `rag_search_strategy.py`, `mcp_tools.py`, 和 `manual_indexing.py` 的核心邏輯拆分到獨立的模組/類別中。
*   針對 `code_parser_service.py` 和 `rag_search_strategy.py`，應用**策略模式 (Strategy Pattern)** 來管理特定於語言或演算法的邏輯。
*   針對 `mcp_tools.py`，將其從單一檔案重構為一個結構化的工具套件，並建立一個動態註冊中心。
*   顯著降低重構後模組的圈複雜度 (Cyclomatic Complexity)。
*   確保所有新建立的模組都有對應的單元測試。

## 3. 使用者故事

*   **開發人員:** 作為一名開發人員，我希望能處理更小、職責更單一的模組，以便我能更容易地理解、維護和擴展程式碼，而不會引發意想不到的副作用。
*   **AI Agent:** 作為一個 AI Agent，我希望能讀取更小、更專注的檔案，以便我能更快地處理上下文，並提供更準確、更高效的程式碼生成或修改輔助。

## 4. 功能需求

1.  **重構 `src/services/code_parser_service.py`**
    *   1.1. 建立 `LanguageSupportService`，專門用於管理 Tree-sitter 的 parsers。
    *   1.2. 建立 `AstExtractionService`，用於封裝所有與 AST 節點查詢相關的邏輯。
    *   1.3. 建立一個抽象基礎類別 `BaseChunkingStrategy` 和多個具體的策略類別 (如 `PythonChunkingStrategy`)，以處理不同語言的分塊邏輯。
    *   1.4. 將 `CodeParserService` 重構為一個協調者，負責調用上述新服務和策略來完成工作。

2.  **重構 `src/services/rag_search_strategy.py`**
    *   2.1. 建立一個抽象基礎類別 `BaseSearchStrategy` 和多個具體的策略類別 (如 `SemanticSearchStrategy`, `HybridSearchStrategy`) 來封裝不同的搜尋演算法。
    *   2.2. 建立 `QueryBuilderService`，專門負責建構 Qdrant 的查詢。
    *   2.3. 建立 `ResultProcessingService`，專門負責處理和格式化從資料庫返回的結果。
    *   2.4. 將主要的 RAG 服務重構為一個協調者，調用策略和輔助服務。

3.  **重構 `src/mcp_tools.py`**
    *   3.1. 將 `mcp_tools.py` 中的所有工具函式，根據其功能分類並移動到 `src/tools/` 下的對應子模組中 (例如 `indexing/`, `project/`, `core/`)。
    *   3.2. 在 `src/tools/registry.py` 中建立一個工具註冊中心和 `@register_tool` 裝飾器。
    *   3.3. 將所有工具函式用 `@register_tool` 裝飾，使其能被自動發現。
    *   3.4. 更新整個專案中對舊工具模組的引用，然後安全地刪除 `mcp_tools.py`。

4.  **重構 `manual_indexing.py`**
    *   4.1. 分析 `manual_indexing.py` 的腳本邏輯，將其分解為可重用的元件。
    *   4.2. 建立 `FileDiscoveryService`，負責根據規則尋找要索引的檔案。
    *   4.3. 建立 `IndexingPipeline`，負責協調整個索引流程 (讀取、解析、分塊、嵌入、儲存)。
    *   4.4. 建立 `IndexingReporter`，負責產生索引過程的報告和統計數據。
    *   4.5. 將 `manual_indexing.py` 重構為一個更簡潔的腳本，主要負責設定和啟動 `IndexingPipeline`。

## 5. 非目標 (範圍之外)

*   重構此 PRD 中未列出的其他檔案。
*   在此次重構範圍內實現任何新的功能。
*   對外暴露的工具 API 簽名 (除非重構的必要性)。
*   更新專案的整體架構文件 (可在重構完成後單獨進行)。

## 6. 技術考量

*   廣泛使用**策略設計模式**來處理可變的演算法和邏輯。
*   為 `mcp_tools.py` 的重構實現**註冊中心和裝飾器模式**。
*   所有新建立的模組都必須有對應的**單元測試**，並整合到現有的測試流程中。

## 7. 成功指標

*   重構後，原先在四個大型檔案中的核心業務邏輯，其**圈複雜度**應有顯著降低。
*   所有新建立的、包含業務邏輯的模組，都必須有單元測試覆蓋。
*   `code_parser_service.py`, `rag_search_strategy.py`, 和 `manual_indexing.py` 的程式碼行數顯著減少。
*   `mcp_tools.py` 檔案被完全移除。

## 8. 開放問題

*   目前主要的功能性問題都已釐清。後續問題可能涉及具體實作細節。
