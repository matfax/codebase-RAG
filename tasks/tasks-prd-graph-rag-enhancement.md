# Tasks for Graph RAG 增強功能

Based on the PRD: `prd-graph-rag-enhancement.md`

## Relevant Files

- `src/models/code_chunk.py` - 擴展 CodeChunk 模型以支援 breadcrumb 和 parent_name 欄位（已有部分實作）
- `src/services/graph_rag_service.py` - 新建 Graph RAG 核心服務，處理結構關係圖建構和分析
- `src/services/structure_analyzer_service.py` - 新建結構分析服務，負責 breadcrumb 和 parent_name 提取
- `src/services/pattern_recognition_service.py` - 新建架構模式識別服務
- `src/services/cross_project_search_service.py` - 新建跨專案搜尋服務
- `src/tools/graph_rag/` - 新建目錄，包含所有 Graph RAG 相關的 MCP 工具
- `src/tools/graph_rag/structure_tools.py` - `graph_analyze_structure` MCP 工具實作
- `src/tools/graph_rag/search_tools.py` - `graph_find_similar_implementations` MCP 工具實作
- `src/tools/graph_rag/pattern_tools.py` - `graph_identify_patterns` MCP 工具實作
- `src/utils/breadcrumb_extractor.py` - 新建工具，從不同語言的代碼中提取 breadcrumb 資訊
- `src/utils/structure_relationship_builder.py` - 新建工具，建構代碼結構關係圖
- `tests/test_graph_rag_service.py` - Graph RAG 服務的單元測試
- `tests/test_structure_analyzer.py` - 結構分析服務的單元測試
- `tests/test_pattern_recognition.py` - 架構模式識別的單元測試
- `tests/test_cross_project_search.py` - 跨專案搜尋的單元測試
- `tests/test_graph_rag_tools.py` - Graph RAG MCP 工具的整合測試
- `docs/graph-rag-usage.md` - Graph RAG 功能使用說明文檔

### Notes

- Graph RAG 功能需要與現有的 Qdrant 服務、embedding 服務和搜尋工具深度整合
- 新的 MCP 工具應保持與現有工具介面的一致性
- breadcrumb 提取需要支援專案中已實作的所有程式語言（Python, JavaScript, TypeScript, Go, Rust, Java, C++）
- 使用 `python -m pytest tests/` 執行所有測試
- 測試前記得啟用 .venv
- 優先使用 Codebase RAG mcp tool

## Tasks

- [x] 1.0 擴展 CodeChunk 模型和結構分析功能
  - [x] 1.1 完善 CodeChunk 模型中的 breadcrumb 和 parent_name 欄位實作
  - [x] 1.2 實作 breadcrumb_extractor.py，支援從不同程式語言提取層次關係
  - [x] 1.3 建立 structure_analyzer_service.py，整合 breadcrumb 提取和 parent_name 識別
  - [x] 1.4 更新現有的 code_parser_service.py，在解析時填充新的結構欄位
  - [x] 1.5 建立 breadcrumb 和 parent_name 的資料驗證和正規化機制

- [ ] 2.0 實作 Graph RAG 核心服務層
  - [ ] 2.1 建立 graph_rag_service.py 作為 Graph RAG 功能的主要控制器
  - [ ] 2.2 實作 structure_relationship_builder.py，建構代碼結構關係圖
  - [ ] 2.3 開發關係圖的層次遍歷和相關組件查找算法
  - [ ] 2.4 實作結構關係的快取機制，提升查詢效能
  - [ ] 2.5 建立與現有 Qdrant 服務的深度整合介面

- [ ] 3.0 開發跨專案搜尋和架構模式識別
  - [ ] 3.1 實作 cross_project_search_service.py，支援基於結構關係的跨專案搜尋
  - [ ] 3.2 開發混合搜尋算法，結合語義相似性和結構關係過濾
  - [ ] 3.3 建立 pattern_recognition_service.py，識別常見架構模式
  - [ ] 3.4 實作完整實作鏈追蹤功能，從入口點到實作細節
  - [ ] 3.5 開發架構模式比較和分析功能

- [ ] 4.0 創建新的 MCP 工具介面
  - [ ] 4.1 建立 src/tools/graph_rag/ 目錄結構
  - [ ] 4.2 實作 graph_analyze_structure MCP 工具，分析特定 breadcrumb 的結構關係
  - [ ] 4.3 實作 graph_find_similar_implementations MCP 工具，跨專案相似實作搜尋
  - [ ] 4.4 實作 graph_identify_patterns MCP 工具，架構模式識別
  - [ ] 4.5 更新 main.py 註冊新的 MCP 工具
  - [ ] 4.6 確保新工具與現有搜尋工具的兼容性和一致性

- [ ] 5.0 整合測試和文檔更新
  - [ ] 5.1 建立所有新服務的單元測試（graph_rag_service, structure_analyzer, pattern_recognition, cross_project_search）
  - [ ] 5.2 建立 Graph RAG MCP 工具的整合測試
  - [ ] 5.3 測試與現有功能的兼容性，確保不影響現有搜尋和索引
  - [ ] 5.4 撰寫 Graph RAG 功能的使用說明文檔
  - [ ] 5.5 更新 CLAUDE.md，添加新的 MCP 工具說明
  - [ ] 5.6 進行端到端測試，驗證所有用戶故事場景
