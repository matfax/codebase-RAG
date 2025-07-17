# Tasks for Function Chain MCP Tools Implementation

## Relevant Files

- `src/services/breadcrumb_resolver_service.py` - Core service for converting natural language inputs to breadcrumb format
- `src/services/breadcrumb_resolver_service.test.py` - Unit tests for breadcrumb resolver service
- `src/tools/graph_rag/function_chain_analysis.py` - Implementation of trace_function_chain_tool MCP tool
- `src/tools/graph_rag/function_chain_analysis.test.py` - Unit tests for function chain analysis tool
- `src/tools/graph_rag/function_path_finding.py` - Implementation of find_function_path_tool MCP tool
- `src/tools/graph_rag/function_path_finding.test.py` - Unit tests for function path finding tool
- `src/tools/graph_rag/project_chain_analysis.py` - Implementation of analyze_project_chains_tool MCP tool
- `src/tools/graph_rag/project_chain_analysis.test.py` - Unit tests for project chain analysis tool
- `src/utils/output_formatters.py` - Utility functions for formatting tool outputs (arrow format, Mermaid diagrams)
- `src/utils/output_formatters.test.py` - Unit tests for output formatters
- `src/utils/complexity_calculator.py` - Utility for calculating function complexity scores based on specified weights
- `src/utils/complexity_calculator.test.py` - Unit tests for complexity calculator
- `src/tools/registry.py` - Updated to register new function chain MCP tools
- `src/tools/graph_rag/__init__.py` - Updated to export new function chain tools
- `docs/MCP_TOOLS.md` - Updated documentation for new function chain tools
- `docs/GRAPH_RAG_ARCHITECTURE.md` - Updated to include function chain concepts and examples
- `docs/examples/function_chain_examples.md` - Usage examples and tutorials for function chain tools

### Notes

- Unit tests should be placed alongside the code files they are testing
- Use `python -m pytest [optional/path/to/test/file]` to run tests
- All MCP tools should follow existing patterns in the codebase
- Integration tests should verify tools work with existing Graph RAG infrastructure

## Tasks

- [x] 1.0 實現共用 BreadcrumbResolver 服務
  - [x] 1.1 創建 BreadcrumbResolver 服務類，包含 resolve() 方法
  - [x] 1.2 實現 is_valid_breadcrumb() 檢查函數，驗證輸入是否為標準 breadcrumb 格式
  - [x] 1.3 實現 convert_natural_to_breadcrumb() 函數，使用現有搜尋服務進行語意轉換
  - [x] 1.4 添加錯誤處理和多候選結果支援，包含信心分數
  - [x] 1.5 實現快取機制，避免重複轉換相同的自然語言輸入
  - [x] 1.6 編寫 BreadcrumbResolver 的完整單元測試，覆蓋正常情況和邊界情況
  - [x] 1.7 添加日誌記錄，用於調試和監控轉換過程

- [ ] 2.0 實現 trace_function_chain_tool（最高優先級）
  - [ ] 2.1 創建 trace_function_chain() 函數，接受所有必要參數
  - [ ] 2.2 整合 BreadcrumbResolver 進行自然語言輸入處理
  - [ ] 2.3 實現與 ImplementationChainService 的整合，支援 forward/backward/bidirectional 追蹤
  - [ ] 2.4 實現預設箭頭格式輸出 (A => B => C)
  - [ ] 2.5 添加可選的 Mermaid 圖表輸出格式
  - [ ] 2.6 實現深度控制邏輯，預設最大深度為 10
  - [ ] 2.7 添加分支點和終端點識別功能
  - [ ] 2.8 實現錯誤處理，包含建議使用搜尋工具的提示
  - [ ] 2.9 添加執行時間追蹤和效能監控
  - [ ] 2.10 編寫完整的單元測試，包含各種追蹤方向和邊界情況
  - [ ] 2.11 實現整合測試，驗證與現有 Graph RAG 基礎設施的相容性

- [ ] 3.0 實現 find_function_path_tool（次優先級）
  - [ ] 3.1 創建 find_function_path() 函數，支援起點和終點參數
  - [ ] 3.2 整合 BreadcrumbResolver 處理兩個函數的自然語言輸入
  - [ ] 3.3 實現多路徑查找邏輯，支援 shortest/optimal/all 策略
  - [ ] 3.4 實現路徑品質評估，包含可靠性和複雜度評分
  - [ ] 3.5 添加路徑多樣性分析，計算關係類型的多樣性
  - [ ] 3.6 實現路徑結果限制，預設最多返回 5 個路徑
  - [ ] 3.7 添加路徑比較和推薦功能（最直接、最可靠路徑）
  - [ ] 3.8 實現箭頭格式和 Mermaid 格式的路徑輸出
  - [ ] 3.9 添加路徑不存在時的錯誤處理和建議
  - [ ] 3.10 編寫完整的單元測試，涵蓋各種路徑查找策略
  - [ ] 3.11 實現效能測試，確保在大型代碼庫中的響應時間

- [ ] 4.0 實現 analyze_project_chains_tool（最低優先級）
  - [ ] 4.1 創建 analyze_project_chains() 函數，支援專案範圍分析
  - [ ] 4.2 實現範圍限制功能，支援 breadcrumb 模式匹配 (如 "api.*")
  - [ ] 4.3 整合複雜度計算器，使用指定權重：分支(35%)、循環複雜度(30%)、調用深度(25%)、函數行數(10%)
  - [ ] 4.4 實現熱點路徑識別，分析使用頻率和關鍵性評分
  - [ ] 4.5 添加覆蓋率和連接性統計計算
  - [ ] 4.6 實現重構建議邏輯，基於複雜度分析識別需要重構的函數
  - [ ] 4.7 添加專案級指標計算：平均鏈深度、總入口點數、連接評分
  - [ ] 4.8 實現鏈類型篩選，支援 execution_flow/data_flow/dependency_chain
  - [ ] 4.9 添加報告生成功能，包含建議和統計摘要
  - [ ] 4.10 編寫完整的單元測試，涵蓋各種專案範圍和分析類型
  - [ ] 4.11 實現大型專案的效能最佳化，包含分批處理和進度追蹤

- [ ] 5.0 整合、測試與文件更新
  - [ ] 5.1 創建輸出格式化工具，統一處理箭頭格式和 Mermaid 圖表生成
  - [ ] 5.2 實現複雜度計算器，支援可配置的權重系統
  - [ ] 5.3 在 tools/registry.py 中註冊所有三個新的 MCP 工具
  - [ ] 5.4 更新 tools/graph_rag/__init__.py 以導出新工具
  - [ ] 5.5 實現端到端整合測試，驗證所有工具的完整工作流程
  - [ ] 5.6 編寫效能測試，確保在大型代碼庫中的回應時間 <2 秒
  - [ ] 5.7 更新 docs/MCP_TOOLS.md，添加三個新工具的完整文件
  - [ ] 5.8 更新 docs/GRAPH_RAG_ARCHITECTURE.md，包含 Function Chain 概念和架構說明
  - [ ] 5.9 創建 docs/examples/function_chain_examples.md，提供實用範例和教學
  - [ ] 5.10 實現使用者接受度測試，驗證自然語言輸入轉換的準確性 (目標 >90%)
  - [ ] 5.11 執行完整的回歸測試，確保不影響現有 Graph RAG 工具功能
  - [ ] 5.12 建立監控和日誌系統，追蹤工具使用情況和效能指標
