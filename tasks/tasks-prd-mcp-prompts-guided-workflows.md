## Relevant Files

- `src/mcp_prompts.py` - 新增的 MCP Prompts 核心實現模組
- `src/mcp_prompts.test.py` - MCP Prompts 功能的單元測試
- `src/services/prompt_service.py` - Prompts 執行和管理服務
- `src/services/prompt_service.test.py` - Prompt 服務的單元測試
- `src/services/recommendation_service.py` - 智能推薦系統服務
- `src/services/recommendation_service.test.py` - 推薦服務的單元測試
- `src/services/context_manager_service.py` - 上下文記憶和狀態管理服務
- `src/services/context_manager_service.test.py` - 上下文管理服務的單元測試
- `src/mcp_tools.py` - 擴展現有 MCP 工具註冊以支援 Prompts
- `src/utils/prompt_validator.py` - Prompt 參數驗證和安全性檢查工具
- `src/utils/workflow_orchestrator.py` - Prompt 鏈接和工作流程編排工具
- `src/models/prompt_context.py` - Prompt 上下文資料模型
- `src/models/recommendation.py` - 推薦結果資料模型

### Notes

- 使用 `python -m pytest tests/` 執行所有測試
- 新增的 Prompts 功能需要與現有的智能代碼分塊系統整合
- 確保新功能不影響現有搜索和索引功能的性能

## Tasks

- [x] 1.0 實現 MCP Prompts 基礎架構
  - [x] 1.1 研究 FastMCP Prompts API 和實現方式
  - [x] 1.2 設計 Prompt 註冊和發現機制
  - [x] 1.3 建立 Prompt 參數驗證框架
  - [x] 1.4 實現 Prompt 執行引擎核心邏輯
  - [x] 1.5 建立 Prompt 錯誤處理和日誌系統
  - [x] 1.6 創建 Prompt 基礎類別和介面定義
  - [x] 1.7 編寫 Prompts 基礎架構的單元測試

- [ ] 2.0 開發核心探索性 Prompts
  - [ ] 2.1 實現 explore_project Prompt - 項目架構分析
  - [ ] 2.2 實現 understand_component Prompt - 組件深度分析
  - [ ] 2.3 實現 trace_functionality Prompt - 功能追蹤分析
  - [ ] 2.4 實現 find_entry_points Prompt - 入口點識別
  - [ ] 2.5 建立項目分析結果的格式化和展示邏輯
  - [ ] 2.6 整合現有的代碼解析和搜索服務
  - [ ] 2.7 編寫探索性 Prompts 的測試案例和驗證

- [ ] 3.0 建構智能推薦系統
  - [ ] 3.1 設計推薦演算法和評分機制
  - [ ] 3.2 實現 suggest_next_steps Prompt - 下一步行動推薦
  - [ ] 3.3 實現 optimize_search Prompt - 搜索策略優化
  - [ ] 3.4 建立用戶行為分析和偏好學習機制
  - [ ] 3.5 實現基於上下文的個性化推薦
  - [ ] 3.6 建立推薦結果的品質評估機制
  - [ ] 3.7 編寫推薦系統的效能和準確性測試

- [ ] 4.0 實現工作流程編排和上下文管理
  - [ ] 4.1 設計 Prompt 鏈接和工作流程狀態機制
  - [ ] 4.2 實現上下文記憶和持久化存儲
  - [ ] 4.3 建立工作流程暫停和恢復功能
  - [ ] 4.4 實現跨 Prompt 的資料傳遞機制
  - [ ] 4.5 建立用戶探索歷程的追蹤和分析
  - [ ] 4.6 實現工作流程的錯誤恢復和重試邏輯
  - [ ] 4.7 編寫工作流程管理的整合測試

- [ ] 5.0 整合現有系統並優化使用者體驗
  - [ ] 5.1 修改 mcp_tools.py 以註冊新的 Prompts
  - [ ] 5.2 確保 Prompts 在 Claude Code 中正確顯示為斜杠命令
  - [ ] 5.3 實現 Prompts 的說明文檔和使用範例
  - [ ] 5.4 整合效能監控和記憶體管理機制
  - [ ] 5.5 建立 Prompts 功能的配置和客製化選項
  - [ ] 5.6 實現完整的端到端測試場景
  - [ ] 5.7 編寫使用者指南和技術文檔
  - [ ] 5.8 執行效能基準測試和優化調整