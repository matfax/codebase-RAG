# Agent 開發與偵錯日誌

本文檔記錄了在為 `multi_modal_search` 工具新增 `minimal_output` 功能時，對專案架構的分析、遇到的問題、以及最終的解決方案，供未來開發參考。

## 1. 專案測試架構分析

在為新功能撰寫單元測試的過程中，我們釐清了本專案的測試架構與開發流程：

- **測試框架**: 專案使用 `pytest` 作為主要的測試框架，並搭配 `pytest-asyncio` 處理非同步程式碼的測試。

- **CI/CD 流程**: `.github/workflows/main.yml` 中定義了持續整合流程。此流程會在對 `main` 或 `develop` 分支的 `push` 及 `pull_request` 事件時自動觸發。流程中包含程式碼品質檢查 (Ruff, Black, Mypy) 和單元測試。

- **測試品質要求**: CI 流程中執行 `pytest` 時帶有 `--cov-fail-under=90` 參數，這表示專案有嚴格的測試覆蓋率要求，必須達到 90% 以上，否則 CI 會失敗。

- **Python 路徑與環境**:
    1. 專案原始碼位於 `src` 目錄下。
    2. 為了讓 Python 和 `pytest` 能正確解析 `from src.tools...` 這樣的匯入，必須先透過 `uv pip install -e .` 將專案以「可編輯模式」安裝到虛擬環境中。
    3. 為了讓第 2 點成功，`pyproject.toml` 必須正確設定 `[tool.hatch.build.targets.wheel]`，將 `packages` 指向 `src` 目錄。

- **模擬 (Mocking)**:
    1. 專案使用 `unittest.mock` 進行模擬，特別是 `@patch` 裝飾器。
    2. **關鍵發現**: `@patch` 裝飾器中的路徑**必須是從專案根目錄 (`src`) 開始的絕對路徑** (例如：`@patch("src.tools.indexing.search_tools.get_qdrant_client")`)，而不是相對於 `tests` 目錄的路徑。
    3. 對於非同步函式 (`async def`) 的模擬，必須使用 `AsyncMock` 而非標準的 `Mock`。

### 測試執行指令

根據我們的偵錯經驗，以下是在本機環境中正確執行測試的標準流程：

1.  **安裝/更新依賴**: 在執行任何測試之前，最關鍵的步驟是確保專案已在虛擬環境中以可編輯模式安裝。如果 `pyproject.toml` 有任何變更，或是不確定環境是否為最新，都應執行此指令。
    ```bash
    source .venv/bin/activate && uv pip install -e .
    ```

2.  **執行測試**: 安裝完成後，可以直接使用 `pytest` 執行測試。`pytest` 會讀取 `pytest.ini` 中的設定 (`pythonpath = src`)，並正確找到所有模組。
    ```bash
    source .venv/bin/activate && pytest tests/test_advanced_search_tool.py -v
    ```

**註**: 我們曾嘗試過其他方法，例如直接設定 `PYTHONPATH` 環境變數，但都因為會與 `src` 目錄內的相對匯入 (`...`) 產生衝突而失敗。因此，`pip install -e .` 是確保測試環境與 CI 環境一致的最佳實踐。

## 2. 主要修正：`minimal_output` 功能實作與偵錯歷程

本次任務的核心是為 `multi_modal_search` 工具新增一個 `minimal_output` 參數，使其預設回傳簡潔的結果，同時保留回傳完整詳細資訊的選項。

整個過程充滿挑戰，以下是我們的偵錯與修正歷程：

1.  **初步實作**: 我們首先修改了工具的原始碼檔案 (`src/tools/indexing/multi_modal_search_tools.py`)，在函式簽名中加入了 `minimal_output` 參數，並在函式末尾加入了根據此參數回傳不同結果的邏輯。

2.  **問題 1：功能未生效**: 經過手動測試，發現即使傳遞了 `minimal_output=False`，回傳的結果依然是簡化後的版本。

3.  **調查 1：找到真正根源 `registry.py`**: 在您的指導下，我們檢查了 `src/tools/registry.py` 檔案。我們發現，`mcp server` 是透過 `@mcp_app.tool()` 裝飾器來註冊工具的，而這個註冊區塊中的函式簽名是固定的，並沒有我們新加的 `minimal_output` 參數。**這是問題的真正根源**，伺服器使用的是註冊時的舊簽名，因此完全忽略了我們傳遞的新參數。

4.  **修正 1：更新註冊表**: 我們修正了 `registry.py`，在 `@mcp_app.tool()` 裝飾的函式簽名中加入了 `minimal_output`，並將其傳遞給底層的實作函式。

5.  **問題 2：單元測試環境配置**: 在核心邏輯修正後，我們開始編寫單元測試，但遭遇了一連串的 `ModuleNotFoundError` 和 `ImportError`。

6.  **調查 2：解決測試路徑問題**: 我們逐步排查，嘗試了多種方法（設定 `PYTHONPATH`、修改 `sys.path`），最終確定了正確的配置方法，如本文件第一部分所述。

7.  **修正 2：完善測試案例**: 我們修正了測試檔案中的所有 `@patch` 路徑，使其從 `src` 開始。同時，我們也修正了測試邏輯中的錯誤，例如將 `Mock` 改為 `AsyncMock`，以及傳遞正確的參數名稱 (`target_projects` 而非 `project_names`)。

## 3. 結論

經過多次的偵錯與修正，我們成功地完成了 `minimal_output` 功能的開發，並為其建立了穩固的單元測試。整個過程不僅達成了最初的目標，更讓我們對專案的註冊機制、測試架構和環境配置有了非常深入的了解。

這次的經驗證明了在修改功能時，必須考慮到從「**實作層**」到「**註冊層**」再到「**測試層**」的完整鏈路，任何一個環節的疏忽都可能導致功能不符預期。
