# PRD: 智慧程式碼分塊系統 (Intelligent Code Chunking System)

## 1. Introduction/Overview

目前的 Codebase RAG MCP Server 採用「整檔案 embedding」策略，將每個完整檔案作為一個區塊進行向量化和索引。這種方法存在三個關鍵問題，嚴重影響了程式碼搜尋和理解的效果：

### 核心問題

1. **上下文稀釋 (Context Dilution)**：檔案中不相關的部分（多個函數、類別、匯入語句）會互相干擾，使得產生的單一向量無法精確地代表檔案中任何一個具體的功能。

2. **長度限制 (Length Limitations)**：大多數 embedding 模型都有輸入長度限制，大型檔案無法完整處理或會被截斷。

3. **檢索精度低 (Poor Retrieval Precision)**：當用戶搜尋特定函數或類別時，系統只能返回整個檔案，無法精確定位到具體的程式碼片段。

### 解決方案

本 PRD 提出實作基於 **Tree-sitter** 的智慧程式碼分塊系統，利用程式語言的語法結構進行語義感知的程式碼分割，將每個函數、類別、方法作為獨立的可搜尋單元，從根本上提升程式碼搜尋的精確度和相關性。

## 2. Goals

### 主要目標

- **G-1**: 提升程式碼搜尋精確度，實現函數級和類別級的精確檢索
- **G-2**: 改善 embedding 向量的語義表現，每個向量代表一個完整的語義單元
- **G-3**: 保持系統穩定性和效能，確保大型專案的可處理性
- **G-4**: 支援主流程式語言的智慧分塊，覆蓋 80% 的常見開發場景

### 成功指標

- 搜尋精度提升：函數命中率從當前的整檔案匹配提升到函數級精確匹配
- 處理效率：AST 解析時間控制在毫秒級，不顯著影響總體索引時間
- 錯誤容忍：語法錯誤檔案的處理成功率達到 95% 以上
- 語言覆蓋：支援 10+ 主流程式語言的智慧分塊

## 3. User Stories

### US-1: 精確函數搜尋
**作為** 開發者
**我想要** 搜尋「用戶驗證」時能直接找到 `validateUser()` 函數
**而不是** 得到包含該函數的整個檔案
**以便** 快速理解和使用特定功能

### US-2: 類別方法導航
**作為** AI 助手
**我想要** 理解 `UserService` 類別的具體方法
**我希望** 每個方法都是獨立的可搜尋單元
**以便** 提供精確的程式碼建議和解釋

### US-3: 大型專案程式碼理解
**作為** 新加入專案的開發者
**我想要** 搜尋「資料庫連接」時獲得相關的函數和配置
**而不是** 大量無關的整檔案結果
**以便** 快速了解專案架構和實作細節

### US-4: 錯誤程式碼處理
**作為** 系統使用者
**我想要** 即使專案中有語法錯誤的檔案
**系統仍能** 索引其他正確的程式碼部分
**以便** 不因個別檔案問題影響整體搜尋功能

## 4. Functional Requirements

### FR-1: CodeParser 服務架構

**FR-1.1**: 建立新的 `src/services/code_parser_service.py` 服務
**FR-1.2**: 整合 Tree-sitter 語法解析器框架
**FR-1.3**: 提供統一的檔案解析和分塊介面
**FR-1.4**: 支援多語言解析器的動態載入和管理

### FR-2: 智慧分塊邏輯

**FR-2.1**: **函數/方法宣告分塊**
- 將每個函數定義作為獨立區塊
- 包含完整的函數體和相關文檔
- 保留函數簽名和參數資訊

**FR-2.2**: **類別結構分塊**
- 類別宣告本身作為一個區塊（包含屬性和類別文檔）
- 類別內的每個方法作為獨立區塊
- 保留繼承關係和介面實作資訊

**FR-2.3**: **頂層常數和變數分塊**
- 複雜的頂層物件字面量（如配置物件）作為獨立區塊
- 簡單變數宣告歸入檔案級上下文區塊

**FR-2.4**: **介面和型別定義分塊**
- TypeScript/Java 介面定義作為獨立區塊
- 型別別名和泛型定義作為獨立區塊

### FR-3: 語言支援

**FR-3.1**: **Phase 1 語言支援** (MVP)
```python
SUPPORTED_LANGUAGES = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript'
}
```

**FR-3.2**: **Phase 2 語言擴展**
```python
ADDITIONAL_LANGUAGES = {
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c'
}
```

**FR-3.3**: **解析器生命週期管理**
- 自動載入所需的 Tree-sitter 語言解析器
- 解析器快取和重用機制
- 解析失敗時的降級處理

### FR-4: 錯誤處理和容錯機制

**FR-4.1**: **Tree-sitter ERROR 節點處理**
- 跳過純 ERROR 節點，避免無效內容索引
- 保留包含少量錯誤的大型正確區塊
- 將錯誤附近的正確程式碼納入以保持語義完整性

**FR-4.2**: **錯誤統計和報告**
- 記錄每個檔案的語法錯誤數量和位置
- 在 `manual_indexing.py` 完成後提供錯誤摘要
- 區分「輕微錯誤」vs「嚴重錯誤」的影響等級

**FR-4.3**: **降級策略**
- AST 解析完全失敗時，回退到整檔案處理
- 保持系統穩定性，不因個別檔案問題中斷整個索引過程

### FR-5: 元資料豐富化

**FR-5.1**: **基礎元資料** (MVP)
```python
{
    "file_path": str,              # 原始檔案路徑
    "chunk_type": str,             # function|class|method|constant|interface
    "name": str,                   # 函數名、類別名等
    "signature": str,              # 函數簽名、類別繼承關係
    "start_line": int,             # 起始行號
    "end_line": int,               # 結束行號
    "language": str,               # 程式語言
    "docstring": str,              # 提取的文檔字串
    "access_modifier": str,        # public/private/protected
    "parent_class": str,           # 如果是方法，所屬的類別
    "imports_used": List[str],     # 靜態分析得出的依賴
    "has_syntax_errors": bool,     # 是否包含語法錯誤
    "error_details": str           # 語法錯誤詳情
}
```

**FR-5.2**: **進階元資料** (Phase 2)
```python
{
    "chunk_id": str,               # UUID v5(file_path + name + start_line)
    "content_hash": str,           # SHA256(chunk_content)
    "embedding_text": str,        # 預處理後的 embedding 輸入文字
    "cyclomatic_complexity": int   # 圈複雜度分析（可選）
}
```

### FR-6: 搜尋結果上下文增強

**FR-6.1**: **檔案內上下文**
- 提供程式碼片段前後 5 行的上下文
- 自動包含相關的匯入語句
- 顯示麵包屑導覽 (`file_path > class_name > method_name`)

**FR-6.2**: **文檔和註解整合**
- 自動提取和關聯函數/類別的文檔字串
- 包含相關的內嵌註解
- 保留原始程式碼的格式和縮排

### FR-7: 非程式碼檔案處理

**FR-7.1**: **JSON/YAML 結構化分塊**
```python
# package.json 範例
{
    "scripts": {...},         # 作為一個區塊
    "dependencies": {...},    # 作為另一個區塊
    "devDependencies": {...}  # 作為第三個區塊
}
```

**FR-7.2**: **Markdown 標題分塊**
```markdown
## Section 1           # 一個區塊（包含到下個 ## 前的所有內容）
### Subsection A       # 子區塊
### Subsection B       # 另一個子區塊
## Section 2           # 新的頂層區塊
```

**FR-7.3**: **純文字段落分塊**
- 按空白行分隔的段落進行分塊
- 保持段落的語義完整性

## 5. Technical Architecture

### 架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Tools Layer                         │
│  (index_directory, search - 使用智慧分塊作為預設行為)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                IndexingService                             │
│  (orchestrator - 呼叫 CodeParser 而非直接讀取檔案)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               CodeParserService (新增)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Tree-sitter Integration                                │ │
│  │ - Language Parser Management                           │ │
│  │ - AST Generation & Traversal                          │ │
│  │ - Error Node Handling                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Intelligent Chunking Engine                           │ │
│  │ - Function/Method Extraction                          │ │
│  │ - Class Structure Analysis                            │ │
│  │ - Metadata Enrichment                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Multi-Language Support                                │ │
│  │ - Python, JavaScript, TypeScript                     │ │
│  │ - Go, Rust, Java (Phase 2)                          │ │
│  │ - JSON/YAML/Markdown Structured Parsing              │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Existing Services                             │
│  EmbeddingService │ QdrantService │ ProjectAnalysisService  │
└─────────────────────────────────────────────────────────────┘
```

### 核心服務設計

#### CodeParserService (`src/services/code_parser_service.py`)

```python
class CodeParserService:
    def __init__(self):
        self.parser_cache = {}  # 語言解析器快取
        self.supported_languages = SUPPORTED_LANGUAGES

    def parse_file(self, file_path: str) -> List[CodeChunk]:
        """
        解析檔案並返回智慧分塊結果

        Returns:
            List[CodeChunk]: 解析出的程式碼區塊清單
        """

    def get_parser(self, language: str) -> TreeSitterParser:
        """獲取或建立指定語言的解析器"""

    def extract_chunks_from_ast(self, ast_tree, file_content: str) -> List[CodeChunk]:
        """從 AST 中提取語義區塊"""

    def handle_syntax_errors(self, node, file_content: str) -> Optional[CodeChunk]:
        """處理包含語法錯誤的節點"""

    def enrich_metadata(self, chunk: CodeChunk, ast_node) -> CodeChunk:
        """豐富區塊的元資料資訊"""
```

#### CodeChunk 資料結構

```python
@dataclass
class CodeChunk:
    content: str                    # 程式碼內容
    file_path: str                 # 檔案路徑
    chunk_type: ChunkType          # 區塊類型枚舉
    name: str                      # 函數/類別名稱
    signature: str                 # 函數簽名
    start_line: int               # 起始行號
    end_line: int                 # 結束行號
    language: str                 # 程式語言
    docstring: Optional[str]      # 文檔字串
    access_modifier: Optional[str] # 存取修飾符
    parent_class: Optional[str]   # 父類別（如果是方法）
    imports_used: List[str]       # 使用的匯入
    has_syntax_errors: bool       # 是否有語法錯誤
    error_details: Optional[str]  # 錯誤詳情

    # Phase 2 進階欄位
    chunk_id: Optional[str] = None
    content_hash: Optional[str] = None
    embedding_text: Optional[str] = None
    cyclomatic_complexity: Optional[int] = None

class ChunkType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    CONSTANT = "constant"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    CONFIG_OBJECT = "config_object"
    IMPORT_BLOCK = "import_block"
```

### 與現有系統整合

#### IndexingService 修改

```python
# 原有的 _process_single_file 方法將被替換
def _process_single_file(self, file_path: str) -> List[Chunk]:
    """
    使用 CodeParserService 進行智慧分塊
    降級機制：如果智慧分塊失敗，回退到整檔案處理
    """
    try:
        # 使用新的 CodeParserService
        code_chunks = self.code_parser.parse_file(file_path)

        # 轉換為現有的 Chunk 格式
        return [self._convert_to_legacy_chunk(cc) for cc in code_chunks]

    except Exception as e:
        # 降級到原有的整檔案處理
        self.logger.warning(f"智慧分塊失敗，回退到整檔案處理: {e}")
        return self._fallback_to_whole_file(file_path)
```

#### MCP Tools 增強

```python
@app.tool()
def index_directory(
    directory: str,
    patterns: Optional[List[str]] = None,
    recursive: bool = True,
    clear_existing: bool = False,
    incremental: bool = False,
    project_name: Optional[str] = None  # 新增：允許自訂項目名稱
) -> Dict[str, Any]:
    """
    智慧分塊現在是預設行為
    project_name 參數允許用戶控制集合命名
    """
    if project_name is None:
        project_name = Path(directory).name  # 向下相容

    # ... 其餘邏輯保持不變，但內部使用智慧分塊
```

## 6. Non-Goals (明確排除的功能)

### NG-1: 向後相容性支援
- **不支援**：同時維護整檔案和智慧分塊兩套索引
- **理由**：專案尚未達到穩定發布階段，用戶基數較小，直接升級更簡潔有效

### NG-2: 用戶自訂分塊規則
- **不支援**：允許用戶配置分塊策略或閾值
- **理由**：增加系統複雜度，最佳實踐已經能覆蓋大部分場景

### NG-3: 大型函數細分
- **不支援**：將超長函數進一步切分為子區塊
- **理由**：保持函數語義完整性，現代模型已支援足夠長的上下文

### NG-4: 即時增量分析
- **不支援**：檔案修改時的即時重新分塊
- **理由**：複雜度高，推薦使用離線 `manual_indexing.py` 處理大型變更

### NG-5: 跨檔案依賴分析 (Phase 1)
- **不支援**：函數呼叫關係、繼承鏈分析
- **理由**：留待 Phase 2 實作，需要額外的程式碼圖譜支援

## 7. Implementation Plan

### Phase 1: MVP 智慧分塊核心 (4-6 週)

#### Week 1-2: 基礎架構
- [ ] 建立 `CodeParserService` 服務框架
- [ ] 整合 Tree-sitter 核心功能
- [ ] 在 `pyproject.toml` 中新增語言解析器依賴
- [ ] 實作基礎的 AST 遍歷和節點識別

#### Week 3-4: 核心分塊邏輯
- [ ] 實作 Python/JavaScript/TypeScript 分塊邏輯
- [ ] 建立 `CodeChunk` 資料結構和轉換機制
- [ ] 實作錯誤節點處理和降級策略
- [ ] 基礎元資料提取（函數名、簽名、行號等）

#### Week 5-6: 系統整合與測試
- [ ] 修改 `IndexingService` 整合智慧分塊
- [ ] 更新 `index_directory` MCP 工具
- [ ] 實作檔案內上下文增強
- [ ] 全面測試和錯誤處理完善

### Phase 2: 語言擴展與優化 (3-4 週)

#### Week 7-8: 語言支援擴展
- [ ] 新增 Go、Rust、Java 語言支援
- [ ] 實作 JSON/YAML 結構化分塊
- [ ] 新增 Markdown 標題層次分塊
- [ ] 純文字檔案段落分塊

#### Week 9-10: 進階功能
- [ ] 實作進階元資料（chunk_id, content_hash）
- [ ] 改進 embedding 文字預處理
- [ ] 新增圈複雜度分析（可選）
- [ ] 效能優化和快取機制

### 關鍵里程碑

1. **M1**: Tree-sitter 基礎架構完成，可解析 Python 檔案
2. **M2**: 智慧分塊邏輯完成，可提取函數和類別
3. **M3**: 系統整合完成，新分塊機制作為預設行為
4. **M4**: 多語言支援完成，覆蓋主流開發語言
5. **M5**: 進階功能完成，系統達到生產就緒狀態

## 8. Success Metrics

### 定量指標

#### 搜尋精度指標
- **函數級命中率**: 搜尋特定函數名時的精確匹配率 > 90%
- **相關性評分**: 搜尋結果的語義相關性評分提升 > 40%
- **檢索召回率**: 相關程式碼片段的檢索召回率 > 85%

#### 效能指標
- **AST 解析時間**: 單檔案解析時間 < 100ms (99th percentile)
- **記憶體使用**: 解析過程記憶體增量 < 50MB per 1000 files
- **索引總時間**: 相比整檔案方式的時間增量 < 20%

#### 錯誤處理指標
- **語法錯誤容忍率**: 包含語法錯誤檔案的成功處理率 > 95%
- **降級成功率**: AST 解析失敗時的降級處理成功率 = 100%
- **錯誤檢測準確率**: 語法錯誤的正確識別和分類 > 90%

### 定性指標

#### 用戶體驗改善
- **搜尋結果相關性**: 用戶反饋搜尋結果更相關、更有用
- **程式碼理解速度**: 新開發者理解專案結構的時間縮短
- **AI 助手準確性**: Claude 等 AI 助手提供的程式碼建議更精確

#### 系統穩定性
- **向下相容性**: 現有 MCP 工具使用者無感知升級
- **系統可靠性**: 無因智慧分塊導致的系統崩潰或資料損失
- **錯誤恢復能力**: 遇到問題時能夠優雅降級，不影響整體功能

## 9. Risk Assessment

### 高風險項目

#### R-1: Tree-sitter 依賴風險
**風險**: Tree-sitter 語言解析器更新或不相容問題
**影響**: 特定語言的解析可能失敗
**緩解措施**:
- 在 `pyproject.toml` 中鎖定解析器版本
- 實作降級機制，解析失敗時回退到整檔案處理
- 建立解析器版本測試流程

#### R-2: 大檔案效能風險
**風險**: 超大檔案的 AST 解析可能導致記憶體不足或處理緩慢
**影響**: 系統回應變慢或崩潰
**緩解措施**:
- 維持現有的檔案大小限制 (5MB)
- 實作記憶體監控和自動降級
- 大型專案推薦使用離線 `manual_indexing.py`

#### R-3: 語言覆蓋不足風險
**風險**: 少數程式語言可能缺乏 Tree-sitter 支援
**影響**: 這些語言的檔案無法進行智慧分塊
**緩解措施**:
- 優先支援主流語言 (80/20 法則)
- 不支援的語言自動降級到整檔案處理
- 建立語言支援優先級清單

### 中風險項目

#### R-4: 索引遷移風險
**風險**: 現有索引結構與新分塊格式不相容
**影響**: 用戶需要重新索引所有專案
**緩解措施**:
- 提供清晰的升級指南和自動化工具
- 設計向前相容的索引結構
- 實作索引健康檢查工具

#### R-5: 錯誤處理複雜性風險
**風險**: 語法錯誤處理邏輯過於複雜，可能引入新的 bug
**影響**: 系統穩定性和可維護性下降
**緩解措施**:
- 採用保守的錯誤處理策略
- 充分的單元測試和整合測試
- 錯誤統計和監控機制

### 低風險項目

#### R-6: 用戶接受度風險
**風險**: 用戶可能不滿意新的搜尋結果格式
**影響**: 用戶體驗下降
**緩解措施**:
- 基於明確的技術問題驅動改進
- 提供豐富的上下文資訊
- 持續收集用戶反饋並優化

## 10. Open Questions

### 技術實作問題

#### Q-1: 特殊語言的分塊策略
**問題**: 如何處理像 SQL、Shell Script 等特殊語言的分塊？
**考量**: 這些語言的語法結構與傳統程式語言差異較大
**建議方案**: Phase 2 研究，優先實作主流語言

#### Q-2: 動態語言的靜態分析限制
**問題**: Python、JavaScript 等動態語言的型別資訊較少，如何豐富元資料？
**考量**: 無法像 TypeScript、Java 那樣獲得豐富的型別資訊
**建議方案**: 基於命名慣例和文檔字串進行啟發式分析

#### Q-3: 嵌套函數和閉包處理
**問題**: JavaScript/Python 中的嵌套函數應該如何分塊？
**考量**: 嵌套函數與外層函數有上下文依賴
**建議方案**: 將嵌套函數包含在外層函數區塊中，保持語義完整性

### 效能和擴展性問題

#### Q-4: 解析結果快取策略
**問題**: AST 解析結果是否需要快取？快取策略如何設計？
**考量**: 解析速度 vs 記憶體使用 vs 快取一致性
**建議方案**: 基於檔案 hash 的智慧快取，避免快取 AST 物件本身

#### Q-5: 並行解析的執行緒安全
**問題**: Tree-sitter 解析器是否執行緒安全？如何設計並行解析？
**考量**: 多執行緒環境下的解析器共享和狀態管理
**建議方案**: 研究 Tree-sitter 的執行緒安全性，必要時使用解析器池

### 用戶體驗問題

#### Q-6: 搜尋結果排序策略
**問題**: 多個函數匹配時，如何智慧排序結果？
**考量**: 相關性評分、檔案重要性、使用頻率等因素
**建議方案**: 結合向量相似度和檔案結構資訊的混合排序算法

#### Q-7: 上下文展示的最佳長度
**問題**: 前後 5 行上下文是否足夠？是否需要動態調整？
**考量**: 不同程式語言和程式碼風格的差異
**建議方案**: 從固定 5 行開始，根據用戶反饋和使用資料調整

---

## 結論

智慧程式碼分塊系統將從根本上改變 Codebase RAG MCP Server 的程式碼理解和搜尋能力。通過基於 Tree-sitter 的語法感知分塊，我們可以實現：

1. **精確的函數級搜尋** - 使用者可以直接找到特定的函數或類別
2. **語義完整的程式碼單元** - 每個索引單元都是一個完整的語義實體
3. **強大的錯誤容忍能力** - 語法錯誤不會影響整體系統功能
4. **豐富的元資料支援** - 為未來的進階功能奠定基礎

這個實作將使 Codebase RAG MCP Server 從一個基礎的檔案搜尋工具，升級為一個真正理解程式碼結構的智慧助手。
