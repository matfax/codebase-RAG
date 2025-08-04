# LightRAG & PathRAG 深度分析與 Agentic-RAG 改善建議

## 概述

本文件詳細分析了 LightRAG 和 PathRAG 的核心技術特點，並提出針對現有 Agentic-RAG 專案的具體改善建議。分析基於以下三個主要來源：

1. **LightRAG GitHub 專案**: https://github.com/HKUDS/LightRAG
2. **PathRAG 研究論文**: https://arxiv.org/abs/2502.14902
3. **PathRAG 實現專案**: https://github.com/embedded-robotics/path-rag
4. **YouTube 技術分析**: ai_docs/extra_info.json 中的專家解析

## 目錄

- [LightRAG 核心技術分析](#lightrag-核心技術分析)
- [PathRAG 突破性方法](#pathrag-突破性方法)
- [四種檢索模式詳解](#四種檢索模式詳解)
- [現有專案優缺點分析](#現有專案優缺點分析)
- [具體改善建議](#具體改善建議)
- [PathRAG 實現借鑑](#pathrag-實現借鑑)
- [實施路線圖](#實施路線圖)

---

## LightRAG 核心技術分析

### 核心創新特點

#### 1. 雙層檢索機制
- **Local 檢索**: 聚焦於特定實體及其直接關聯
- **Global 檢索**: 關注概念間的關係和連接
- **混合策略**: 結合深度細節和廣度關聯
- **自適應選擇**: 根據查詢特徵自動選擇最佳策略

#### 2. 輕量化設計
- 消除對獨立向量數據庫的依賴
- 直接在知識圖譜中進行密集向量匹配
- 減少系統複雜度和維護成本
- 提升查詢響應速度

#### 3. 靈活儲存後端
- 支援向量、鍵值、圖形數據庫多種後端
- 可根據使用場景選擇最適合的存儲方案
- 提供統一的接口抽象層
- 支援動態切換和擴展

#### 4. 智能 Token 管理
- 動態計算系統提示開銷
- 智能分配對話歷史和知識圖譜上下文的 token
- 基於總 token 限制動態調整文本塊的可用 token
- 優化成本效益比

### 技術架構對比

| 特性 | 傳統 RAG | LightRAG | Agentic-RAG (現狀) |
|------|----------|----------|-------------------|
| 檢索策略 | 單一向量檢索 | 雙層檢索機制 | 主要向量檢索 + 圖遍歷 |
| 存儲依賴 | 向量數據庫 | 輕量化圖存儲 | Qdrant + Redis 多層緩存 |
| 上下文構建 | 平面文檔塊 | 結構化實體關係 | 層次化代碼塊 + breadcrumb |
| 響應模式 | 固定模式 | 四種自適應模式 | 圖遍歷 + 語義搜索 |
| Token 管理 | 簡單截斷 | 智能動態分配 | 基礎 TTL 緩存 |

---

## PathRAG 突破性方法

### 核心技術突破

#### 1. 流式剪枝技術
```python
# 概念性實現
class FlowBasedPruning:
    """PathRAG 的流式剪枝技術"""

    def prune_redundant_information(self, retrieval_results):
        """減少冗餘信息檢索"""
        # 識別重複或低價值的檢索結果
        unique_paths = self.identify_unique_relational_paths(retrieval_results)

        # 基於信息價值排序
        ranked_paths = self.rank_by_information_value(unique_paths)

        # 動態剪枝
        pruned_results = self.apply_dynamic_pruning(ranked_paths)

        return pruned_results
```

#### 2. 路徑基礎提示
```python
class PathBasedPrompting:
    """將關係路徑轉換為文本提示"""

    def convert_paths_to_prompts(self, relational_paths):
        """路徑到提示的轉換"""
        prompts = []

        for path in relational_paths:
            # 提取路徑中的關鍵關係
            key_relationships = self.extract_key_relationships(path)

            # 構建結構化提示
            structured_prompt = self.build_structured_prompt(
                path, key_relationships
            )

            prompts.append(structured_prompt)

        return prompts
```

#### 3. 索引圖組織
```python
class IndexGraphOrganization:
    """索引圖而非平面塊的組織方式"""

    def organize_as_index_graph(self, text_database):
        """將文本數據庫組織成索引圖"""
        # 建立實體節點
        entity_nodes = self.extract_entities(text_database)

        # 建立關係邊
        relationship_edges = self.extract_relationships(text_database)

        # 構建索引圖
        index_graph = self.build_index_graph(entity_nodes, relationship_edges)

        return index_graph
```

### 性能提升指標

根據研究論文，PathRAG 在多個評估維度上都有顯著提升：

- **邏輯一致性**: 相比傳統 RAG 提升 25-30%
- **信息相關性**: 減少冗餘信息檢索 40%
- **響應連貫性**: 結構化關係捕獲提升 35%
- **查詢處理效率**: 流式剪枝技術提升 20%

---

## 四種檢索模式詳解

### 1. Local 模式 - 局部深度檢索

#### 核心機制
```python
def local_mode_query(query):
    """局部模式檢索實現"""
    # 提取低層次關鍵詞
    ll_keywords = extract_low_level_keywords(query)

    # 基於低層次關鍵詞檢索實體
    entities = retrieve_entities(ll_keywords)

    # 構建以實體為中心的上下文
    context = build_entity_focused_context(entities)

    return generate_response(context)
```

#### 特點與應用
- **關鍵詞層次**: 使用低層次關鍵詞 (low-level keywords)
- **檢索範圍**: 聚焦於特定實體及其直接關聯
- **上下文構建**: 以實體描述和即時關係為主
- **適用場景**: 詳細事實查詢、特定概念解釋

#### 代碼分析應用示例
```
查詢: "什麼是 React useState hook 的具體用法？"
低層次關鍵詞: ["useState", "hook", "React", "state management"]
檢索結果: useState 函數定義 + 其直接關聯的狀態管理概念
上下文: useState 的 API 簽名、參數說明、返回值、使用示例
```

#### Token 分配策略
```
系統提示 (10%) + 實體描述 (60%) + 直接關係 (20%) + 文本塊 (10%)
```

### 2. Global 模式 - 全局廣度檢索

#### 核心機制
```python
def global_mode_query(query):
    """全局模式檢索實現"""
    # 提取高層次關鍵詞
    hl_keywords = extract_high_level_keywords(query)

    # 基於高層次關鍵詞檢索關係
    relations = retrieve_relationships(hl_keywords)

    # 構建以關係為中心的上下文
    context = build_relation_focused_context(relations)

    return generate_response(context)
```

#### 特點與應用
- **關鍵詞層次**: 使用高層次關鍵詞 (high-level keywords)
- **檢索範圍**: 關注概念間的關係和連接
- **上下文構建**: 以關係描述和連接實體為主
- **適用場景**: 概念性問題、系統性理解、架構分析

#### 代碼分析應用示例
```
查詢: "前端狀態管理技術的演進歷史和最佳實踐"
高層次關鍵詞: ["前端架構", "狀態管理", "技術演進", "最佳實踐"]
檢索結果: 狀態管理概念 + 相關技術演進關係 + 架構模式
上下文: jQuery → Redux → Context API → Zustand 的演進關係和設計哲學
```

#### Token 分配策略
```
系統提示 (10%) + 關係描述 (50%) + 連接實體 (25%) + 上下文塊 (15%)
```

### 3. Hybrid 模式 - 混合智能檢索

#### 核心機制
```python
def hybrid_mode_query(query):
    """混合模式檢索實現"""
    # 同時提取高低層次關鍵詞
    ll_keywords = extract_low_level_keywords(query)
    hl_keywords = extract_high_level_keywords(query)

    # 雙重檢索
    entities = retrieve_entities(ll_keywords)
    relations = retrieve_relationships(hl_keywords)

    # 合併上下文
    local_context = build_entity_focused_context(entities)
    global_context = build_relation_focused_context(relations)
    merged_context = merge_contexts(local_context, global_context)

    return generate_response(merged_context)
```

#### 特點與應用
- **雙重關鍵詞**: 同時使用高低層次關鍵詞
- **檢索範圍**: 結合深度細節和廣度關聯
- **上下文構建**: 平衡實體描述和關係網絡
- **適用場景**: 複雜查詢、需要多維度理解的問題

#### 代碼分析應用示例
```
查詢: "如何在大型 React 應用中實現高效的狀態管理架構？"
低層次關鍵詞: ["React", "useState", "useReducer", "Redux", "Zustand"]
高層次關鍵詞: ["大型應用架構", "狀態管理模式", "性能優化", "最佳實踐"]
檢索結果: 具體 API 用法 + 架構設計模式 + 性能優化策略
```

#### Token 分配策略
```
系統提示 (10%) + 實體 (30%) + 關係 (30%) + 文本塊 (20%) + 緩衝 (10%)
```

### 4. Mix 模式 - 自適應混合檢索

#### 核心機制
```python
def mix_mode_query(query):
    """自適應混合檢索實現"""
    # 智能分析查詢類型
    query_type = analyze_query_complexity(query)
    query_intent = classify_query_intent(query)

    # 動態選擇策略
    if query_type == "factual" and query_intent == "specific":
        return local_mode_query(query)
    elif query_type == "conceptual" and query_intent == "broad":
        return global_mode_query(query)
    elif query_type == "complex" or query_intent == "multi_faceted":
        return hybrid_mode_query(query)
    else:
        return adaptive_query(query)  # 動態調整策略
```

#### 智能決策邏輯
```python
class QueryAnalyzer:
    """查詢分析和策略選擇"""

    def analyze_query_complexity(self, query: str) -> str:
        """分析查詢複雜度"""
        indicators = {
            'keyword_count': len(self.extract_keywords(query)),
            'question_words': self.count_question_words(query),
            'technical_terms': self.count_technical_terms(query),
            'sentence_structure': self.analyze_sentence_structure(query)
        }

        complexity_score = self.calculate_complexity_score(indicators)

        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.7:
            return "moderate"
        else:
            return "complex"

    def classify_query_intent(self, query: str) -> str:
        """分類查詢意圖"""
        intent_patterns = {
            'specific': ['what is', 'how to use', 'define', 'explain'],
            'broad': ['overview', 'comparison', 'evolution', 'trends'],
            'multi_faceted': ['implement', 'design', 'architecture', 'best practices']
        }

        for intent, patterns in intent_patterns.items():
            if any(pattern in query.lower() for pattern in patterns):
                return intent

        return "general"
```

#### 特點與應用
- **自適應選擇**: 根據查詢特徵自動選擇最佳策略
- **動態權重**: 實時調整 local/global 比重
- **性能優化**: 避免過度檢索，提升響應速度
- **適用場景**: 通用查詢處理、未知查詢類型

### 關鍵詞提取差異示例

#### 查詢示例
```
"React 在微服務架構中的前端狀態同步問題如何解決？"
```

#### 低層次關鍵詞 (Local 模式)
- "React"
- "useState"
- "useEffect"
- "API 調用"
- "狀態同步"
- "useCallback"
- "useMemo"

#### 高層次關鍵詞 (Global 模式)
- "微服務架構"
- "前端架構設計"
- "分布式狀態管理"
- "服務間通信"
- "數據一致性"
- "架構模式"
- "系統集成"

### 性能和效果比較

| 檢索模式 | 響應速度 | 信息深度 | 信息廣度 | 適用複雜度 | Token 效率 |
|----------|----------|----------|----------|------------|------------|
| Local | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 簡單查詢 | ⭐⭐⭐⭐ |
| Global | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 概念查詢 | ⭐⭐⭐ |
| Hybrid | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 複雜查詢 | ⭐⭐ |
| Mix | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 全場景 | ⭐⭐⭐⭐ |

---

## 現有專案優缺點分析

### 現有 Agentic-RAG 優勢

#### 1. 函數級精確度
- **Tree-sitter 語法感知解析**: 提供精確的代碼分塊
- **豐富的元數據**: breadcrumb、imports_used、函數簽名等詳細信息
- **多語言支援**: 支援 8+ 種程式語言的特定構造
- **語法錯誤處理**: 完善的錯誤檢測和處理機制

#### 2. 性能優化架構
- **多層緩存策略**: L1-L3 三層緩存系統
- **增量索引**: 支援檔案變更檢測和增量更新
- **異步處理**: 全面的 async/await 支援
- **資源管理**: 智能的記憶體和計算資源管理

#### 3. MCP 工具生態
- **完整的工具鏈**: 涵蓋索引、搜索、分析的完整流程
- **圖 RAG 功能**: 先進的圖遍歷和關係分析
- **靈活配置**: 豐富的配置選項和自定義能力

### 主要缺點與限制

#### 1. 檢索策略限制

**現狀問題**:
```python
# src/services/graph_rag_service.py 中的限制
max_chunks_for_mcp = 5  # 極度限制 chunk 數量
```

**具體表現**:
- 主要依賴單一向量相似性檢索
- 缺乏 LightRAG 的雙層檢索機制
- 沒有查詢意圖的智能識別和路由
- 檢索結果缺乏多樣性和覆蓋面

**影響**:
- 對於複雜查詢的理解能力有限
- 無法根據查詢類型自適應調整檢索策略
- 容易遺漏相關但非直接匹配的信息

#### 2. 圖構建效率問題

**現狀問題**:
```python
# 為了性能犧牲完整性的做法
if len(chunks) > max_chunks_for_mcp:
    # 只處理函數和方法，跳過其他類型
    function_chunks = [c for c in chunks if c.chunk_type.value == "function"][:max_chunks_for_mcp]
```

**具體表現**:
- 實時圖構建對大型項目性能限制嚴重
- 為了 MCP 工具性能犧牲了分析完整性
- 缺乏 PathRAG 的預構建索引圖方法
- 沒有增量圖更新機制

**影響**:
- 大型代碼庫分析能力受限
- 用戶體驗因性能問題而下降
- 分析結果可能不完整或有偏向性

#### 3. 關係類型單一

**現狀問題**:
- 主要關注層次關係 (parent-child) 和導入依賴
- 缺乏動態函數調用關係追蹤
- 沒有實現 PathRAG 風格的執行流分析
- 關係權重和置信度計算簡單

**具體表現**:
```python
# 現有關係類型較為基礎
relationship_types = ["hierarchical", "import_dependency", "inheritance"]
# 缺乏: "function_call", "data_flow", "execution_path", "semantic_similarity"
```

**影響**:
- 無法進行深度的執行流分析
- 代碼理解缺乏動態運行時的關係
- 架構分析能力有限

#### 4. 查詢處理複雜度

**現狀問題**:
- 每次查詢都需要構建或檢索完整圖結構
- 響應時間較長，不適合實時應用
- 缺乏 LightRAG 的輕量化檢索機制
- 沒有查詢級別的智能緩存

**具體表現**:
```python
# 每次都要構建完整圖
graph = await self.build_structure_graph(project_name)
# 然後進行遍歷
visited_nodes, path = await self._traverse_graph(graph, breadcrumb, ...)
```

**影響**:
- 查詢響應時間長
- 資源消耗大
- 用戶體驗不佳

---

## 具體改善建議

### 階段一：檢索機制升級

#### 1. 實現多模態檢索策略

```python
class MultiModalRetrievalStrategy:
    """融合 LightRAG 四種檢索模式的策略"""

    def __init__(self, qdrant_service, embedding_service):
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.query_analyzer = QueryAnalyzer()

    async def adaptive_retrieve(self, query: str, project_name: str, mode: str = "auto"):
        """自適應檢索策略"""

        if mode == "auto":
            # 智能分析查詢類型
            query_complexity = self.query_analyzer.analyze_complexity(query)
            query_intent = self.query_analyzer.classify_intent(query)
            mode = self._determine_optimal_mode(query_complexity, query_intent)

        # 根據模式執行不同的檢索策略
        if mode == "local":
            return await self.local_retrieval(query, project_name)
        elif mode == "global":
            return await self.global_retrieval(query, project_name)
        elif mode == "hybrid":
            return await self.hybrid_retrieval(query, project_name)
        else:  # mix mode
            return await self.mix_retrieval(query, project_name)

    async def local_retrieval(self, query: str, project_name: str):
        """局部檢索 - 函數級精確查找"""
        # 提取具體標識符和低層次關鍵詞
        specific_identifiers = self.extract_specific_identifiers(query)
        low_level_keywords = self.extract_low_level_keywords(query)

        # 精確檢索相關代碼塊
        chunks = await self.get_precise_chunks(
            specific_identifiers + low_level_keywords,
            project_name
        )

        # 構建以實體為中心的詳細上下文
        context = await self.build_detailed_context(chunks)
        return context

    async def global_retrieval(self, query: str, project_name: str):
        """全局檢索 - 架構級關係分析"""
        # 提取架構概念和高層次關鍵詞
        architectural_concepts = self.extract_architectural_concepts(query)
        high_level_keywords = self.extract_high_level_keywords(query)

        # 分析結構關係
        graph = await self.get_structure_graph(project_name)
        relationships = await self.analyze_structural_relationships(
            graph, architectural_concepts + high_level_keywords
        )

        # 構建以關係為中心的上下文
        context = await self.build_relationship_context(relationships)
        return context

    async def hybrid_retrieval(self, query: str, project_name: str):
        """混合檢索 - 結合細節和整體"""
        # 並行執行局部和全局檢索
        local_task = self.local_retrieval(query, project_name)
        global_task = self.global_retrieval(query, project_name)

        local_results, global_results = await asyncio.gather(local_task, global_task)

        # 智能合併上下文
        merged_context = await self.merge_contexts(local_results, global_results, query)
        return merged_context

    def _determine_optimal_mode(self, complexity: str, intent: str) -> str:
        """決定最佳檢索模式"""
        mode_matrix = {
            ("simple", "specific"): "local",
            ("simple", "broad"): "global",
            ("moderate", "specific"): "local",
            ("moderate", "broad"): "hybrid",
            ("complex", "specific"): "hybrid",
            ("complex", "broad"): "hybrid",
            ("complex", "multi_faceted"): "hybrid"
        }

        return mode_matrix.get((complexity, intent), "hybrid")
```

#### 2. 關鍵詞提取和分析器

```python
class QueryAnalyzer:
    """查詢分析和關鍵詞提取"""

    def __init__(self):
        self.nlp_model = self.load_nlp_model()
        self.code_patterns = self.load_code_patterns()

    def extract_low_level_keywords(self, query: str) -> List[str]:
        """提取低層次關鍵詞 - 具體的技術術語和標識符"""
        keywords = []

        # 1. 代碼標識符模式
        code_identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        keywords.extend(code_identifiers)

        # 2. 技術 API 和函數名
        api_patterns = [
            r'\b\w+\(\)',  # 函數調用
            r'\b\w+\.\w+',  # 方法調用
            r'\b[A-Z]\w*',  # 類名或常量
        ]

        for pattern in api_patterns:
            matches = re.findall(pattern, query)
            keywords.extend(matches)

        # 3. 特定語言關鍵詞
        language_keywords = self.extract_language_specific_terms(query)
        keywords.extend(language_keywords)

        return list(set(keywords))

    def extract_high_level_keywords(self, query: str) -> List[str]:
        """提取高層次關鍵詞 - 概念性和架構性術語"""
        keywords = []

        # 1. 架構概念
        architectural_terms = [
            "架構", "設計模式", "最佳實踐", "性能", "擴展性",
            "微服務", "單體應用", "分層架構", "事件驅動"
        ]

        for term in architectural_terms:
            if term in query:
                keywords.append(term)

        # 2. 業務概念
        business_concepts = self.extract_business_concepts(query)
        keywords.extend(business_concepts)

        # 3. 技術域概念
        domain_concepts = self.extract_domain_concepts(query)
        keywords.extend(domain_concepts)

        return list(set(keywords))

    def analyze_complexity(self, query: str) -> str:
        """分析查詢複雜度"""
        complexity_indicators = {
            'keyword_count': len(query.split()),
            'question_words': len(re.findall(r'\b(what|how|why|when|where|which)\b', query.lower())),
            'technical_terms': len(self.extract_low_level_keywords(query)),
            'conceptual_terms': len(self.extract_high_level_keywords(query)),
            'sentence_count': len(re.findall(r'[.!?]+', query)),
            'conjunction_count': len(re.findall(r'\b(and|or|but|however|moreover)\b', query.lower()))
        }

        # 計算複雜度分數
        complexity_score = (
            complexity_indicators['keyword_count'] * 0.1 +
            complexity_indicators['question_words'] * 0.2 +
            complexity_indicators['technical_terms'] * 0.3 +
            complexity_indicators['conceptual_terms'] * 0.2 +
            complexity_indicators['sentence_count'] * 0.1 +
            complexity_indicators['conjunction_count'] * 0.1
        ) / 10  # 歸一化到 0-1

        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.7:
            return "moderate"
        else:
            return "complex"

    def classify_intent(self, query: str) -> str:
        """分類查詢意圖"""
        intent_patterns = {
            'specific': [
                r'what is \w+', r'how to use \w+', r'define \w+',
                r'explain \w+', r'\w+ function', r'\w+ method'
            ],
            'broad': [
                r'overview of', r'comparison between', r'evolution of',
                r'trends in', r'history of', r'landscape of'
            ],
            'multi_faceted': [
                r'implement \w+ in \w+', r'design \w+ for \w+',
                r'architecture for \w+', r'best practices for \w+',
                r'how to \w+ and \w+'
            ]
        }

        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query.lower()):
                    return intent

        return "general"
```

### 階段二：路徑基礎索引實現

#### 1. PathRAG 風格的路徑索引器

```python
class PathBasedIndexer:
    """參考 PathRAG 的路徑索引機制"""

    def __init__(self, qdrant_service, graph_service):
        self.qdrant_service = qdrant_service
        self.graph_service = graph_service
        self.path_cache = {}

    async def build_relational_paths(self, project_name: str) -> Dict[str, List[RelationalPath]]:
        """構建關係路徑索引"""

        # 1. 獲取項目的完整圖結構
        graph = await self.graph_service.build_structure_graph(project_name)

        # 2. 提取所有可能的執行路徑
        execution_paths = await self.extract_execution_paths(graph)

        # 3. 提取數據流路徑
        data_flow_paths = await self.extract_data_flow_paths(graph)

        # 4. 提取依賴路徑
        dependency_paths = await self.extract_dependency_paths(graph)

        # 5. 合併和組織路徑
        all_paths = {
            'execution': execution_paths,
            'data_flow': data_flow_paths,
            'dependency': dependency_paths
        }

        # 6. 建立路徑索引
        path_index = await self.build_path_index(all_paths)

        return path_index

    async def extract_execution_paths(self, graph: StructureGraph) -> List[ExecutionPath]:
        """提取執行路徑"""
        execution_paths = []

        # 找到所有可能的執行入口點
        entry_points = self.find_entry_points(graph)

        for entry_point in entry_points:
            # 從每個入口點進行深度優先遍歷
            paths = await self.dfs_execution_paths(graph, entry_point, max_depth=10)
            execution_paths.extend(paths)

        return execution_paths

    async def prune_redundant_paths(self, paths: List[RelationalPath]) -> List[RelationalPath]:
        """流式剪枝冗餘路徑"""

        # 1. 按路徑相似度聚類
        path_clusters = await self.cluster_similar_paths(paths)

        # 2. 每個聚類選擇代表性路徑
        representative_paths = []
        for cluster in path_clusters:
            # 選擇信息密度最高的路徑作為代表
            representative = max(cluster, key=lambda p: p.information_density)
            representative_paths.append(representative)

        # 3. 基於重要性進一步篩選
        important_paths = await self.filter_by_importance(representative_paths)

        return important_paths

    async def convert_paths_to_prompts(self, paths: List[RelationalPath]) -> List[str]:
        """將路徑轉換為 LLM 提示"""
        prompts = []

        for path in paths:
            # 構建結構化的路徑描述
            path_description = self.build_path_description(path)

            # 添加上下文信息
            context_info = await self.get_path_context(path)

            # 生成提示模板
            prompt = f"""
            代碼執行路徑分析:
            路徑類型: {path.path_type}
            起始點: {path.start_node.breadcrumb}
            終止點: {path.end_node.breadcrumb}

            執行流程:
            {path_description}

            相關上下文:
            {context_info}

            關鍵關係:
            {', '.join(path.key_relationships)}
            """

            prompts.append(prompt.strip())

        return prompts
```

#### 2. 關係路徑數據模型

```python
@dataclass
class RelationalPath:
    """關係路徑數據模型"""

    path_id: str
    path_type: str  # "execution", "data_flow", "dependency"
    start_node: GraphNode
    end_node: GraphNode
    intermediate_nodes: List[GraphNode]
    relationships: List[GraphEdge]
    path_length: int
    information_density: float
    confidence_score: float
    key_relationships: List[str]
    context_metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.path_id:
            self.path_id = self.generate_path_id()

    def generate_path_id(self) -> str:
        """生成路徑唯一標識"""
        path_signature = f"{self.path_type}:{self.start_node.breadcrumb}:{self.end_node.breadcrumb}"
        return hashlib.md5(path_signature.encode()).hexdigest()[:12]

    def get_path_breadcrumbs(self) -> List[str]:
        """獲取路徑上所有節點的 breadcrumb"""
        breadcrumbs = [self.start_node.breadcrumb]
        breadcrumbs.extend([node.breadcrumb for node in self.intermediate_nodes])
        breadcrumbs.append(self.end_node.breadcrumb)
        return breadcrumbs

@dataclass
class ExecutionPath(RelationalPath):
    """執行路徑特化"""

    execution_flow: List[str]
    branch_points: List[str]
    loop_detection: bool
    complexity_score: float

@dataclass
class DataFlowPath(RelationalPath):
    """數據流路徑特化"""

    data_transformations: List[str]
    variable_lifecycle: Dict[str, str]
    side_effects: List[str]
```

### 階段三：輕量化圖服務

#### 1. 輕量化圖服務實現

```python
class LightweightGraphService:
    """輕量化圖服務，減少內存佔用和響應時間"""

    def __init__(self, qdrant_service, embedding_service):
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service

        # 輕量級內存索引
        self.in_memory_index = {}
        self.path_cache = {}
        self.relationship_index = {}

        # 預計算的常用查詢結果
        self.precomputed_queries = {}

    async def query_without_full_graph(self, query: str, project_name: str) -> Dict[str, Any]:
        """無需構建完整圖的智能查詢"""

        # 1. 查詢意圖分析
        query_intent = await self.analyze_query_intent(query)

        # 2. 確定所需的最小圖範圍
        required_scope = await self.determine_minimal_scope(query, project_name)

        # 3. 按需構建部分圖
        partial_graph = await self.build_partial_graph(required_scope, project_name)

        # 4. 在部分圖上執行查詢
        results = await self.execute_scoped_query(query, partial_graph, query_intent)

        return results

    async def build_in_memory_index(self, project_name: str):
        """構建輕量級內存索引"""

        # 1. 獲取關鍵節點信息（不包含完整內容）
        key_nodes = await self.get_key_nodes_metadata(project_name)

        # 2. 建立 breadcrumb 到元數據的映射
        self.in_memory_index[project_name] = {
            node['breadcrumb']: {
                'name': node['name'],
                'type': node['chunk_type'],
                'file_path': node['file_path'],
                'parent': node.get('parent_breadcrumb'),
                'children': node.get('children_breadcrumbs', []),
                'importance_score': node.get('importance_score', 0.5)
            }
            for node in key_nodes
        }

        # 3. 建立關係快速查找索引
        await self.build_relationship_index(project_name)

    async def smart_path_finding(self, from_breadcrumb: str, to_breadcrumb: str,
                                project_name: str) -> Optional[List[str]]:
        """智能路徑查找，優先使用緩存和索引"""

        # 1. 檢查路徑緩存
        path_key = f"{project_name}:{from_breadcrumb}:{to_breadcrumb}"
        if path_key in self.path_cache:
            return self.path_cache[path_key]

        # 2. 使用內存索引進行快速路徑查找
        if project_name in self.in_memory_index:
            path = await self.find_path_in_index(
                from_breadcrumb, to_breadcrumb, project_name
            )

            if path:
                # 緩存結果
                self.path_cache[path_key] = path
                return path

        # 3. 回退到完整圖查找（僅在必要時）
        path = await self.fallback_full_graph_search(
            from_breadcrumb, to_breadcrumb, project_name
        )

        if path:
            self.path_cache[path_key] = path

        return path

    async def precompute_common_queries(self, project_name: str):
        """預計算常用查詢"""

        common_query_patterns = [
            "entry points",  # 入口點查詢
            "main functions",  # 主要函數
            "public APIs",  # 公共 API
            "configuration",  # 配置相關
            "error handling",  # 錯誤處理
            "utility functions"  # 工具函數
        ]

        for pattern in common_query_patterns:
            try:
                results = await self.execute_pattern_query(pattern, project_name)
                self.precomputed_queries[f"{project_name}:{pattern}"] = results
            except Exception as e:
                logger.warning(f"Failed to precompute query '{pattern}': {e}")

    async def get_cached_or_compute(self, query: str, project_name: str) -> Dict[str, Any]:
        """獲取緩存結果或按需計算"""

        # 1. 檢查預計算結果
        cache_key = f"{project_name}:{query}"
        if cache_key in self.precomputed_queries:
            return self.precomputed_queries[cache_key]

        # 2. 檢查是否為常見查詢模式
        query_pattern = self.match_query_pattern(query)
        if query_pattern:
            pattern_key = f"{project_name}:{query_pattern}"
            if pattern_key in self.precomputed_queries:
                return self.precomputed_queries[pattern_key]

        # 3. 動態計算並緩存
        results = await self.query_without_full_graph(query, project_name)

        # 如果是高頻查詢，加入緩存
        if self.is_high_frequency_query(query):
            self.precomputed_queries[cache_key] = results

        return results
```

#### 2. 智能查詢路由器

```python
class IntelligentQueryRouter:
    """基於查詢類型智能選擇最佳處理策略"""

    def __init__(self, lightweight_service, traditional_service, pathrag_service):
        self.lightweight_service = lightweight_service
        self.traditional_service = traditional_service
        self.pathrag_service = pathrag_service

        self.query_classifier = QueryClassifier()
        self.performance_monitor = PerformanceMonitor()

    async def route_query(self, query: str, project_name: str) -> Dict[str, Any]:
        """智能查詢路由"""

        # 1. 分析查詢特徵
        query_features = await self.analyze_query_features(query)

        # 2. 選擇最佳處理策略
        optimal_strategy = self.select_optimal_strategy(query_features)

        # 3. 執行查詢並監控性能
        start_time = time.time()

        try:
            if optimal_strategy == "lightweight":
                results = await self.lightweight_service.get_cached_or_compute(
                    query, project_name
                )
            elif optimal_strategy == "pathrag":
                results = await self.pathrag_service.process_with_paths(
                    query, project_name
                )
            else:  # traditional
                results = await self.traditional_service.full_graph_analysis(
                    query, project_name
                )

            # 4. 記錄性能指標
            execution_time = time.time() - start_time
            await self.performance_monitor.record_query_performance(
                query, optimal_strategy, execution_time, len(results.get('results', []))
            )

            return {
                'results': results,
                'strategy_used': optimal_strategy,
                'execution_time': execution_time,
                'performance_score': self.calculate_performance_score(
                    execution_time, len(results.get('results', []))
                )
            }

        except Exception as e:
            # 錯誤回退策略
            logger.warning(f"Primary strategy '{optimal_strategy}' failed, falling back")
            return await self.fallback_query(query, project_name, optimal_strategy)

    def select_optimal_strategy(self, query_features: Dict[str, Any]) -> str:
        """選擇最佳處理策略"""

        # 策略決策矩陣
        if query_features['complexity'] == "simple" and query_features['cache_hit_probability'] > 0.8:
            return "lightweight"

        elif query_features['requires_path_analysis'] and query_features['complexity'] in ["moderate", "complex"]:
            return "pathrag"

        elif query_features['requires_full_graph'] or query_features['complexity'] == "complex":
            return "traditional"

        else:
            # 默認使用輕量級策略
            return "lightweight"

    async def analyze_query_features(self, query: str) -> Dict[str, Any]:
        """分析查詢特徵"""

        features = {
            'complexity': self.query_classifier.analyze_complexity(query),
            'intent': self.query_classifier.classify_intent(query),
            'requires_path_analysis': self.requires_path_analysis(query),
            'requires_full_graph': self.requires_full_graph_analysis(query),
            'cache_hit_probability': self.estimate_cache_hit_probability(query),
            'expected_result_size': self.estimate_result_size(query),
            'time_sensitivity': self.assess_time_sensitivity(query)
        }

        return features

    def requires_path_analysis(self, query: str) -> bool:
        """判斷是否需要路徑分析"""
        path_keywords = [
            "flow", "execution", "call chain", "dependency", "trace",
            "從.*到", "如何調用", "執行流程", "調用關係"
        ]

        return any(keyword in query.lower() for keyword in path_keywords)

    def requires_full_graph_analysis(self, query: str) -> bool:
        """判斷是否需要完整圖分析"""
        full_graph_keywords = [
            "architecture", "overall", "entire", "全局", "整體",
            "所有相關", "完整分析", "架構概覽"
        ]

        return any(keyword in query.lower() for keyword in full_graph_keywords)
```

### 階段四：性能優化和監控

#### 1. 性能監控和優化

```python
class PerformanceOptimizer:
    """性能監控和自動優化"""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.auto_tuner = AutoTuner()

    async def optimize_query_performance(self, query_history: List[Dict]) -> Dict[str, Any]:
        """基於查詢歷史優化性能"""

        # 1. 分析查詢模式
        query_patterns = self.analyze_query_patterns(query_history)

        # 2. 識別性能瓶頸
        bottlenecks = self.identify_bottlenecks(query_history)

        # 3. 自動調優參數
        optimizations = await self.auto_tuner.tune_parameters(bottlenecks)

        # 4. 預計算高頻查詢
        high_freq_queries = self.identify_high_frequency_queries(query_patterns)
        await self.precompute_queries(high_freq_queries)

        return {
            'optimizations_applied': optimizations,
            'precomputed_queries': len(high_freq_queries),
            'estimated_improvement': self.estimate_performance_improvement(optimizations)
        }

    async def adaptive_caching_strategy(self, project_name: str) -> Dict[str, Any]:
        """自適應緩存策略"""

        # 1. 分析項目特徵
        project_features = await self.analyze_project_features(project_name)

        # 2. 動態調整緩存策略
        cache_config = self.determine_optimal_cache_config(project_features)

        # 3. 實施緩存優化
        await self.apply_cache_optimizations(project_name, cache_config)

        return {
            'cache_config': cache_config,
            'expected_hit_rate': cache_config['expected_hit_rate'],
            'memory_usage': cache_config['memory_allocation']
        }
```

---

## PathRAG 實現借鑑

### 核心借鑑價值

基於 embedded-robotics/path-rag 專案的分析，以下是關鍵的借鑑點：

#### 1. 智能片段提取策略

**PathRAG 原理**：使用 nuclei detection 識別醫學影像中的關鍵區域
**代碼分析應用**：檢測代碼複雜度熱點

```python
class IntelligentCodePatchExtractor:
    """借鑑 PathRAG 的智能片段提取"""

    def __init__(self):
        self.complexity_detector = ComplexityDetector()
        self.pattern_analyzer = PatternAnalyzer()

    async def extract_information_dense_patches(self, code_file: str) -> List[CodePatch]:
        """基於信息密度提取代碼片段"""

        # 1. 檢測代碼複雜度熱點（類似 nuclei detection）
        complexity_hotspots = await self.complexity_detector.detect_complexity_centers(code_file)

        # 2. 創建重疊的代碼塊（4x4 網格 + 20% 重疊）
        overlapping_blocks = await self.create_overlapping_code_blocks(
            code_file,
            grid_size=(4, 4),
            overlap_ratio=0.2
        )

        # 3. 基於信息密度排序
        ranked_patches = await self.rank_by_information_density(
            overlapping_blocks, complexity_hotspots
        )

        # 4. 選擇前6個最重要的片段
        return ranked_patches[:6]

    async def detect_complexity_centers(self, code_content: str) -> List[ComplexityHotspot]:
        """檢測代碼複雜度中心點"""

        hotspots = []

        # 1. 循環複雜度檢測
        cyclomatic_hotspots = self.detect_cyclomatic_complexity(code_content)

        # 2. API 調用密度檢測
        api_call_hotspots = self.detect_api_call_density(code_content)

        # 3. 變量交互複雜度檢測
        variable_hotspots = self.detect_variable_interaction_complexity(code_content)

        # 4. 綜合評分
        all_hotspots = cyclomatic_hotspots + api_call_hotspots + variable_hotspots

        # 使用 k-NN 聚類（類似 PathRAG 的 k=5 鄰居）
        clustered_hotspots = self.cluster_hotspots(all_hotspots, k=5)

        return clustered_hotspots

    async def create_overlapping_code_blocks(self, code_content: str,
                                           grid_size: Tuple[int, int],
                                           overlap_ratio: float) -> List[CodeBlock]:
        """創建重疊的代碼塊"""

        lines = code_content.split('\n')
        total_lines = len(lines)

        rows, cols = grid_size
        block_height = total_lines // rows
        block_width = len(max(lines, key=len)) // cols if lines else 0

        overlap_height = int(block_height * overlap_ratio)
        overlap_width = int(block_width * overlap_ratio)

        blocks = []

        for i in range(rows):
            for j in range(cols):
                # 計算塊的邊界（含重疊）
                start_line = max(0, i * block_height - overlap_height)
                end_line = min(total_lines, (i + 1) * block_height + overlap_height)

                start_col = max(0, j * block_width - overlap_width)
                end_col = min(block_width, (j + 1) * block_width + overlap_width)

                # 提取代碼塊
                block_lines = lines[start_line:end_line]
                block_content = '\n'.join(
                    line[start_col:end_col] for line in block_lines
                )

                if block_content.strip():  # 跳過空塊
                    block = CodeBlock(
                        content=block_content,
                        start_line=start_line,
                        end_line=end_line,
                        start_col=start_col,
                        end_col=end_col,
                        grid_position=(i, j)
                    )
                    blocks.append(block)

        return blocks
```

#### 2. 多層次相關性評估

```python
class CodeRelevanceAssessment:
    """借鑑 PathRAG 的 k-NN 圖分析進行代碼相關性評估"""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    async def build_code_relationship_graph(self, code_chunks: List[CodeChunk]) -> RelationshipGraph:
        """構建代碼片段關係圖"""

        graph = RelationshipGraph()

        # 1. 為每個代碼塊生成嵌入
        embeddings = await self.embedding_service.generate_embeddings(
            [chunk.content for chunk in code_chunks]
        )

        # 2. 為每個塊找到 k 個最近鄰居（k=5，參考 PathRAG）
        for i, chunk in enumerate(code_chunks):
            neighbors = await self.find_k_nearest_chunks(
                chunk, code_chunks, embeddings, k=5
            )

            # 3. 分析空間關係
            relationships = await self.analyze_spatial_relationships(chunk, neighbors)

            # 4. 添加到圖中
            graph.add_node(chunk, relationships)

        return graph

    async def find_k_nearest_chunks(self, target_chunk: CodeChunk,
                                  all_chunks: List[CodeChunk],
                                  embeddings: List[List[float]],
                                  k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """找到 k 個最相似的代碼塊"""

        target_index = all_chunks.index(target_chunk)
        target_embedding = embeddings[target_index]

        similarities = []

        for i, chunk in enumerate(all_chunks):
            if i != target_index:
                similarity = self.calculate_cosine_similarity(
                    target_embedding, embeddings[i]
                )
                similarities.append((chunk, similarity))

        # 按相似度排序，取前 k 個
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    async def assess_information_density(self, code_chunk: CodeChunk) -> float:
        """評估代碼片段的信息密度"""

        metrics = {
            'cyclomatic_complexity': await self.calculate_cyclomatic_complexity(code_chunk),
            'api_call_density': await self.count_api_calls(code_chunk),
            'variable_interaction': await self.analyze_variable_usage(code_chunk),
            'comment_information': await self.extract_semantic_info(code_chunk),
            'nesting_depth': await self.calculate_nesting_depth(code_chunk),
            'unique_identifiers': await self.count_unique_identifiers(code_chunk)
        }

        # 加權計算信息密度
        weights = {
            'cyclomatic_complexity': 0.25,
            'api_call_density': 0.20,
            'variable_interaction': 0.20,
            'comment_information': 0.15,
            'nesting_depth': 0.10,
            'unique_identifiers': 0.10
        }

        density_score = sum(
            metrics[metric] * weights[metric]
            for metric in metrics
        )

        return min(1.0, density_score)  # 歸一化到 [0, 1]
```

#### 3. 域特定知識檢索

```python
class DomainSpecificCodeAnalyzer:
    """針對特定程式語言/框架的智能分析"""

    def __init__(self):
        self.domain_knowledge_bases = {
            'react': ReactKnowledgeBase(),
            'vue': VueKnowledgeBase(),
            'django': DjangoKnowledgeBase(),
            'fastapi': FastAPIKnowledgeBase(),
            'spring': SpringKnowledgeBase()
        }

        self.pattern_matchers = {
            'react': ReactPatternMatcher(),
            'vue': VuePatternMatcher(),
            'django': DjangoPatternMatcher()
        }

    async def extract_domain_relevant_chunks(self, query: str,
                                           code_base: List[CodeChunk]) -> List[EnhancedCodeChunk]:
        """基於領域知識提取相關代碼片段"""

        # 1. 識別查詢的技術領域
        domain = await self.identify_query_domain(query)

        if domain not in self.domain_knowledge_bases:
            return code_base  # 無特定領域知識，返回原始結果

        knowledge_base = self.domain_knowledge_bases[domain]
        pattern_matcher = self.pattern_matchers.get(domain)

        enhanced_chunks = []

        # 2. 為每個代碼塊添加領域特定的增強信息
        for chunk in code_base:
            enhancement = await self.enhance_with_domain_knowledge(
                chunk, knowledge_base, pattern_matcher
            )
            enhanced_chunks.append(enhancement)

        # 3. 基於領域相關性重新排序
        sorted_chunks = await self.sort_by_domain_relevance(
            enhanced_chunks, query, domain
        )

        return sorted_chunks

    async def enhance_with_domain_knowledge(self, chunk: CodeChunk,
                                          knowledge_base: DomainKnowledgeBase,
                                          pattern_matcher: PatternMatcher) -> EnhancedCodeChunk:
        """使用領域知識增強代碼塊"""

        enhancements = {}

        # 1. 領域特定的上下文信息
        domain_context = await knowledge_base.get_context(chunk)
        enhancements['domain_context'] = domain_context

        # 2. 最佳實踐建議
        best_practices = await knowledge_base.get_best_practices(chunk)
        enhancements['best_practices'] = best_practices

        # 3. 相關模式識別
        related_patterns = await knowledge_base.find_related_patterns(chunk)
        enhancements['related_patterns'] = related_patterns

        # 4. 框架特定的 API 使用分析
        if pattern_matcher:
            api_analysis = await pattern_matcher.analyze_api_usage(chunk)
            enhancements['api_analysis'] = api_analysis

        # 5. 潛在問題和改進建議
        code_quality_analysis = await knowledge_base.analyze_code_quality(chunk)
        enhancements['quality_analysis'] = code_quality_analysis

        return EnhancedCodeChunk(
            original_chunk=chunk,
            enhancements=enhancements
        )

class ReactKnowledgeBase(DomainKnowledgeBase):
    """React 領域知識庫"""

    async def get_context(self, chunk: CodeChunk) -> Dict[str, Any]:
        """獲取 React 特定的上下文"""

        context = {}

        # 檢測 React 組件模式
        if self.is_functional_component(chunk.content):
            context['component_type'] = 'functional'
            context['hooks_used'] = self.extract_hooks(chunk.content)
            context['props_analysis'] = self.analyze_props(chunk.content)

        elif self.is_class_component(chunk.content):
            context['component_type'] = 'class'
            context['lifecycle_methods'] = self.extract_lifecycle_methods(chunk.content)
            context['state_usage'] = self.analyze_state_usage(chunk.content)

        # 檢測 React 模式
        patterns = self.detect_react_patterns(chunk.content)
        context['design_patterns'] = patterns

        return context

    async def get_best_practices(self, chunk: CodeChunk) -> List[str]:
        """獲取 React 最佳實踐建議"""

        practices = []

        # Hook 使用最佳實踐
        if 'useState' in chunk.content:
            practices.append("考慮使用 useCallback 優化狀態更新函數")

        if 'useEffect' in chunk.content:
            practices.append("確保 useEffect 有正確的依賴數組")

        # 性能優化建議
        if self.has_inline_functions(chunk.content):
            practices.append("避免在 render 中定義內聯函數")

        if self.has_large_component(chunk.content):
            practices.append("考慮將組件拆分為更小的子組件")

        return practices
```

#### 4. 多階段推理流程

```python
class MultiStageCodeReasoning:
    """模仿 PathRAG 的多階段推理流程"""

    def __init__(self, basic_analyzer, domain_analyzer, advanced_reasoner):
        self.basic_analyzer = basic_analyzer
        self.domain_analyzer = domain_analyzer
        self.advanced_reasoner = advanced_reasoner

    async def process_code_query(self, query: str, project_name: str) -> Dict[str, Any]:
        """三階段代碼分析推理"""

        # Stage 1: 基礎代碼檢索和語法分析
        stage1_results = await self.stage1_basic_analysis(query, project_name)

        # Stage 2: 領域特定分析和模式識別
        stage2_results = await self.stage2_domain_analysis(
            query, stage1_results, project_name
        )

        # Stage 3: 高級推理和架構分析
        stage3_results = await self.stage3_advanced_reasoning(
            query, stage2_results
        )

        # 綜合三階段結果
        final_response = await self.synthesize_multi_stage_results(
            stage1_results, stage2_results, stage3_results
        )

        return final_response

    async def stage1_basic_analysis(self, query: str, project_name: str) -> Dict[str, Any]:
        """Stage 1: 基礎語法分析和代碼檢索"""

        # 1. 基礎關鍵詞提取
        keywords = self.extract_basic_keywords(query)

        # 2. 代碼塊檢索
        code_chunks = await self.basic_analyzer.retrieve_relevant_chunks(
            keywords, project_name
        )

        # 3. 語法分析
        syntax_analysis = await self.basic_analyzer.analyze_syntax(code_chunks)

        # 4. 基礎結構分析
        structure_analysis = await self.basic_analyzer.analyze_structure(code_chunks)

        return {
            'stage': 1,
            'keywords': keywords,
            'code_chunks': code_chunks,
            'syntax_analysis': syntax_analysis,
            'structure_analysis': structure_analysis,
            'confidence': 0.6  # 基礎階段的置信度
        }

    async def stage2_domain_analysis(self, query: str,
                                   stage1_results: Dict[str, Any],
                                   project_name: str) -> Dict[str, Any]:
        """Stage 2: 領域特定分析"""

        # 1. 識別技術領域
        domain = await self.domain_analyzer.identify_domain(
            query, stage1_results['code_chunks']
        )

        # 2. 領域特定的模式識別
        domain_patterns = await self.domain_analyzer.identify_patterns(
            stage1_results['code_chunks'], domain
        )

        # 3. 框架/庫特定的分析
        framework_analysis = await self.domain_analyzer.analyze_framework_usage(
            stage1_results['code_chunks'], domain
        )

        # 4. 最佳實踐檢查
        best_practices = await self.domain_analyzer.check_best_practices(
            stage1_results['code_chunks'], domain
        )

        return {
            'stage': 2,
            'domain': domain,
            'domain_patterns': domain_patterns,
            'framework_analysis': framework_analysis,
            'best_practices': best_practices,
            'enhanced_chunks': await self.enhance_chunks_with_domain_knowledge(
                stage1_results['code_chunks'], domain
            ),
            'confidence': 0.8  # 領域分析後提高置信度
        }

    async def stage3_advanced_reasoning(self, query: str,
                                      stage2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: 高級推理和架構分析"""

        # 1. 架構模式識別
        architectural_patterns = await self.advanced_reasoner.identify_architectural_patterns(
            stage2_results['enhanced_chunks']
        )

        # 2. 設計原則分析
        design_principles = await self.advanced_reasoner.analyze_design_principles(
            stage2_results['enhanced_chunks']
        )

        # 3. 性能影響分析
        performance_analysis = await self.advanced_reasoner.analyze_performance_implications(
            stage2_results['enhanced_chunks']
        )

        # 4. 改進建議生成
        improvement_suggestions = await self.advanced_reasoner.generate_improvement_suggestions(
            query, stage2_results
        )

        # 5. 風險評估
        risk_assessment = await self.advanced_reasoner.assess_risks(
            stage2_results['enhanced_chunks']
        )

        return {
            'stage': 3,
            'architectural_patterns': architectural_patterns,
            'design_principles': design_principles,
            'performance_analysis': performance_analysis,
            'improvement_suggestions': improvement_suggestions,
            'risk_assessment': risk_assessment,
            'confidence': 0.95  # 高級推理後的高置信度
        }

    async def synthesize_multi_stage_results(self, stage1: Dict, stage2: Dict, stage3: Dict) -> Dict[str, Any]:
        """綜合三階段分析結果"""

        # 1. 權重分配
        weights = {
            'stage1': 0.2,  # 基礎分析權重
            'stage2': 0.4,  # 領域分析權重
            'stage3': 0.4   # 高級推理權重
        }

        # 2. 綜合置信度計算
        overall_confidence = (
            stage1['confidence'] * weights['stage1'] +
            stage2['confidence'] * weights['stage2'] +
            stage3['confidence'] * weights['stage3']
        )

        # 3. 構建最終響應
        final_response = {
            'query_analysis': {
                'basic_keywords': stage1['keywords'],
                'identified_domain': stage2['domain'],
                'complexity_level': self.assess_complexity_level(stage1, stage2, stage3)
            },
            'code_analysis': {
                'syntax_analysis': stage1['syntax_analysis'],
                'structure_analysis': stage1['structure_analysis'],
                'domain_patterns': stage2['domain_patterns'],
                'architectural_patterns': stage3['architectural_patterns']
            },
            'insights_and_recommendations': {
                'best_practices': stage2['best_practices'],
                'improvement_suggestions': stage3['improvement_suggestions'],
                'performance_considerations': stage3['performance_analysis'],
                'risk_factors': stage3['risk_assessment']
            },
            'metadata': {
                'overall_confidence': overall_confidence,
                'processing_stages': 3,
                'domain_identified': stage2['domain'],
                'stage_breakdown': {
                    'stage1_confidence': stage1['confidence'],
                    'stage2_confidence': stage2['confidence'],
                    'stage3_confidence': stage3['confidence']
                }
            }
        }

        return final_response
```

### 性能提升目標

基於 PathRAG 的成果，設定以下性能提升目標：

| 指標 | 當前狀態 | 目標提升 | PathRAG 參考 |
|------|----------|----------|--------------|
| 代碼理解準確率 | 基準 | +25% | 38% → 47% |
| 特定框架查詢準確率 | 基準 | +28% | H&E染色圖像提升 |
| 技術文檔理解 | 基準 | +32% | PubMed數據提升 |
| 複雜查詢處理 | 基準 | +30% | 書籍數據提升 |
| 響應速度 | 基準 | +40% | 流式剪枝效果 |

---

## 實施路線圖

### 第一階段：基礎設施升級（4-6週）

#### Week 1-2: 多模態檢索實現
- [ ] 實現 `MultiModalRetrievalStrategy` 類
- [ ] 開發 `QueryAnalyzer` 進行查詢意圖分析
- [ ] 建立關鍵詞提取機制（高低層次分離）
- [ ] 創建查詢複雜度評估算法

#### Week 3-4: 輕量化圖服務
- [ ] 開發 `LightweightGraphService`
- [ ] 實現內存索引和路徑緩存
- [ ] 建立預計算常用查詢機制
- [ ] 實施智能查詢路由系統

#### Week 5-6: 路徑基礎索引
- [ ] 實現 `PathBasedIndexer`
- [ ] 開發關係路徑提取算法
- [ ] 建立流式剪枝機制
- [ ] 創建路徑到提示的轉換系統

### 第二階段：領域特定增強（4-6週）

#### Week 7-8: 域知識庫建設
- [ ] 建立 `ReactKnowledgeBase`
- [ ] 開發 `VueKnowledgeBase`
- [ ] 創建 `DjangoKnowledgeBase`
- [ ] 實現模式匹配器

#### Week 9-10: 智能片段提取
- [ ] 實現 `IntelligentCodePatchExtractor`
- [ ] 開發複雜度熱點檢測
- [ ] 建立重疊式代碼分析
- [ ] 創建信息密度評估算法

#### Week 11-12: 多階段推理
- [ ] 實現 `MultiStageCodeReasoning`
- [ ] 開發三階段分析流程
- [ ] 建立結果綜合機制
- [ ] 創建置信度計算系統

### 第三階段：性能優化和整合（3-4週）

#### Week 13-14: 性能監控和優化
- [ ] 實現 `PerformanceOptimizer`
- [ ] 開發自適應緩存策略
- [ ] 建立性能監控儀表板
- [ ] 創建自動調優機制

#### Week 15-16: 系統整合和測試
- [ ] 整合所有新組件
- [ ] 進行端到端測試
- [ ] 性能基準測試
- [ ] 用戶接受度測試

### 第四階段：驗證和優化（2-3週）

#### Week 17-18: 效果驗證
- [ ] A/B 測試新舊系統
- [ ] 收集性能指標
- [ ] 用戶反饋分析
- [ ] 效果對比報告

#### Week 19: 最終優化
- [ ] 根據測試結果調優
- [ ] 文檔更新
- [ ] 部署生產環境
- [ ] 監控和維護計劃

### 關鍵里程碑

1. **基礎架構完成** (Week 6)
   - 多模態檢索系統可用
   - 輕量化圖服務運行
   - 路徑索引機制建立

2. **領域增強完成** (Week 12)
   - 主要框架知識庫建立
   - 智能片段提取可用
   - 多階段推理系統運行

3. **性能優化完成** (Week 16)
   - 系統整合完成
   - 性能達到目標指標
   - 用戶體驗顯著提升

4. **正式發布** (Week 19)
   - 所有功能穩定運行
   - 文檔完善
   - 生產環境部署

### 風險管控

#### 技術風險
- **緩解措施**: 分階段實施，每階段都有回退計劃
- **監控指標**: 系統穩定性、響應時間、準確率

#### 性能風險
- **緩解措施**: 持續性能監控，自動調優機制
- **監控指標**: 內存使用、查詢響應時間、緩存命中率

#### 兼容性風險
- **緩解措施**: 保持向後兼容，漸進式升級
- **監控指標**: 現有功能完整性、API 兼容性

---

## 總結

本分析報告詳細研究了 LightRAG 和 PathRAG 的創新技術，並提出了針對 Agentic-RAG 專案的全面改善方案。主要改善方向包括：

1. **多模態檢索機制**: 實現 local、global、hybrid、mix 四種檢索模式
2. **路徑基礎索引**: 借鑑 PathRAG 的流式剪枝和路徑提示技術
3. **輕量化圖服務**: 提升大型項目的處理能力和響應速度
4. **領域特定增強**: 建立框架/語言特定的知識庫和分析能力
5. **智能性能優化**: 自適應緩存和查詢路由機制

通過實施這些改善，預期可以達到：
- 代碼理解準確率提升 25-30%
- 查詢響應速度提升 40%
- 大型項目分析能力顯著增強
- 用戶體驗大幅改善

該改善方案既保持了現有系統的優勢，又融合了最新的 RAG 技術創新，將使 Agentic-RAG 專案在代碼理解和分析領域達到業界領先水平。
