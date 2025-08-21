# PRD: Simplify `multi_modal_search` Tool Output

## 1. Problem Statement

The `multi_modal_search` tool is powerful, but its JSON output is excessively verbose for standard use cases. It returns a large amount of metadata related to performance, query analysis, and scoring. While this information is valuable for debugging or deep analysis, it clutters the output for general queries and negatively impacts usability. This contrasts with the cleaner, more direct output of the standard `search` tool.

**Goal:** Improve the user experience of `multi_modal_search` by providing a concise, simple output by default, while retaining the option to access the full, detailed output for advanced analysis or debugging purposes.

## 2. Proposed Solution

We will introduce a new boolean parameter to the `multi_modal_search` function signature: `minimal_output`.

- **Parameter:** `minimal_output: bool`
- **Default Value:** `True`

### Behavior Specification

- **When `minimal_output=True` (Default Behavior):**
  - The tool will return a simplified JSON object, closely aligning with the `search` tool's output for consistency.
  - The top-level object will only contain the `results` array and a basic `_performance` summary. Keys like `query_analysis`, `performance`, and `multi_modal_metadata` will be removed.
  - Each object within the `results` array will be simplified to include only essential information: `file_path`, `content`, `breadcrumb`, `chunk_type`, `language`, `line_start`, `line_end`. Detailed scoring fields (`local_score`, `global_score`, `combined_score`, etc.) will be omitted.

- **When `minimal_output=False` (Expert/Debug Mode):**
  - The tool will return the complete, unabridged JSON object as it currently does. This allows developers or power users to access all metadata for deep analysis.

## 3. Implementation Plan

1.  **Modify Tool Signature:**
    - Locate the definition of the `multi_modal_search` tool in the codebase.
    - Add the `minimal_output: bool = True` parameter to the function signature.

2.  **Implement Conditional Logic:**
    - Inside the `multi_modal_search` function, before returning the result, add a conditional block.
    - If `minimal_output` is `True`, process the full result dictionary to strip out the non-essential keys and simplify the `results` array as specified above.
    - If `minimal_output` is `False`, return the full result dictionary untouched.

3.  **Update Documentation:**
    - Modify the documentation for `multi_modal_search` (e.g., in `docs/MCP_TOOLS.md` and any other relevant docstrings) to explain the new `minimal_output` parameter, its default value, and how to use it.

4.  **Add/Update Unit Tests:**
    - In the relevant test file (e.g., `tests/test_advanced_search_tool.py`), add assertions to verify both modes:
        - A default call to `multi_modal_search` returns the simplified output.
        - A call with `minimal_output=False` returns the full, verbose output.

## 4. Success Criteria

- The feature is considered complete when `multi_modal_search()` by default returns a concise output, and `multi_modal_search(minimal_output=False)` returns the full, verbose output.
- All relevant documentation and tests are updated to reflect the change.
