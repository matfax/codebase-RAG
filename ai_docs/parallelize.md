# /parallelize

Create N git worktrees and run AI agents in each worktree for concurrent development.

## Variables

FEATURE_NAME: $ARGUMENTS
PLAN_TO_EXECUTE: $ARGUMENTS
NUMBER_OF_PARALLEL_WORKTREES: $ARGUMENTS

## Execute Phase 1: Environment Setup

**Safety Check:** Before proceeding, I will check for any uncommitted changes in the current directory. If changes are found, I will exit with an error, prompting you to commit or stash them.

**Directory Setup:** I will ensure the `trees/` directory exists at the project root.

> The following steps will be executed in parallel for each worktree to ensure efficiency. I will use absolute paths for all commands.

For each instance from 1 to NUMBER_OF_PARALLEL_WORKTREES, I will:

1.  **Create Worktree:** `git worktree add -b <FEATURE_NAME>-<instance-number> ./trees/<FEATURE_NAME>-<instance-number>`
2.  **Copy Environment:** Copy the `.env` file to `./trees/<FEATURE_NAME>-<instance-number>/.env` if it exists.
3.  **Install Dependencies:** Change directory to `./trees/<FEATURE_NAME>-<instance-number>` and run `uv sync` to install dependencies specified in `uv.lock`.

**Verification:** After setup, I will run `git worktree list` to confirm all worktrees were created successfully.

## Execute Phase 2: Parallel Development

I will now create NUMBER_OF_PARALLEL_WORKTREES independent AI sub-agents. Each agent will work in its own dedicated worktree to build the feature concurrently. This allows for isolated development and testing.

Each sub-agent will operate in its respective directory:
-   Agent 1: `trees/<FEATURE_NAME>-1/`
-   Agent 2: `trees/<FEATURE_NAME>-2/`
-   ...and so on.

**Core Task:**
Each agent will independently and meticulously implement the engineering plan detailed in **PLAN_TO_EXECUTE**.

**Progress Tracking:**
Each agent will report its progress by writing to a JSON file in the shared `progress/` directory:

```json
{
  "agent": "<FEATURE_NAME>-<instance-number>",
  "status": "in-progress",
  "step": "parser module implemented",
  "last_updated": "'$(date -Iseconds)'"
}
```
The file will be named `progress/<FEATURE_NAME>-<instance-number>.json`.

**Completion and Validation:**
Upon completing its task, each agent will:

1.  **Generate Report:** Create a comprehensive `RESULTS.md` file in its worktree root, detailing the changes made and providing usage examples if applicable.
2.  **Run Tests:** Execute `.venv/bin/pytest tests/` within its worktree to ensure all tests pass.
3.  **Fix Issues:** Address any test failures, linting errors, or type-checking issues identified.
4.  **Final Status:** Update its progress file to reflect "completed" status.

This process ensures that we can review multiple, complete, and validated implementations to select the best one for merging into the main branch.
