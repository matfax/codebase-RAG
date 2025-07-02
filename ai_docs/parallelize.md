# /parallelize

Create N number of git worktrees and run Claude Code instances in each worktree for concurrent development.

## Variables

NUMBER_OF_PARALLEL_WORKTREES: $ARGUMENTS
FEATURE_NAME: $ARGUMENTS
PLAN_TO_EXECUTE: $ARGUMENTS

## Execute Phase 1

Always check that the current directory uncommit changes. If there are, immediately exit the command with an error message, letting user know to commit or stash their changes before running the command.

CREATE new directory trees/ if it doesn't already exist.

> Execute these steps in parallel for concurrency
>
> Use absolute paths for all commands

CREATE NUMBER_OF_PARALLEL_WORKTREES worktrees using the following commands:

RUN git worktree add -b FEATURE_NAME-<instance-number> ./trees/FEATURE_NAME-<instance-number>
COPY ./.env to ./trees/FEATURE_NAME-<instance-number>/.env (if exists)
RUN cd ./trees/FEATURE_NAME-<instance-number>
RUN .venv/bin/poetry install

VERIFY setup by running git worktree list

## Execute Phase 2

Read and understand the PLAN_TO_EXECUTE meticulously before proceeding.

Create NUMBER_OF_PARALLEL_WORKTREES new subagents that use the Task tool to create N versions of the same feature in parallel. Kick off all subagents to start work at the same time DO NOT run one after another. Each subagent will run in it's own worktree directory located in trees/. DO NOT make changes to the main directory.

This enables the subagents to concurrently build the same feature in separate worktree directories in parallel so we can test and validate each subagent's changes in isolation then pick the best worktree to merge into main. Each subagent's implementation must be complete and production ready.

The first subagent will run in trees/<FEATURE_NAME>-<instance-number>/
The second subagent will run in trees/<FEATURE_NAME>-<instance-number>/
...
The last subagent will run in trees/<FEATURE_NAME>-<instance-number>/

The code in trees/<FEATURE_NAME>-<instance-number>/ will be identical to the code in the current branch. It will be setup and ready for you to build the feature end to end.

Each subagent will independently meticulously read and implement the engineering plan detailed in PLAN_TO_EXECUTE in their respective workspace.

When the subagent completes it's work, have the subagent report their final changes made in a comprehensive `RESULTS.md` file at the root of their respective workspace. They should always include example of how to use the code if applicable.

Each subagent must also run `.venv/bin/pytest tests/` at the root of their respective worktree and fix any test failures that they encounter. They should also run any additional validation commands like linting or type checking as specified in the project's CLAUDE.md file.
