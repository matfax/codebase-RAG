# PRD: Documentation Modernization and IDE Integration

## Introduction/Overview

The Codebase RAG MCP Server project needs comprehensive documentation updates to align with current implementation and improve user experience. The documentation currently references outdated Poetry dependency management and lacks proper IDE integration guides. This feature will modernize all project documentation to reflect the current uv-based setup and provide clear integration instructions for popular development environments.

## Goals

1. **Modernize dependency management documentation** - Completely migrate from Poetry to uv references
2. **Improve user onboarding experience** - Provide clear, step-by-step setup instructions for different IDEs
3. **Ensure documentation accuracy** - Align all documentation with current codebase implementation
4. **Enhance project discoverability** - Create comprehensive tool references and best practice guides
5. **Streamline documentation structure** - Organize documentation into logical, easily navigable sections

## User Stories

**As a new MCP user**, I want to quickly understand how to install and configure this server so that I can start using it in my preferred development environment within 10 minutes.

**As a Claude Code user**, I want simple registration commands so that I can add this MCP server without complex configuration.

**As a VS Code user**, I want example configuration files so that I can integrate this MCP server into my existing workflow.

**As a Gemini CLI user**, I want specific setup instructions so that I can use this server with my preferred AI assistant.

**As a developer evaluating this tool**, I want to understand supported languages and capabilities so that I can determine if it fits my project needs.

**As an existing user**, I want access to comprehensive tool references so that I can utilize all available MCP tools effectively.

## Functional Requirements

### Documentation Structure Updates
1. **The system must update CLAUDE.md first** to serve as the foundation for README.md updates
2. **The system must create a docs/ directory** to house additional documentation files
3. **The system must completely remove Poetry references** from all documentation
4. **The system must update dependency management instructions** to use uv exclusively

### IDE Integration Documentation
5. **The system must provide Claude Code integration instructions** with simple `claude mcp add` commands
6. **The system must include Gemini CLI integration examples** with configuration snippets
7. **The system must provide VS Code integration guidance** with example configuration files
8. **The system must include placeholder sections** marked for future screenshot and detailed instruction additions

### Content Updates
9. **The system must update the Supported Languages section** to reflect only currently implemented language support
10. **The system must revise the Setup section** with modern uv-based installation steps
11. **The system must replace the "Running the MCP Server" section** with IDE integration instructions
12. **The system must update the Project Structure section** to reflect current codebase organization

### New Documentation Files
13. **The system must create MCP_TOOLS.md** with comprehensive tool reference documentation
14. **The system must create a best practices guide** covering cross-project search, organization tips, and performance optimization
15. **The system must update README.md** to reference the search tool as the primary example while linking to comprehensive documentation

### Content Removal
16. **The system must remove the "Usage Examples" section** containing outdated Via MCP Tool, Via Manual Tool, and Using with Python examples
17. **The system must remove the "Try These Live Examples" section** completely
18. **The system must remove all Poetry-related installation and setup instructions**

## Non-Goals (Out of Scope)

- **Developer-focused documentation** - Will be addressed in future updates
- **Comprehensive testing documentation** - Current testing approach is adequate and doesn't need README inclusion
- **Live example implementations** - Focus on clear instructions rather than working examples
- **Return value schemas for MCP tools** - Tool documentation will focus on parameters and usage

## Design Considerations

- **Documentation hierarchy**: CLAUDE.md → README.md → docs/ folder structure
- **User-friendly language**: Target general MCP users rather than developers
- **Progressive disclosure**: README.md provides overview, docs/ provides detailed references
- **Visual placeholders**: Mark sections where screenshots and detailed instructions will be added later

## Technical Considerations

- **Maintain consistency with current CLAUDE.md structure** while updating content
- **Ensure all file paths and commands reflect current project structure**
- **Preserve existing environment variable documentation and configuration examples**
- **Keep MCP server communication protocols and technical details accurate**

## Success Metrics

- **User onboarding time reduction** - Setup completion within 10 minutes for new users
- **Documentation accuracy** - All installation and setup instructions work as documented
- **Reduced support questions** - Fewer issues related to installation and IDE integration
- **Improved project adoption** - Clear capability understanding leads to appropriate tool usage

## Open Questions

- Should the best practices guide include specific query examples for different programming languages?
- How detailed should the VS Code configuration examples be (workspace settings vs. user settings)?
- Should we include troubleshooting sections for common IDE integration issues?
- What level of detail is needed for the MCP_TOOLS.md parameter descriptions?
