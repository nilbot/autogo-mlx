# Custom Rules for AutoGo-MLX Workspace

## Q&A Log Rule

Whenever the user says "wow good to know, can you save it to QnA?" (or similar requests to save a key insight/concept to the QnA folder):
1. **Identify the Concept**: Extract the core question/concept and the detailed technical answer/explanation from the recent conversation context.
2. **File Location**: Save the file inside the `docs/qna/` directory of the workspace.
3. **Naming Convention**: Use a lowercase kebab-case filename describing the topic (e.g., `docs/qna/replay-buffer-sampling.md`).
4. **Document Structure**: Use the following standard markdown layout:
   ```markdown
   # [Descriptive Question/Title]

   ## Context
   [Brief context of when/why this question arose during training/development]

   ## Answer
   [Detailed technical explanation, including code snippets, math formulas, and architectural design choices if applicable]
   ```
5. **Execution**: Proactively create or update the file using your file writing tool. Propose/explain the creation to the user briefly.
