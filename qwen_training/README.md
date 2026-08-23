# Qwen Model Capacity Limitations

This folder contains local testing scripts (like `test_qwen.py`) for running the AugAgent framework against local, quantized 7B parameter models (e.g., Qwen 2.5 7B) via Ollama. 

During rigorous testing, we have identified severe limitations when comparing this setup to frontier models (like Gemini 3.1 Pro coupled with Antigravity).

## Identified Capability Gaps

1. **Attention Drift and Instruction Following Decay**
   - **Issue**: Despite strict system prompts demanding JSON-only tool outputs, the model's chat fine-tuning biases it heavily toward conversational text. 
   - **Impact**: After a few turns in the ReAct loop, the model "forgets" the system constraints and starts outputting raw conversational markdown instead of structured JSON, completely breaking execution parsing.

2. **Brittle Error Recovery**
   - **Issue**: The model has a very short reasoning horizon. When a terminal command fails (e.g., PowerShell piping syntax errors), the model lacks the global context to infer the root cause.
   - **Impact**: Instead of pivoting its strategy, the model tends to panic, hallucinate missing modules, or endlessly loop by trying the exact same broken workaround repeatedly.

3. **Conversational Bias Over Execution Bias**
   - **Issue**: The model is trained to act as an "advisor."
   - **Impact**: When asked to solve a problem, it often just outputs a command block and tells the human user *how* to run it, rather than autonomously utilizing its own Tool Calling functions to execute the command directly.

### Recommended Workarounds
To achieve reliable agentic loops with these small models, developers must implement **Guided Generation** (e.g., using `Outlines` or strict JSON Grammars at the API level) to coerce the model's output syntax forcefully, rather than relying on the model to follow prompt instructions.
