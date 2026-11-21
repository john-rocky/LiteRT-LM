# IBM Granite 4.2

Canonical LiteRT-LM prompt template and metadata configuration for the IBM
Granite 4.2 family (e.g. Granite 4.2 3B).

## Chat Template

-   Canonical template: `chat_template.jinja`
-   Specification reference:
    [Granite 4.2 Chat Template](https://huggingface.co/ibm-granite/granite-4.2-3b/blob/main/chat_template.jinja)

### Features & Standardization

1.  **Role Markers**:

    -   Uses `<|im_start|>` and `<|im_end|>` delimiters for message roles:
        `system`, `user`, `assistant`, `tool`.
    -   The `system` turn is always rendered, empty when no system message is
        given, as in the vendor template.
    -   No BOS token is emitted; the Granite tokenizer adds none.

2.  **Tool Calling**:

    -   Function/tool signatures are listed as JSON inside `<tools>` and
        `</tools>` in the `system` turn, followed by the vendor's format
        instructions.
    -   Assistant function calls are formatted as XML:
        `<tool_call>\n<function=name>\n<parameter=arg>\nvalue\n</parameter>\n</function>\n</tool_call>`.
    -   Tool outputs are supplied under the `tool` role, each response wrapped
        in `<tool_response>` tags inside a `user` turn.
    -   `LlmMetadataProto.pbtext` sets `tool_call_config`, so LiteRT-LM 0.19.0
        and later return these calls as structured `tool_calls`.

3.  **Thinking Mode Toggle**:

    -   Adheres to the LiteRT-LM Chat Template Standard `enable_thinking`
        configuration (defaults to `true`).
    -   When thinking is enabled, the generation prompt pre-opens the thought
        channel: `<|im_start|>assistant\n<think>\n`.
    -   When thinking is disabled (`enable_thinking=false`), the generation
        prompt is `<|im_start|>assistant\n<think></think>` and the model
        answers directly.

4.  **Thinking Channel**:

    -   `LlmMetadataProto.pbtext` defines the `thought` channel with delimiter
        tokens `<think>` and `</think>`.
    -   Past assistant turns keep their reasoning as `<think>\n...\n</think>\n`
        before the answer (LiteRT-LM fills `reasoning_content` from the thought
        channel). This is the vendor template's `truncate_history_thinking`
        set to `false`: the default here, because each new render of a
        conversation must start with the previous one.
    -   `extra_context` `{"truncate_history_thinking": true}` gives the vendor
        default: assistant turns before the last user message render with an
        empty `<think></think>` pair. After a thinking turn, the next user
        message then no longer extends the previous render, and LiteRT-LM
        rejects it.
