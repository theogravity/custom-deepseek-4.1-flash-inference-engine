# LiteLLM context overflow

This optional image preserves the OpenAI `context_length_exceeded` code when the engine rejects an oversized request. Codex uses that code to classify context overflow through the proxy. It extends the same pinned LiteLLM release used on the workstation. The image build runs the regression test.

```sh
docker build -t litellm-context-errors:local integrations/litellm-context-errors
```
