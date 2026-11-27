export default function(pi) {
  const config = JSON.parse(process.env.MODEL_CLI_PI_CONFIG);
  pi.registerProvider(config.provider, config.connection);
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== config.provider) return;
    const payload = { ...event.payload };
    if (config.direct && config.deepseek) {
      // vLLM's native encoder reads these controls from chat_template_kwargs.
      payload.chat_template_kwargs = {
        ...payload.chat_template_kwargs,
        thinking: payload.thinking?.type !== "disabled",
        reasoning_effort: payload.reasoning_effort || "high",
      };
      delete payload.thinking;
      delete payload.reasoning_effort;
    }
    return payload;
  });
}
