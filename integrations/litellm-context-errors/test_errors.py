"""Exercise LiteLLM's conversion from provider exceptions to Responses HTTP errors."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.proxy_server import openai_exception_handler
from starlette.requests import Request


async def check_errors():
    logging.disable(logging.CRITICAL)
    hooks = AsyncMock()
    hooks.post_call_failure_hook.return_value = None
    hooks.post_call_response_headers_hook.return_value = {}
    auth = UserAPIKeyAuth()
    request = Request({"type": "http", "method": "POST", "path": "/v1/responses", "headers": []})
    context = "context_length_exceeded"
    cases = [
        (litellm.ContextWindowExceededError, 400, None, context),
        (litellm.BadRequestError, 400, context, context),
        (litellm.BadRequestError, 400, "invalid_request", "400"),
        (litellm.BadRequestError, 400, None, "400"),
        (litellm.BadRequestError, 429, context, "429"),
        (litellm.ContextWindowExceededError, 500, None, "500"),
    ]
    processor = ProxyBaseLLMRequestProcessing({})
    with patch.object(ProxyBaseLLMRequestProcessing, "get_custom_headers", return_value={}):
        for error_type, status, provider_code, expected in cases:
            exc = error_type("model input", model="test", llm_provider="openai")
            exc.status_code, exc.code = status, provider_code
            try:
                await processor.handle_llm_api_exception(exc, auth, hooks)
            except ProxyException as converted:
                response = await openai_exception_handler(request, converted)
            else:
                raise AssertionError("Expected LiteLLM to raise an HTTP exception")
            actual = response.status_code, json.loads(response.body)["error"]["code"]
            assert actual == (status, expected), (error_type.__name__, actual)


if __name__ == "__main__":
    asyncio.run(check_errors())
