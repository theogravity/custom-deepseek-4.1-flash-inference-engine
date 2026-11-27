"""Preserve the OpenAI context error code across LiteLLM's HTTP error conversion."""

from importlib.util import find_spec
from pathlib import Path


def replace_once(path, before, after):
    source = path.read_text()
    if source.count(before) != 1:
        raise RuntimeError(f"Expected one matching block in {path}")
    updated = source.replace(before, after)
    compile(updated, str(path), "exec")
    path.write_text(updated)


proxy = Path(find_spec("litellm").origin).parent / "proxy"
replace_once(
    proxy / "common_request_processing.py",
    '            openai_code=getattr(e, "code", None),\n            code=_code,',
    '''            openai_code=(
                "context_length_exceeded"
                if _code == 400 and isinstance(e, litellm.ContextWindowExceededError)
                else getattr(e, "code", None)
            ),
            code=_code,''',
)
replace_once(
    proxy / "_types.py",
    '            "param": self.param,\n            "code": self.code,',
    '''            "param": self.param,
            "code": (
                "context_length_exceeded"
                if self.code == "400" and self.openai_code == "context_length_exceeded"
                else self.code
            ),''',
)
