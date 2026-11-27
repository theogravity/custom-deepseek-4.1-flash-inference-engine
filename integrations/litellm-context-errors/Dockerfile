FROM litellm/litellm:v1.109.0@sha256:26aecfb02fb17ac6590f0549478561f9fbb3f69314317320b2e32a21beed4006
RUN --mount=type=bind,source=apply.py,target=/tmp/patch-context-errors.py \
    python /tmp/patch-context-errors.py
RUN --network=none --mount=type=bind,source=test_errors.py,target=/tmp/test-errors.py \
    LITELLM_LOCAL_MODEL_COST_MAP=True python -B /tmp/test-errors.py
