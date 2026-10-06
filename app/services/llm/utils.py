import re

# Reasoning models (Qwen3, DeepSeek-R1, ...) may emit <think> blocks, including an
# unterminated one when generation is cut off.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text)
