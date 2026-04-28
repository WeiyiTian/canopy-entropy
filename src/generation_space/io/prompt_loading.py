import json
from itertools import islice
from pathlib import Path


def load_prompts(prompt_file: Path, num_prompts: int) -> list[str]:
    """
    Loads prompt strings from a JSONL file. Removes trailing code block markers 
    and whitespace.

    Args:
        prompt_file: Path to a JSONL file with a `prompt` field on each line.
        num_prompts: Maximum number of prompts to load. Loads all prompts when
            set to 0 or less.

    Returns:
        List of prompt strings in file order, with whitespace and trailing code block 
        markers removed.
    """
    with prompt_file.open(encoding="utf-8") as f:
        prompts = (
            json.loads(line)["prompt"].rstrip().removesuffix("```").rstrip()
            for line in f if line.strip()
        )
        return list(islice(prompts, num_prompts)) if num_prompts > 0 else list(prompts)
