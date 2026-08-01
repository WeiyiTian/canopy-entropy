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


def load_judge_prompt(prompts_dir: Path, task: str) -> tuple[Path, str]:
    """
    Loads the few-shot similarity judge prompt for one task.

    Args:
        prompts_dir: Directory holding one judge prompt per task.
        task: Task name, matching a `{task}.txt` file in that directory.

    Returns:
        (prompt_path, prompt_contents):
        - prompt_path: Path to the prompt file.
        - prompt_contents: Contents of the prompt file.
    """
    prompt_path = prompts_dir / f"{task}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"No judge prompt for task '{task}' at {prompt_path}.")
    return prompt_path, prompt_path.read_text(encoding="utf-8")
