import asyncio
import re
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from openai import AsyncOpenAI

SIMILAR_MARKER = "response: similar"
DIFFERENT_MARKER = "response: different"
PLACEHOLDER_PATTERN = re.compile(r"\{(task_prompt|response_a|response_b)\}")


class JudgeStatus(StrEnum):
    """
    Outcome of a single judge call.

    Attributes:
        OK: A verdict was parsed from the reply.
        PARSE_FAILED: The reply carried neither verdict marker.
        TRUNCATED: The reply hit the output cap before stating a verdict.
        REQUEST_FAILED: The request errored after the client's retries were exhausted,
            or came back carrying no choices to parse.
    """

    OK = "ok"
    PARSE_FAILED = "parse_failed"
    TRUNCATED = "truncated"
    REQUEST_FAILED = "request_failed"


@dataclass(slots=True, frozen=True)
class PairJudgment:
    """
    One judge verdict on an ordered pair of rollouts.

    Attributes:
        row: Index of the rollout presented as "Response 1".
        column: Index of the rollout presented as "Response 2".
        is_similar: Whether the judge called the pair similar. Unparseable and failed
            calls resolve to False.
        status: Whether the call succeeded, came back unparseable, or errored.
        text: Raw judge reply, retained so verdicts can be audited or re-parsed.
    """

    row: int
    column: int
    is_similar: bool
    status: JudgeStatus
    text: str


@dataclass(slots=True)
class JudgmentMatrix:
    """
    Ordered-pair verdicts for one prompt's subsampled rollouts.

    Attributes:
        judgments: Boolean tensor of shape [m, m], where `[i, j]` is True when the judge
            called ordered pair (i, j) similar. Pairs never asked stay False.
        pair_judgments: Every verdict collected, in the order it was requested.
    """

    judgments: torch.Tensor
    pair_judgments: list[PairJudgment] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """
        Returns the fraction of calls that yielded no usable verdict.
        Failures resolve to not similar, so they remove edges and inflate diversity.
        """
        if not self.pair_judgments:
            return 0.0
        failures = sum(judgment.status is not JudgeStatus.OK for judgment in self.pair_judgments)
        return failures / len(self.pair_judgments)

    def count_status(self, status: JudgeStatus) -> int:
        """Returns how many calls ended with the given status."""
        return sum(judgment.status is status for judgment in self.pair_judgments)


class SimilarityJudge:
    """
    Prompted binary similarity judge backed by any OpenAI-compatible chat endpoint.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        prompt_template: str,
        max_concurrency: int = 32,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        """
        Args:
            model: Model identifier the endpoint expects.
            base_url: OpenAI-compatible API root, e.g. `https://openrouter.ai/api/v1`.
            api_key: API Key for the endpoint.
            prompt_template: Few-shot judge prompt containing the `{task_prompt}`,
                `{response_a}` and `{response_b}` placeholders.
            max_concurrency: Maximum number of requests in flight at once.
            temperature: Judge sampling temperature.
            top_p: Judge nucleus sampling cutoff.
            max_tokens: Output cap, sized to fit the reasoning the few-shot prompt elicits.
            timeout: Per-request timeout in seconds.
            max_retries: Client-level retries before a request is recorded as failed.

        Raises:
            ValueError: If the template is missing any required placeholder.
        """
        missing = {"task_prompt", "response_a", "response_b"} - set(
            PLACEHOLDER_PATTERN.findall(prompt_template)
        )
        if missing:
            raise ValueError(f"Judge prompt template is missing placeholders: {sorted(missing)}.")

        self._model = model
        self._prompt_template = prompt_template
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def build_prompt(self, task_prompt: str, response_a: str, response_b: str) -> str:
        """
        Fills the judge template for one ordered pair.
        """
        values = {"task_prompt": task_prompt, "response_a": response_a, "response_b": response_b}
        return PLACEHOLDER_PATTERN.sub(lambda match: values[match.group(1)], self._prompt_template)

    async def judge_pairs(
        self,
        task_prompt: str,
        responses: list[str],
        pairs: list[tuple[int, int]],
    ) -> list[PairJudgment]:
        """
        Judges the given ordered pairs concurrently.

        Args:
            task_prompt: Prompt the rollouts were generated from.
            responses: The m subsampled rollout texts.
            pairs: Ordered `(row, column)` index pairs to judge.

        Returns:
            List of PairJudgment objects aligned with `pairs`.
            Launch one async task for every (row, column) pair, wait until all of them 
                finish, then return the results as a list.
        """
        return list(await asyncio.gather(
            *(self._judge_pair(task_prompt, responses[row], responses[column], row, column)
              for row, column in pairs) # unpacks the generator
        ))

    async def _judge_pair(
        self,
        task_prompt: str,
        response_a: str,
        response_b: str,
        row: int,
        column: int,
    ) -> PairJudgment:
        """Issues one judge call and parses its verdict."""
        prompt = self.build_prompt(task_prompt, response_a, response_b)
        async with self._semaphore:
            try:
                # chat: use the Chat Completions API
                # completions: the endpoint for generating completions
                # create(...): send one request
                completion = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    top_p=self._top_p,
                    max_tokens=self._max_tokens,
                    stop=["###"],
                )
            except Exception as error:
                return PairJudgment(
                    row=row,
                    column=column,
                    is_similar=False,
                    status=JudgeStatus.REQUEST_FAILED,
                    text=repr(error),
                )

        if not completion.choices:
            return PairJudgment(
                row=row,
                column=column,
                is_similar=False,
                status=JudgeStatus.REQUEST_FAILED,
                text=repr(getattr(completion, "error", completion)),
            )

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        is_similar, status = _parse_verdict(text)
        if status is JudgeStatus.PARSE_FAILED and choice.finish_reason == "length":
            status = JudgeStatus.TRUNCATED
        return PairJudgment(row=row, column=column, is_similar=is_similar, status=status, text=text)


async def build_judgment_matrix(
    judge: SimilarityJudge,
    task_prompt: str,
    responses: list[str],
) -> JudgmentMatrix:
    """
    Collects the reciprocal similarity verdicts for one prompt's rollouts.

    Judges each unordered pair once, then re-judges only the reversed direction of
    pairs that came back similar. A pair already called different cannot produce a
    reciprocal edge, so its reverse is never asked.

    Args:
        judge: Similarity judge to issue the calls.
        task_prompt: Prompt the rollouts were generated from.
        responses: List of m subsampled rollout texts.

    Returns:
        `JudgmentMatrix` holding the verdict matrix and every raw reply.
    """
    num_responses = len(responses)
    judgments = torch.zeros((num_responses, num_responses), dtype=torch.bool) # [m, m]

    forward_pairs = [
        (row, column)
        for row in range(num_responses)
        for column in range(row + 1, num_responses)
    ] # C(m, 2)
    forward_judgments = await judge.judge_pairs(task_prompt, responses, forward_pairs) # C(m, 2)

    reverse_pairs = [(j.column, j.row) for j in forward_judgments if j.is_similar]
    reverse_judgments = await judge.judge_pairs(task_prompt, responses, reverse_pairs)

    pair_judgments = [*forward_judgments, *reverse_judgments] # concatenates
    for judgment in pair_judgments:
        judgments[judgment.row, judgment.column] = judgment.is_similar

    return JudgmentMatrix(judgments=judgments, pair_judgments=pair_judgments)


def _parse_verdict(text: str) -> tuple[bool, JudgeStatus]:
    """
    Extracts the verdict from a judge reply.

    Takes the last marker in the reply, so a restated instruction or a phrase used
    mid-reasoning cannot outrank the verdict the judge actually settled on.

    Returns:
        (is_similar, status): Tuple containing
        - is_similar: Whether the judge considers the two responses similar.
        - status: The status of the parsing operation.
    
    Notes:
        A reply carrying neither marker resolves to not similar.
    """
    lowered = text.lower()
    last_similar = lowered.rfind(SIMILAR_MARKER)
    last_different = lowered.rfind(DIFFERENT_MARKER)
    # neither marker was found
    if last_similar == last_different == -1:
        return False, JudgeStatus.PARSE_FAILED
    return last_similar > last_different, JudgeStatus.OK
