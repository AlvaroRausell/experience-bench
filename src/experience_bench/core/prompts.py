from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptRendered:
    system: str
    user: str
    rendered_sha256: str


def render_prompt(
    *,
    template_path: Path,
    years_experience: int,
    problem_statement: str,
) -> PromptRendered:
    template = template_path.read_text(encoding="utf-8")

    system = (
        f"You are a software engineer with {years_experience} years of experience.\n"
        "Solve the user's programming task one-shot, using your years of experience as a source of expertise.\n"
        "Return only Python code in a single fenced code block.\n"
        "Your program must read from stdin and print exactly two lines:\n"
        "- line 1: Part A answer\n"
        "- line 2: Part B answer\n"
        "Do not print anything else.\n"
    )

    # IMPORTANT: Never pass the actual puzzle input to the model.
    # If a prompt template still contains `{input_payload}`, fail fast with a
    # clear message so the user removes it from the template.
    try:
        user = template.format(
            years=years_experience,
            problem_statement=problem_statement,
        )
    except KeyError as e:
        missing = str(e).strip("'")
        if missing == "input_payload":
            raise ValueError(
                "Prompt template references {input_payload}, but this benchmark tool "
                "never includes the puzzle input in the LLM prompt. Remove the input "
                "section/placeholder from the template."
            ) from e
        raise

    import hashlib

    sha = hashlib.sha256((system + "\n\n" + user).encode("utf-8")).hexdigest()
    return PromptRendered(system=system, user=user, rendered_sha256=sha)
