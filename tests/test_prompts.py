from __future__ import annotations


from pathlib import Path

from experience_bench.core.prompts import render_prompt


def test_render_prompt_does_not_include_input_payload(tmp_path: Path) -> None:
    template_path = tmp_path / "template.md"
    template_path.write_text(
        "{problem_statement}\n\n(No input shown)\n",
        encoding="utf-8",
    )

    secret_input = "VERY_SECRET_INPUT_LINE_1\nVERY_SECRET_INPUT_LINE_2\n"
    rendered = render_prompt(
        template_path=template_path,
        years_experience=10,
        problem_statement="Solve X",
    )

    assert "VERY_SECRET_INPUT_LINE_1" not in rendered.user


def test_render_prompt_errors_if_template_references_input_payload(tmp_path: Path) -> None:
    template_path = tmp_path / "template.md"
    template_path.write_text(
        "{problem_statement}\nInput: {input_payload}\n",
        encoding="utf-8",
    )

    try:
        render_prompt(
            template_path=template_path,
            years_experience=1,
            problem_statement="Solve X",
        )
    except ValueError as e:
        assert "{input_payload}" in str(e)
    else:
        raise AssertionError("Expected ValueError")

