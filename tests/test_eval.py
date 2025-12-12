from experience_bench.core.eval import eval_two_line_stdout


def test_eval_two_line_stdout_ok():
    r = eval_two_line_stdout(stdout="1\n2\n", expected_a="1", expected_b="2")
    assert r.passed_all


def test_eval_two_line_stdout_parse_error():
    r = eval_two_line_stdout(stdout="1\n", expected_a="1", expected_b="2")
    assert r.error_type == "output_parse_error"
