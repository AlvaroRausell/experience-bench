from experience_bench.core.code_extract import extract_first_code_block


def test_extract_first_code_block():
    text = "hello\n```python\nprint('x')\n```\nbye"
    code = extract_first_code_block(text)
    assert code.strip() == "print('x')"
