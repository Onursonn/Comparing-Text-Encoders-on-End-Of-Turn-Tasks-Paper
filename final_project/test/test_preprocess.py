from final_project.src.preprocess import (
    build_input_text,
    normalize,
    pad_sequence,
    parse_ftad_line,
    tokenize,
)


def test_normalize_strips_punctuation() -> None:
    assert normalize("Hello, world?!", mode="normalized") == "hello world"


def test_tokenize_matches_course_pattern() -> None:
    assert tokenize("Hello, world!") == ["hello", ",", "world", "!"]


def test_pad_sequence() -> None:
    assert pad_sequence([2, 3, 4], max_length=6, pad_id=0) == [2, 3, 4, 0, 0, 0]
    assert pad_sequence([2, 3, 4, 5, 6], max_length=3, pad_id=0) == [2, 3, 4]


def test_build_input_text_without_context() -> None:
    text = build_input_text("agent:hi|user:hello", "okay", use_context=False)
    assert text == "okay"


def test_parse_ftad_line() -> None:
    ex = parse_ftad_line("ctx\tutt\t0\n")
    assert ex.context == "ctx"
    assert ex.utterance == "utt"
    assert ex.label == 0
