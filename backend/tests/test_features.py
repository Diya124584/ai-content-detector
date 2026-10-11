import math

from app.features import (FEATURE_NAMES, burstiness, extract_features, feature_vector, preprocess_text,
                          repetition_ratio, split_sentences, tokenize_words, type_token_ratio)
from .helpers import SAMPLE


def test_empty_input_returns_zero_features():
    f = extract_features("")
    assert f.word_count == 0 and f.sentence_count == 0 and f.char_count == 0
    assert f.ttr == 0.0 and f.burstiness == 0.0


def test_non_string_and_whitespace_input_do_not_crash():
    assert extract_features(None).word_count == 0
    assert extract_features("   \n\t ").word_count == 0


def test_punctuation_only_input():
    f = extract_features("...!!! ???")
    assert f.word_count == 0 and f.sentence_count == 0
    assert f.punctuation_ratio == 1.0


def test_very_short_input():
    f = extract_features("Hi")
    assert f.word_count == 1 and f.sentence_count == 1 and f.ttr == 1.0 and f.burstiness == 0.0


def test_ttr_definition():
    assert type_token_ratio(["a", "b", "a", "b"]) == 0.5
    assert type_token_ratio([]) == 0.0
    assert extract_features("one two three four").ttr == 1.0


def test_burstiness_definition():
    assert burstiness([5, 5, 5]) == 0.0  # perfectly uniform
    # lengths 2 and 6: mean 4, population std 2 -> 0.5
    assert math.isclose(burstiness([2, 6]), 0.5)
    assert burstiness([]) == 0.0


def test_burstiness_from_text():
    uniform = extract_features("One two three. Four five six. Seven eight nine.")
    varied = extract_features("Go. This sentence is a fair bit longer than the previous one was. Ok.")
    assert uniform.burstiness == 0.0
    assert varied.burstiness > uniform.burstiness


def test_repetition_ratio():
    assert repetition_ratio("a b c d e f".split(), 3) == 0.0
    assert repetition_ratio("a b c a b c a b c".split(), 3) > 0.4
    assert repetition_ratio(["a"], 3) == 0.0


def test_sentence_split_and_tokenize():
    assert len(split_sentences("Hello there. How are you? Fine!\nNew line")) == 4
    assert tokenize_words("Don't stop-believing, 42 times") == ["don't", "stop", "believing", "42", "times"]


def test_long_input_is_handled():
    f = extract_features(SAMPLE * 500)
    assert f.word_count > 10000 and math.isfinite(f.ttr)


def test_feature_vector_matches_names():
    v = feature_vector(SAMPLE)
    assert len(v) == len(FEATURE_NAMES) and all(math.isfinite(x) for x in v)


def test_preprocess_strips_control_chars():
    assert preprocess_text("a\x00b   c") == "a b c"
