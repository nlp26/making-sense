import builtins
import pytest
from tokenizer import spacy_tokenizer


def test_simple_sentence():
    tokens = spacy_tokenizer("Hello world! This is a test.")
    assert isinstance(tokens, list)
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_numbers_and_punctuation_removed():
    tokens = spacy_tokenizer("Numbers 123 and symbols #! aren't kept.")
    assert all(token.isalpha() for token in tokens)
