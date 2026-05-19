# making-sense

A sandbox for keeping common ML/NLP frameworks fresh. Each script in `examples/` is the smallest runnable thing for that framework. Each notebook in the root is a deeper task using one or more of them.

## examples/

Standalone scripts. Run any one with `python examples/<name>.py`.

- `simple_nn.py` — neural network from scratch in NumPy.
- `pytorch_example.py` — basic model in PyTorch.
- `sklearn_example.py` — logistic regression with scikit-learn.
- `tensorflow_example.py` — Keras model in TensorFlow.
- `huggingface_example.py` — sentiment analysis with a pretrained Transformer.
- `agentic_ai_example.py` — minimal agent on the OpenAI SDK.

## Notebooks (root)

- `brush_ML_up.ipynb` — refresher exercises across the core ML toolkit.
- `fuzzy.ipynb` — fuzzy matching / approximate string search.
- `movie-nlp.ipynb` — text classification on a movie dataset.
- `tweet_sentiment.ipynb` — sentiment pipeline on tweet-shaped text.
- `txtai_semantic.ipynb` — semantic search with `txtai`.
- `tokenizer.py` — small tokenizer helper used by some of the notebooks.

## Run

```bash
pip install numpy scikit-learn torch tensorflow transformers openai txtai
python examples/<name>.py
# or
jupyter notebook
```

Tests under `tests/` are runnable with `pytest`.

## License

Apache-2.0
