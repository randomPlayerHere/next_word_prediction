# Next-Word Prediction with LSTM

This repository demonstrates a simple next-word prediction pipeline using an LSTM-based language model (TensorFlow / Keras). The primary artifact is a Jupyter notebook, `next_word_pred.ipynb`, which walks through loading a text corpus, tokenizing and preparing sequences, defining a model, training, and running a short inference loop to generate the next words.

Contents
- `next_word_pred.ipynb` - Notebook with preprocessing, model training, and inference examples.
- `metamorphosis_clean.txt` - Example plain-text training corpus used in the notebook.

Quick start
1. Create a Python virtual environment and activate it (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (tested with Python 3.10+; adjust TensorFlow version for your platform/GPU):

```bash
pip install --upgrade pip
pip install tensorflow matplotlib numpy jupyter
```

If you have a GPU and want GPU-accelerated training, install a matching `tensorflow` package that supports your CUDA/cuDNN versions (or use `pip install tensorflow` on newer platforms where GPU wheels are included).

Usage
- Launch the notebook server and open `next_word_pred.ipynb`:

```bash
jupyter notebook next_word_pred.ipynb
```

- Walk through each cell. Replace `metamorphosis_clean.txt` with your own text corpus to train a model on different data.

Notes and suggestions
- Vocabulary size: the notebook currently demonstrates a fixed vocabulary size in the model declaration. For robust runs, replace hardcoded sizes with `len(tokenizer.word_index) + 1` and set `input_length = max_len - 1` (since X excludes the final token).
- Checkpoints: add `ModelCheckpoint` and `EarlyStopping` callbacks to save best weights and avoid overfitting.
- Larger datasets and longer training will improve quality, but require more compute. Consider using subword tokenizers (SentencePiece / Byte-Pair Encoding) for rare-word handling.
- For production use, export the tokenizer (pickle or JSON) alongside the model so inference uses the same mapping.

License
MIT

Contact
Open an issue if you find bugs or want help reproducing results.