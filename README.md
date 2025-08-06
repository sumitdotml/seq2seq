# Seq2Seq

This is my attempt at recreating the seq2seq paper by Sutskever et al. I learned the core concepts of RNNs and LSTMs as I went along, and while I was considering using the LSTMs I wrote myself for this training pipeline, I decided to study LSTMs on my own and use PyTorch's optimized `nn.LSTM` module for training the model (in order to get the best out of both worlds).

## What this does

I'm translating German sentences to English using a sequence-to-sequence model. The idea is pretty simple - I have an encoder that reads German words and understands what they mean, then a decoder that spits out English words one by one. I trained it on the WMT19 dataset which has tons of German-English sentence pairs. Classic seq2seq stuff.

## Quick Start

If you want to skip straight to testing translations, you can use my pre-trained model (well, it's not that great since it was trained on a subset of the WMT19 dataset, but it's a good starting point):

```bash
# 1. Create and activate virtual environment
# Option a: Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Option b: Using standard Python
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
uv pip install -r requirements.txt
# Or use regular pip: pip install -r requirements.txt

# 3. Download pre-trained model (one-time setup)
python scripts/download_pretrained.py

# 4. Start translating!
python scripts/inference.py --interactive
```

I trained the model on 2M sentence pairs, so expect some basic to slightly unexpected translations and some `<UNK>` tokens for words not in the 30k German / 25k English vocabularies.

## Preparing and Training the Model

First, set up your environment:

```bash
# 1. Create and activate virtual environment
# Option a: Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Option b: Using standard Python
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
uv pip install -r requirements.txt
# Or use regular pip: pip install -r requirements.txt

# Alternative: Install in editable mode
uv pip install -e .
# Or: pip install -e .
```

Then run the data preparation to download and process the WMT19 dataset:

```bash
python scripts/data_preparation.py
```

This takes a while because it downloads a big dataset and processes it. I filter out sentences that are too long or too short to make training easier.

**Important note about tokenization:** After preparing your data, you need to build tokenizers that match your dataset. The existing tokenizer files in the data directory were built for a specific dataset size. If you changed the dataset size in `data_preparation.py`, you need to rebuild the tokenizers:

```bash
python src/data/tokenization.py
```

This step is crucial because the tokenizers build vocabularies based on the actual words in your training data. If you use a smaller dataset but keep tokenizers built for the full dataset, you'll have vocabulary mismatches.

Currently, the tokenization builds separate vocabularies for German (max 30k words) and English (max 25k words) from the actual training pairs, which is not a lot. This is in relation to my dataset of size 2M (the original WMT19 dataset's size is 35M).

If you have compute that you can rely on, I recommend training on the full WMT19 dataset (35M sentence pairs). You can do this by:
- setting `use_full_dataset=True` in [data_preparation.py](scripts/data_preparation.py#L133-134) to prepare the full WMT19 dataset (35M sentence pairs)
- tweaking the hyperparameters in [train.py](scripts/train.py#L30-L39)
- update the DE-EN vocabulary sizes in [tokenization.py](src/data/tokenization.py#L138-144)

Once you confirm the tokenization is done, start training:

```bash
python scripts/train.py
```

Training could take several hours depending on your hardware. I set it up to work with CUDA if you have a GPU, MPS if you're on a Mac with Apple Silicon, or just CPU if that's all you've got.

You can watch the training progress in real-time. I print out loss values and save plots showing how well it's learning. The model gets saved automatically when it improves.

Once you are done training, run the demo:

```bash
python scripts/inference.py
```

### Translating Your Own Sentences

**Single sentence translation:**

```bash
python scripts/inference.py --sentence "Guten Morgen!" --verbose
```

The `--verbose` flag shows you the tokenization process and model's internal workings.

**Interactive mode:**
```bash
python scripts/inference.py --interactive
```

This starts an interactive session where you can type German sentences and get English translations in real-time.

## Why I built this

I wanted to do a whole end-to-end project in terms of building the architecture of the model, downloading the dataset, creating a proper data loader and actually training the whole thing and seeing outcomes. And this was the perfect project for me to do all of it. A very fun project for learners like myself.