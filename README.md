# Seq2Seq

This is my attempt at recreating the seq2seq paper by Sutskever et al. I learned the core concepts of RNNs and LSTMs as I went along, and while I was considering using the LSTMs I wrote myself for this training pipeline, I decided to study LSTMs on my own and use PyTorch's optimized `nn.LSTM` module for training the model (in order to get the best out of both worlds).

## What this does

I'm translating German sentences to English using a sequence-to-sequence model. The idea is pretty simple - I have an encoder that reads German words and understands what they mean, then a decoder that spits out English words one by one. I trained it on the WMT19 dataset which has tons of German-English sentence pairs. Classic seq2seq stuff.

## How I set this up

I organized everything into logical pieces:

- The neural network models live in `src/models/` (where I tried coding the architecture)
- Data handling stuff is in `src/data/` (downloading and processing the data)
- Training scripts are in `scripts/` (training the model, visualizing the training progress, etc.)
- I save model checkpoints in `checkpoints/` (not pushed remotely since it's a lot of data, but this directory will be automatically created once you run the training script)
- The visualizations will be saved in `training_plots/` (the plot automatically updates as the model trains)

## Running this yourself

First, install the dependencies:

```bash
pip install -e .

# or if you wanna just do uv
uv pip install -r requirements.txt

# if there are some installations missing despite this, just run `uv pip install {said  package}`
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

This step is crucial because the tokenizers build vocabularies based on the actual words in your training data. If you use a smaller dataset but keep tokenizers built for the full dataset, you'll have vocabulary mismatches. I learned this the hard way when my model kept seeing unknown tokens everywhere.

Currently, the tokenization builds separate vocabularies for German (max 30k words) and English (max 25k words) from the actual training pairs. This is in relation to my dataset of size 2M (the original WMT19 dataset's size is 35M), and if you decide to either increase or decrease the size of your dataset in the `data_preparation.py` file, you might want to update these vocabulary sizes as well.

Once you confirm the tokenization is done, start training:

```bash
python scripts/train.py
```

Training could take several hours depending on your hardware. I set it up to work with CUDA if you have a GPU, MPS if you're on a Mac with Apple Silicon, or just CPU if that's all you've got.

You can watch the training progress in real-time. I print out loss values and save plots showing how well it's learning. The model gets saved automatically when it improves.

## Playing with it

Once training is done, you can experiment with different sentences. The model is saved in the checkpoints directory. I included visualization tools that show how the loss decreases over time.

If you want to see what the model learned, check out the training plots it generates. You can watch the validation loss go down (hopefully) and see when the model starts overfitting.

## Why I built this

I wanted to do a whole end-to-end project in terms of building the architecture of the model, downloading the dataset, creating a proper data loader and actually training the whole thing and seeing outcomes. And this was the perfect project for me to do all of it. A very fun project for learners like myself.
