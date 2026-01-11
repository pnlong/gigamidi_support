# Plan: Predicting Continuous Valence/Arousal Values for GigaMIDI-Extended

## Overview

> If possible, our preference would be to compute Valence and Arousal at the bar level for the GigaMIDI dataset.

I have a dataset, GigaMIDI. I want to use MuseTok, a symbolic music tokenization framework (which I have git cloned into `../musetok`), to extract latents for each bar of music in GigaMIDI. That is, for each song in GigaMIDI, for each bar within each song, get a latent vector for each bar -- because MuseTok uses VQ-VAE, my understanding is that the dimension of the latent vector is equal to the number of codebook levels? So one question is how many codebook level should I use, and do I use the codes for downstream task or do I use the latent vectors themselves associated with each code.

Anyways, I plan on using these MuseTok latents for an emotion recognition task. Previously, I did something similar (see `../jingyue_latents`), where I did emotion recognition for the MuseTok paper to predict categorical classes using the EMOPIA dataset. The setup was simply an MLP trained with Cross Entropy Loss on MuseTok latents for EMOPIA for the *categorical* emotion class prediction task. However, in that case, the predictions were categorical. Now, I am more interested in *continuous* predictions -- Valence and Arousal on a scale of -1 to 1 for both -- at the bar level. So we could easily modify the original MLP and train it with some different loss criterion to output continuous values as desired.

Broadly, the plan would be that we need to pre-train this MLP that outputs continuous values for the various characteristics we want on EMOPIA. As input to this MLP, we would need to use some pre-trained version of MuseTok to extract the latents. So for each bar, we would predict the relevant continuous characteristics, which would then yield valence/arousal at the bar level for each song. So down the line, you could use those bar level valence/arousal values to plot how the classified emotion changes throughout a song as a line (i.e., time-varying overall valence and arousal curves).

This is a note from my collaborator:

> After we extend the GigaMIDI dataset, the next step will be to train a model that supports fine-grained control via conditioning on Valence, Arousal, and Tension (tension is something else though, and someone else will work on that, so this is not relevant to us). Having continuous targets should also make it easier for the model to learn a meaningful latent space for these attributes.



## Packages

We will need to create a new `mamba` environment for this project.

```
mamba create -n gigamidi python=3.10
```

### Symusic

Quick note on implementation: for parsing MIDI files in GigaMIDI, we use the `Symusic` library for consistency and significantly faster processing compared to PrettyMIDI or Mido: [https://github.com/Yikai-Liao/symusic]

This also helps avoid downstream usability issues, since different MIDI parsers can yield slightly different interpretations (e.g., track counting, channel handling), which could introduce inconsistencies for users later on.

Symusic can be installed with:

```
pip install symusic
```

### PyTorch

We will use PyTorch to create and train the MLP for this project. We will need to create a PyTorch dataset class to load the latents (that can be used in a data loader). Note that this means we will probably want to preprocess the latents for EMOPIA so we don't have to extract them on the fly. Then we need a training pipeline to train on the EMOPIA latents.

### gdown

I will need to download the MuseTok best checkpoints, since we are using pre-trained MuseTok. The checkpoint is stored as a ZIP file hosted on google drive, and I will need to download it. This is what the `gdown` package is for:

```
pip install gdown
```

For example, you can download any file from Google Drive by running one of these commands:

```
gdown https://drive.google.com/uc?id=<file_id>  # for files
gdown <file_id>                                 # alternative format
gdown --folder https://drive.google.com/drive/folders/<file_id>  # for folders
gdown --folder --id <file_id>                                   # this format works for folders too
```



## Using MuseTok

We can find the MuseTok code cloned in `../musetok`. Please reference that code as needed to see what various files do. But one thing to note is that MuseTok might provide some way to extract MuseTok RVQ codes (a.k.a. latents) directly given a MIDI file, but this pipeline may not use `symusic`, so we might have to write another similar pipeline to do the same thing, but using `symusic` to load the MIDI. MuseTok, I'm pretty sure, offers ways to convert from REMI encoding to MuseTok RVQ codes, so we basically only need someway to convert `symusic` MIDI scores into REMI encoding, at which point MuseTok can take over.

### Environment

* Python 3.10 and `torch==2.5.1` used for the experiments
* Install dependencies

```
pip install -r requirements.txt
```

### Quick Start

Download and unzip [best weights](https://drive.google.com/file/d/1HK534lEVdHYl3HMRkKvz8CWYliXRmOq_/view?usp=sharing) in the root directory. 

#### Music Generation

Generate music pieces by continuing the prompts from our test sets with the two-stage music generation framework:

```
python test_generation.py \
        --configuration=config/generation.yaml \
        --model=ckpt/best_generator/model.pt \
        --use_prompt \
        --primer_n_bar=4 \
        --n_pieces=20 \
        --output_dir=samples/generation
```

Or, generate music pieces from scratch:

```
python test_generation.py \
        --configuration=config/generation.yaml \
        --model=ckpt/best_generator/model.pt \
        --n_pieces=20 \
        --output_dir=samples/generation
```


### Train the model

#### Data Preparation
Download the datasets used in the paper (to be released) and unzip in the root directory `MuseTok`. To train with customized datasets, please refer to the [steps](https://github.com/Yuer867/MuseTok/tree/main/data_processing#readme).

#### Music Tokenization

Train a music tokenization model from scratch:

```
python train_tokenizer.py config/tokenization.yaml
```

Test the reconstruction quality with music pieces in the test sets:

```
python test_reconstruction.py config/tokenization.yaml ckpt/best_tokenizer/model.pt samples/reconstruction 20
```

#### Music generation

1. Encode REMI sequences to RVQ tokens offline with data augmentation. Skip this step if you would like to use the tokens encoded with provided tokenizer weights for training **and** have downloaded the datasets in the `Data Preparation` step. 

```
python remi2tokens.py config/remi2tokens.yaml ckpt/best_tokenizer/model.pt
```

2. Train a music generation model with learn tokens.

```
python train_generator.py config/generation.yaml
```

3. Generate music pieces with new checkpoints.

```
python test_generation.py \
        --configuration=config/generation.yaml \
        --model=ckpt/best_generator/model.pt \  # change the checkpoints here
        --use_prompt \
        --primer_n_bar=4 \
        --n_pieces=20 \
        --output_dir=samples/generation
```

### MuseTok Data Preparation

This folder provides the scripts of converting midi data into REMI event sequences, building vocabulary and splitting training / validation / test sets. 

Alternatively, you can download processed datasets in [link] for quickly training and inference.

#### Convert MIDI to events

```
python data_processing/midi2events.py
```

#### Build Vocabulary

```
python data_processing/events2words.py
```

#### Data Splits

```
python data_processing/data_split.py
```

### Using MuseTok in Our Pipeline

We need some way that can take either a MIDI file or the raw bytes sequence of a MIDI file (the former for EMOPIA, the latter for GigaMIDI), and then converting it into some format that MuseTok can process, and then output the latents for bar-by-bar. I think we should carefully investigate `../musetok/data_processing` directory, and then create our own version of that ourselves. It seems like the flow is:

```
midi2events.py --> events2words.py --> data_split.py
```

but `midi2events.py` uses `miditoolkit` instead of `symusic`, so we would need to rewrite this to use `symusic` instead. But then the flow should be the same. We get the events, then convert events to words with `events2words.py`. Then, for EMOPIA, we use the `split_emopia` function in `data_split.py` to get the relevant training/validation/testing partitions for training our emotion recognition model. But anyways, the core of our pipeline is loading a MIDI file with `symusic` similar to `midi2events.py`, and then converting those events to words with `events2words.py`, and then we should be able to feed those "words" into MuseTok just fine, and then get the latents. Those latents will then be used as input for our emotion recognition system (where we can either choose to save that latents tensor to disk in the case of preprocessing, or just use it).



## Downloading EMOPIA

This is something I need to do, but I need to figure out if I have access to EMOPIA, and if not, I will download it. I need to check on this. And once we download it, we need to figure out its structure so that we can get input/output pairs of MIDI files and continuous Valence/Arousal values as outputs that we must predict.



## Extracting Latents with MuseTok

We will likely preprocess EMOPIA using MuseTok, where are MIDI file to MuseTok latents pipeline is described in the previous section. We will store the MuseTok latents as `.safetensors` or `.npy` pickle files, so that we don't have to use MuseTok on the fly when training our emotion recognition system.

In the case of GigaMIDI, where we are just using MuseTok to get the latents for inference, and we don't need to use the latents for a single song more than once, we can probably use MuseTok on the fly and use the pipeline without storing any intermediate tensors. This way, we avoid storing all the preprocessed latents for GigaMIDI to disk (which would take up an insane amount of storage space), and in fact, if we are doing it that way, we can just employ GigaMIDI using the `datasets` package with `streaming=True` to avoid even downloading the whole dataset locally.



## Continuous Valence/Arousal Prediction

We previously did an emotion recognition task on EMOPIA using MuseTok latents, which can be found in `../jingyue_latents`. However, this yields `categorical` emotion class outputs, instead of continuous values for valence/arousal. So we would need to modify a couple things to get this right. Note that we wrote that codebase for a variety of music information retrieval tasks, whereas now we only care about the emotion recognition task.

### Model

The model we used can be found as `CustomMLP` in `../jingyue_latents/model.py`. We would need to modify the number of output dimensions.

### Dataset

We can reuse the custom PyTorch dataset from `../jingyue_latents/dataset.py`, which loads MuseTok latents. We need to figure out how to get valence/arousal labels that we can predict.

### Training

We would need to update the training regime in `../jingyue_latents/train.py` and use a different loss criterion that isn't cross entropy loss for predicting continuous values.



## Annotating GigaMIDI

We need to load in the GigaMIDI-extended dataset (`v2.0.0`, hoping to switch to `v3.0.0` soon) so that we can run our continuous valence/arousal prediction pipeline on each bar of each song. This step should deliver continuous valence/arousal classifications at the bar level for each song in the dataset.

### How to use

The `datasets` library allows you to load and pre-process the GigaMIDI dataset in pure Python at scale. The dataset can be downloaded and prepared in one call to your local drive by using the `load_dataset` function.

```python
from datasets import load_dataset

dataset = load_dataset("Metacreation/GigaMIDI")
```

Using the `datasets` library, you can also stream the dataset on-the-fly by adding a `streaming=True` argument to the `load_dataset` function call. Loading a dataset in streaming mode loads individual samples of the dataset at a time, rather than downloading the entire dataset to disk.

```python
from datasets import load_dataset

dataset = load_dataset("Metacreation/GigaMIDI", split="train", streaming=True)

data_sample = next(iter(dataset))
```

Bonus: create a `PyTorch` dataloader directly with your own datasets (local/streamed).

#### Local

```python
from datasets import load_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

dataset = load_dataset("Metacreation/GigaMIDI", split="train")
batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=32, drop_last=False)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
```

#### Streaming

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("Metacreation/GigaMIDI", split="train")
dataloader = DataLoader(dataset, batch_size=32)
```

### Example scripts

MIDI files can be easily loaded and tokenized with `Symusic` and `MidiTok` respectively.

```python
from datasets import load_dataset
from miditok import REMI
from symusic import Score

dataset = load_dataset("Metacreation/GigaMIDI", split="train")
tokenizer = REMI()
for sample in dataset:
    score = Score.from_midi(sample["music"])
    tokens = tokenizer(score)
```

The dataset can be processed by using the `dataset.map` and `dataset.filter` methods.

```python
from pathlib import Path
from datasets import load_dataset
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.utils import get_bars_ticks
from symusic import Score

def is_score_valid(
    score: Score | Path | bytes, min_num_bars: int, min_num_notes: int
) -> bool:
    """
    Check if a ``symusic.Score`` is valid, contains the minimum required number of bars.

    :param score: ``symusic.Score`` to inspect or path to a MIDI file.
    :param min_num_bars: minimum number of bars the score should contain.
    :param min_num_notes: minimum number of notes that score should contain.
    :return: boolean indicating if ``score`` is valid.
    """
    if isinstance(score, Path):
        try:
            score = Score(score)
        except SCORE_LOADING_EXCEPTION:
            return False
    elif isinstance(score, bytes):
        try:
            score = Score.from_midi(score)
        except SCORE_LOADING_EXCEPTION:
            return False

    return (
        len(get_bars_ticks(score)) >= min_num_bars and score.note_num() > min_num_notes
    )

dataset = load_dataset("Metacreation/GigaMIDI", split="train")
dataset = dataset.filter(
    lambda ex: is_score_valid(ex["music"], min_num_bars=8, min_num_notes=50)
)
```

### GigaMIDI Entry Fields

The GigaMIDI metadata schema defines the following fields for each entry:

- `split` (`string`): The data split (train, validation, or test).
- `md5` (`string`): MD5 hash of the MIDI file, corresponding to its file name.
- `music` (`bytes`): Raw MIDI bytes to be loaded with an external library (e.g., Symusic).
- `NOMML` (`List[int]`): Note Onset Median Metric Level for each track.
- `num_tracks` (`int`): Number of tracks in the MIDI file.
- `TPQN` (`int`): Ticks per quarter note.
- `total_notes` (`int`): Total number of note events.
- `avg_note_duration` (`float`): Average note duration (in ticks).
- `avg_velocity` (`float`): Average MIDI velocity.
- `min_velocity` (`int`): Minimum velocity value.
- `max_velocity` (`int`): Maximum velocity value.
- `tempo` (`string`): Tempo metadata from the MIDI file.
- `loop_track_idx` (`List[int]`): Indices of tracks where loops were detected.
- `loop_instrument_type` (`List[string]`): Instrument types for each detected loop.
- `loop_start` (`List[int]`): Start tick of each loop.
- `loop_end` (`List[int]`): End tick of each loop.
- `loop_duration_beats` (`List[float]`): Duration of each loop in beats.
- `loop_note_density` (`List[float]`): Note density (notes per beat) within each loop.
- `Type` (`string`): Type indicator for the MIDI file.
- `instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2` (`int`): Instrument-category code (0 = drums-only, 1 = all-instruments-with-drums, 2 = no-drums).
- `music_styles_curated` (`List[string]`): Curated music style labels.
- `music_style_scraped` (`string`): Music style scraped from external sources.
- `music_style_audio_text_Discogs` (`List[string]`): Styles from Discogs audio-text matching.
- `music_style_audio_text_Lastfm` (`List[string]`): Styles from Last.fm matching.
- `music_style_audio_text_Tagtraum` (`List[string]`): Styles from Tagtraum matching.
- `title` (`string`): Track title.
- `artist` (`string`): Artist name.
- `audio_text_matches_score` (`float`): Audio-text matching score.
- `audio_text_matches_sid` (`List[string]`): Matched Spotify IDs.
- `audio_text_matches_mbid` (`List[string]`): Matched MusicBrainz IDs.
- `MIDI_program_number__expressive_` (`List[int]`): Program numbers for expressive loops.
- `instrument_group__expressive_` (`List[string]`): Instrument groups for expressive loops.
- `start_tick__expressive_` (`List[int]`): Start ticks for expressive loops.
- `end_tick__expressive_` (`List[int]`): End ticks for expressive loops.
- `duration_beats__expressive_` (`List[float]`): Durations (in beats) of expressive loops.
- `note_density__expressive_` (`List[float]`): Note density of expressive loops.
- `loopability__expressive_` (`List[float]`): Loopability scores for expressive loops.

### GigaMIDI Versions

GigaMIDI is currently released with two versions:

1. `v1.0.0`: the original version of GigaMIDI
2. `v2.0.0`: the current release of GigaMIDI, called GigaMIDI-extended, which contains a lot more music
3. `v3.0.0`: the next release of GigaMIDI, which will include the pruned version of the AriaMIDI dataset

We want to use `v2.0.0` for now, and switch to `v3.0.0` when it is released.

### Running out Pipeline on GigaMIDI

We probably want to use some sort of batching/multiprocessing techniques to speed things up, since GigaMIDI is quite large. Certainly we can use a large batch size to predict the continuous values given MuseTok latents to speed things up. But I'd be curious about what is the bottleneck -- preprocessing with MuseTok latents, or classifying with the MLP. I suspect the bottleneck may be the former. But I will say we probably want to use the `streaming=True` approach for GigaMIDI and avoiding downloading the dataset in full at all costs.



## Conclusion

The various sections of this plan describe our pipeline for predicting continuous valence/arousal values for GigaMIDI-Extended. We will write all our code ini `../valence_arousal`.

This is a rough draft. I want a more refined draft that outlines the actual steps in our pipeline. Then, when organizing our code in `../valence_arousal`, we should try to group different scripts for different parts of our pipeline into different subdirectories (i.e., if there is some script that helps preprocess EMOPIA, there would be a subdirectory called `../valence_arousal/pretrain_model`, and there would presumably be something like `../valence_arousal/process_gigamidi`). But there are inevitably some files that bridge different parts of the pipeline, and those would be in `../valence_arousal.`

As a bonus, it would be nice to have one script that actually calculates some statistics about the predicted valence/arousal values in GigaMIDI -- perhaps certain genres or tags tend to have different levels for these characteristics? This would be another subdirectory, `../valence_arousal/analyze_emotion_annotations`

