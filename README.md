# Fine-Grained Named Entities for Corona News

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flair](https://img.shields.io/badge/Flair-NLP-0F766E?logoColor=white)](https://github.com/flairNLP/flair)
[![PyTorch](https://img.shields.io/badge/PyTorch-framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

This repository contains a fine-grained named entity recognition (NER) model for coronavirus-related news. It uses [Flair](https://github.com/flairNLP/flair) sequence taggers and a BIO-formatted corpus derived from Tagesschau news data.

The work was presented at the [SWAT4HCLS 2023 conference](https://repository.publisso.de/resource/frl%3A6440380) in Basel, Switzerland, on February 15, 2023.

## What it recognizes

In addition to common entity types such as `PERSON`, `ORG`, `GPE`, `DATE`, `CARDINAL`, and `PERCENT`, the corpus includes domain-specific labels such as:

- `CORONAVIRUS`
- `DISEASE_OR_SYNDROME`
- `GROUP`
- `FAC`
- `PRODUCT`

The complete label set is defined by the corpus tag dictionary when a model is trained.

## Quick start: run inference

Install Flair in a Python environment:

```bash
python -m pip install flair
```

Download the trained model from [Google Drive](https://drive.google.com/file/d/1R6WVbynZK81J_aeBkRyTu2koHTf4hTpF/view?usp=sharing), extract it, and pass the path to `SequenceTagger.load`:

```python
from flair.data import Sentence
from flair.models import SequenceTagger

model = SequenceTagger.load("/path/to/model/best-model.pt")
sentence = Sentence(
  "Lauterbach: Omicron is not suitable as a vaccine substitute. "
  "Federal Health Minister Karl Lauterbach refers to a study from South Africa."
)

model.predict(sentence)
for entity in sentence.get_spans("ner"):
  print(entity)
```

Example predictions include `Omicron -> CORONAVIRUS`, `Karl Lauterbach -> PERSON`, `South Africa -> GPE`, and `Twitter -> ORG`. Predictions and confidence scores depend on the downloaded model and input text.

## Dataset format

The training scripts expect a corpus directory containing three files:

```text
corpus/
├── train.txt
├── dev.txt
└── test.txt
```

Each file uses one token and one BIO tag per line. Blank lines separate sentences:

```text
Schleswig-Holstein    B-GPE
,                      O
vaccine                B-FAC
centers                I-FAC

```

The checked-in example data is under [`data/tagesschau/test`](data/tagesschau/test). The training scripts require `train.txt` and `dev.txt` alongside `test.txt`; provide those splits before starting a training run.

## Train a model

The scripts accept the same positional arguments:

```text
python script.py CORPUS_PATH OUTPUT_PATH LEARNING_RATE BATCH_SIZE EPOCHS DOWN_SAMPLING
```

For example:

```bash
python src/model/base_model_train.py \
  data/tagesschau  \
  models/glove  \
  0.1 32 10 1.0
```

Available training variants are:

| Script | Description |
| --- | --- |
| [`base_model_train.py`](src/model/base_model_train.py) | Flair sequence tagger with GloVe embeddings |
| [`model_train.py`](src/model/model_train.py) | Sequence tagger with GloVe and Flair embeddings |
| [`finetuning_hunflair.py`](src/model/finetuning_hunflair.py) | Fine-tunes the `hunflair-disease` tagger |
| [`finetuning_ontonotes.py`](src/model/finetuning_ontonotes.py) | Fine-tunes `flair/ner-english-ontonotes` |

Training downloads the embeddings or base model used by the selected script. The required resources and GPU support are managed by Flair and PyTorch; larger runs may require substantial memory.

## Demo and analysis

- Interactive demo: [`A_NER_Model_for_Corona__News.ipynb`](src/viz/A_NER_Model_for_Corona__News.ipynb)
- Inter-annotator agreement calculation: [`fleiss_kappa.py`](src/analysis/fleiss_kappa.py)

## Repository layout

```text
data/       Annotated Tagesschau data
src/model/  Model training and fine-tuning scripts
src/analysis/Agreement analysis
src/plots/  Plotting utilities
src/viz/    Notebook demo
```

## Citation and links

For the presentation, see the [SWAT4HCLS 2023 record](https://repository.publisso.de/resource/frl%3A6440380). The trained model is available [here](https://drive.google.com/file/d/1R6WVbynZK81J_aeBkRyTu2koHTf4hTpF/view?usp=sharing).
