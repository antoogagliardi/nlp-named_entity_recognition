# NLP - Named Entity Recognition (NER)

A PyTorch-based Named Entity Recognition system built on a word-embedding + LSTM architecture, trained on a BIO-tagged dataset.

# 📋 Table of Contents
 
* [Overview](#-overview)
* [Repository Structure](#-repository-structure)
* [Data](#-data)
* [Requirements](#-requirements)
* [Training](#-training)
* [Usage](#-usage)


## 🔎 Overview

Given a tokenized sentence, the model predicts a named-entity tag for each token using the **BIO tagging scheme** (`B-` = beginning of entity, `I-` = inside entity, `O` = outside any entity). The pipeline covers the full workflow: data loading and preprocessing, dictionary/vocabulary construction, model training with checkpointing, evaluation, and deployment behind a lightweight HTTP API.

### Model Architecture

- **Embedding layer** — maps vocabulary words to dense vectors (`word_embeddings`)
- **LSTM layer** — processes the embedded sequence to produce contextual hidden states
- **Linear layer** — projects hidden states to tag-space logits
- **Log-softmax + NLL loss** — used for training, with the padding index ignored

Training hyperparameters (see `code/train.py`) include a sliding-window sequence construction, an embedding dimension of 512, a hidden dimension of 256, batch size 64, and SGD optimization.


## 🗂 Repository Structure

```
nlp-named_entity_recognition/
├── code/                       # Training code
│   ├── NLP_homework_1a.ipynb
│   ├── train.py                # Main training script
│   ├── ckpt/
│   ├── cm/
│   └── src/
│       ├── data.py             # DataManager, DataEncoder, NERDataset
│       ├── dict.py             # Vocabulary/tag dictionary builder
│       ├── model.py            # NERNetwork model + Trainer
│       ├── plot.py
│       └── utils.py
├── data/
│   ├── train.tsv                # Training data (BIO-tagged)
│   └── dev.tsv                  # Development/validation data
├── model/
│   └── dicts/                   # Serialized word2id / id2word / tag2id / id2tag dictionaries
├── docker/
│   ├── app.py
│   ├── evaluate.py
│   ├── simple_test.py
│   ├── model.py
│   └── src/
│       └── implementation.py
├── logs/
├── Dockerfile                   # Container definition for the inference
├── requirements.txt             # Python dependencies
├── test.sh                      # Build, run, and evaluate through Docker
└── report.pdf                   # Written project report
```


## 📊 Data

Data is store as tab-separated files where each sentence is preceded by an header line where the field `#` denotes the single words composing the sentence and for each of them the field `id` denotes the BIO-tag assigned:

|  #              | id     |
|-----------------|--------|
| `it`            | O      |
| `lies`          | O      |
| `approximately` | O      |
| `north`         | O      |
| `east`          | O      |
| `of`            | O      |
| `bolesławiec`   | B-LOC  |
| `,`             | O      |
| `and`           | O      |
| `west`          | O      |
| `of`            | O      |
| `the`           | O      |
| `regional`      | O      |
| `capital`       | O      |
| `wrocław`       | B-LOC  |
| `.`             | O      |
|                 |        |

- **`data/train.tsv`** — training set (~14.5k sentences)
- **`data/dev.tsv`** — development/validation set (~765 sentences)

### Entity Types

The dataset used enables the model to recognize the following entity classes:

| Tag    | Entity Type      |
|--------|------------------|
| `PER`  | Person           |
| `LOC`  | Location         |
| `GRP`  | Group            |
| `CORP` | Corporation      |
| `PROD` | Product          |
| `CW`   | Creative Work    |
|        |                  |

Each type appears with a `B-` (beginning) and `I-` (inside) prefix, plus the `O` (outside) tag for non-entity tokens.


## 🛠️ Requirements

* Ubuntu distribution: either 20.04 or the current LTS (22.04) are perfectly fine.
* [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community.

### Setup Environment

To evaluate the final model it will be used Docker to remove any issue pertaining the code runnability. To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

`test.sh` essentially setups a server exposing the model through a REST API and then queries this server, evaluating it.

#### 1. Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

> ⚠️ Unfortunately, for the latter command to have effect, you need to **reboot** your Ubuntu OS and re-login. **Do it** before proceeding.

#### 2. Setup Client

The model will be exposed through a REST server, in order to call it during the evaluation we need a client. The client is written in the evaluation script and it needs some dependencies to run: Use conda to create the environment for the client.<br>

```bash
conda create -n nlp-named_entity_recognition python=3.9
conda activate nlp-named_entity_recognition
pip install -r requirements.txt
```

## 🏋️ Training

From the `code/` directory:

```bash
cd code
python train.py
```

This will:
1. Load and preprocess `data/train.tsv`
2. Build word/tag dictionaries and save them to `model/dicts/`
3. Split data into train/validation sets
4. Train the `NERNetwork` LSTM model for the configured number of epochs
5. Save a checkpoint after each epoch to `code/ckpt/`
6. Generate confusion matrices (`code/cm/`) and loss/accuracy plots


## 🚀 Usage

### Running the inference server (Docker)
The `docker/` folder contains a self-contained Flask service that loads a trained checkpoint and serves predictions over HTTP. <br>

*test.sh* is a simple bash script. It automates the full cycle — build the image, start the container, run `docker/evaluate.py` against a JSONL test file, print accuracy/F1 metrics, then stop and remove the container (dumping logs to `logs/server.stdout` and `logs/server.stderr`) <br>
To run it:

```bash
# Build and run the server, then evaluate it against a test file
conda activate nlp-named_entity_recognition
bash test.sh data/dev.tsv
```
 
> ⚠️ Actually, you can replace *data/test.jsonl* to point to a different file, as far as the target file has the same format.


## 👤 Author

**Antonio Gagliardi**  
Email: [gaglia.anto95@gmail.com](mailto:gaglia.anto95@gmail.com)
