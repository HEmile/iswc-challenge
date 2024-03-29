# iswc-challenge

## Getting started

### Prerequisites

This repository uses Python >= 3.10

Be sure to run in a virtual python environment (e.g. conda, venv, mkvirtualenv, etc.)

### Installation

1. In the root directory of this repo run

    ```bash
    pip install -r requirements.txt
    ```

### Usage

For running and evaluating the baseline, run :

```bash
python baseline.py -i "data/dev.jsonl" -o "predictions/baseline.pred.jsonl"
python evaluate.py -p "predictions/baseline.pred.jsonl" -g "data/dev.jsonl"
```

For running and evaluating our proposed GPT3 approach, make sure you set your `OPENAI_API_KEY` in the environmental
variables. This will use the default values for training, i.e. `text-davinci-002` model, `data/dev.jsonl` as input
and `predictions/gpt3.pred.jsonl` as output. Run :

```bash
python gpt3_baseline
python evaluate -p "predictions/gpt3.pred.jsonl" -g "data/dev.jsonl"
```

For the scaling experiment, you need to change the flag `model` to the respective model. The options
include: `['text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001']`

```bash
python gpt3_baseline -i "data/dev.jsonl" -o "predictions/gpt3-ada.pred.jsonl" --model "text-ada-001"
python evaluate -p "predictions/gpt3.pred.jsonl" -g "data/dev.jsonl"
```

## Tasks:

- [X] Make changes that the competition organisers suggest [priority]
    - [X] Pull the changes from their repo
    - [X] Check our performance on the updated train/val dataset
- [X] Dataset statistics (nice to include in the paper)
    - [X] The number of answers per relation
    - [X] Count the number of 'None' per relation
- [X] Logic integrity
    - [X] Run for all prompts.
    - [ ] Report on performance difference.
- [X] Submit current version to leadership board
- [X] Look at failure cases
    - [X] Wrong formatting? :: We tried different formatting - no significant improvement.
- [ ] Improve recall via
    - [ ] Reduce temperature and generate multiple samples (k=3?)
    - [ ] Rephrase
      prompts? :: [link to colab](https://colab.research.google.com/drive/180FCaZYRLEk0pPOWVYGsM_vn1UpIYy4N?usp=sharing#scrollTo=uuU4UPYDsSLP)
- [ ] General improvements
    - [ ] Can we use the logprob?
    - [ ] Are we using other models?

## License

Distributed under the MIT License.
See [`LICENSE`]() for more information.

## Authors (Alphabetical order)

* [Dimitrios Alivanistos](https://dimitrisalivas.github.io/)
* [Selene Báez Santamaría](https://selbaez.github.io/)
* [Michael Cochez](https://www.cochez.nl/)
* [Jan-Christoph Kalo](https://github.com/JanKalo)
* [Emile Van Krieken](https://emilevankrieken.com/)
* [Thiviyan Thanapalasingam](https://thiviyansingam.com/)
