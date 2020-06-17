## Infilling by Language Modeling (ILM)

This repository houses the code for the ILM framework outlined in the ACL 2020 paper [_Enabling language models to fill in the blanks_](https://arxiv.org/abs/2005.05339) (Donahue et al. 2020).

This codebase allows you to fine tune GPT-2 to _infill_, i.e., perform text generation conditioned on both past and future context. For example, you could train GPT-2 to infill proper nouns in news articles, or generate lines of poetry in the middle of the stanza.

An interactive webdemo can be found at [chrisdonahue.com/ilm](https://chrisdonahue.com/ilm).

## Installation

We recommend installing this package using `virtualenv`. After activating the virtual environment, run the following commands:

1. `git clone git@github.com:chrisdonahue/ilm.git`
1. `cd ilm`
1. `pip install -r requirements.txt`
1. `python -c "import nltk; nltk.download('punkt')"`
1. `pip install -e .`

## Training a new model

The ILM framework involves a two step process of (1) creating ILM training examples by randomly masking training data, and (2) fine-tuning GPT-2 on those examples. This section walks through an example of this process for one of the built-in datasets and mask functions.

### Creating ILM training examples

The process of creating ILM examples involves randomly masking spans in complete text. For example, if the original text is `She ate leftover pasta for lunch`, an ILM example might look like `She ate [blank] for [blank] [sep] leftover pasta [answer] lunch [answer]`. For efficiency reasons, this codebase generates these examples up front before training.

The following script will download a dataset of scientific abstracts collected from arXiv and then create ILM examples for training:

```sh
DATASET=arxiv_cs_abstracts

pushd data
./get_${DATASET}.sh
popd

for SPLIT in train valid
do
	python create_ilm_examples.py \
		${SPLIT} \
		data/char_masks/${DATASET} \
		--seed 0 \
		--data_name ${DATASET} \
		--data_split ${SPLIT}
done
```

Before training, you can optionally preview these examples

```sh
python preview_ilm_examples.py \
	data/char_masks/arxiv_cs_abstracts/train.pkl
```

### Training an ILM model

Once you've created training examples, you can start training an ILM model (fine-tuning GPT-2):

```sh
DATASET=arxiv_cs_abstracts
TRAIN_DIR=train
EXAMPLES_DIR=data/char_masks/${DATASET}
python train_ilm.py \
	experiment_${DATASET} \
	${TRAIN_DIR} \
	${EXAMPLES_DIR} \
	--seed 0 \
	--train_examples_tag train \
	--eval_examples_tag valid \
	--eval_max_num_examples 512
```

Note that the training script automatically performs early stopping based on PPL on the validation set. To monitor training, you can set up an account on [Weights and Biases](https://www.wandb.com) and add the `--wandb` flag.

## Using custom datasets and mask functions

This codebase includes scripts to download the three datasets used in our paper: [ROC stories](https://cs.rochester.edu/nlp/rocstories/), abstracts from arXiv, and song lyrics. By default, the scripts are configured to use the hierarchical mask function outlined in our paper. This section outlines how to train ILM models on [custom datasets](#custom-datasets) and [custom mask functions](#custom-mask-functions)

### Custom datasets

To add a new dataset, first split it into three files: `train.txt`, `valid.txt`, `test.txt`. These files each contain complete documents separated by _three_ newline characters, i.e., `'\n\n\n'.join(documents)`. Then, run `create_ilm_examples.py` with the following arguments: `--data_name custom --data_dir path/to/directory/with/splits`.

### Custom mask functions

A mask function takes text and outputs random spans to masked which correspond to intended downstream behavior. By default, this repository trains ILM models which can infill words, ngrams, sentences, paragraphs, and entire documents.

You can add your own mask functions to perform different infilling tasks. A mask function takes as input a complete document and outputs a list of 3-tuples consisting of `(infilling type, span offset, span length)`, where offset and length are measured in characters.

You can add your custom mask function to [`ilm.mask.custom`](https://github.com/chrisdonahue/ilm_final/blob/master/ilm/mask/custom.py), where there are already two simple examples:

- `ilm.mask.custom.MaskPunctuation`: Masks each punctuation token with 50% probability. Special infilling type for sentence terminals.
- `ilm.mask.custom.MaskProperNoun`: Masks all (detected) proper nouns with 100% probability.

Once you add your mask function, you should pass it as an argument to `create_ilm_examples.py` and `train_ilm.py` scripts e.g.: `--mask_cls ilm.mask.custom.YourCustomMaskFn`.

## Infilling with a trained model

See `inference.ipynb` for an example of how to perform infilling with a trained model. If you prefer, you may also run this notebook on [Google Colab](https://colab.research.google.com/drive/1So95M0hHefyNm_eELglCna_ZayoDX6KV?usp=sharing).

## Reproducing results from ACL 2020 paper

### Training

We've included a script `acl20_repro_train.py` which will re-train models using the same hyperparameters we used in our ACL paper. This script will print out another script which, if run, downloads the training examples and re-trains the model. The script takes two arguments:

1. Dataset name: One of `abstracts`, `stories`, or `lyrics`
1. Model type: One of `lm`, `lmrev`, `lmall`, `ilm`, `lmscratch`, `lmrevscratch`, `lmallscratch`, `ilmscratch`

For example, to train an ILM on the stories dataset, run:

```sh
export ILM_DIR=/tmp/ilm
python acl20_repro_train.py stories ilm | bash
```

Each experiment will take 1-2 days on a GPU, and early stopping is performed automatically. Note that the resultant model may differ slightly from the models we evaluated in our paper; our paper experiments were sometimes paused and re-started during training which affected the ordering of training data.

### Evaluation

We've included a script `acl20_repro_eval.py` which _exactly_ reproduces PPL numbers found in our ACL paper. This script will print out another script which, if run, downloads the relevant pre-trained model (~500MB) and pre-masked test data and computes the PPL. It takes three arguments:

1. Dataset name: One of `abstracts`, `stories`, or `lyrics`
1. Model type: One of `lm`, `lmrev`, `lmall`, `ilm`, `lmscratch`, `lmrevscratch`, `lmallscratch`, `ilmscratch`
1. Infilling type: One of `sentence`, `document`, `mixture`, `paragraph`, `ngram`, or `word` for paper Tables 1, 3, 4, 5, 7, and 8, respectively

For example, to reproduce PPL of ILM on the sentence infilling task for the Stories dataset (`15.6` in bottom left of Table 1), run:

```sh
export ILM_DIR=/tmp/ilm
python acl20_repro_eval.py stories ilm sentence | bash 2> /dev/null | grep eval_infill_textonly_ppl
```

You should see the output `eval_infill_textonly_ppl: 15.56...` which matches the value from the paper.

Occasionally, the model will fail to download from Google Drive. If this happens (i.e., the evaluation isn't running to completion), simply run `rm -rf /tmp/ilm_reproduce` and try again.

## Citation

If you use this codebase in your work, please consider citing our paper:

```
@inproceedings{donahue2020ilm,
  author = {Chris Donahue and Mina Lee and Percy Liang},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  title = {Enabling language models to fill in the blanks},
  year = {2020},
}
```
