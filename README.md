## Infilling by Language Modeling (ILM)

This repository houses the code for the ILM framework outlined in the ACL 2020 paper [_Enabling language models to fill in the blanks_](https://arxiv.org/abs/2005.05339) (Donahue et al. 2020).

This codebase allows you to fine tune GPT-2 to _infill_, i.e., perform text generation conditioned on both past and future context. For example, you could train GPT-2 to infill proper nouns in news articles, or generate lines of poetry in the middle of the stanza.

## Installation

We recommend installing this package using `virtualenv`. After activating the virtual environment, run the following commands:

1. `git clone git@github.com:chrisdonahue/ilm.git`
1. `cd ilm`
1. `pip install -r requirements.txt`
1. `python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"`
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

## Adding new datasets and mask functions

## Infilling with a trained model

## Reproducing results from ACL 2020 paper

### Citation

```
@inproceedings{donahue2020ilm,
  author = {Chris Donahue and Mina Lee and Percy Liang},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  title = {Enabling language models to fill in the blanks},
  year = {2020},
}
```
