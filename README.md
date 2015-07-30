# Metadata Embeddings for User and Item Cold-start Recommendations 

Pre-print available on [arXiv](http://arxiv.org). 

## Structure

1. The `paper` directory holds the LaTeX source of the paper.
2. The `experiments` directory holds the code used for experiments.

## Reproducing results

To reproduce the results from the paper, do the following.

1. Download and install the LightFM package (`pip install lightfm`).
2. Download experiment data by running `cd experiments/stackoverflow/ && make` and `cd experiments/movielens/ && make`.
3. To run the primary experiment, run `ipython -- runner.py --table`. This will print LaTeX variable definitions that, when pasted into the paper source file, populate the main results table.
4. To generate the latent dimension sensitivity plots, run `ipython -- runner.py --dim --plot`. This will take some time.

It is possible to run invididual experiments. Examples:

- `ipython -- experiments/stackexchange/model.py --dim 50 --lsi --tags --ids --split 0.2` will run the CrossValidated experiment with 50-dimensional
  latent space, using the LSI-LR model with both post tags and post ids.
- `ipython -- experiments/movielens/model.py --ids --cold --split 0.2` will run the MF model on the MovieLens dataset in the cold-start scenario.

## Citing this paper
Bibtex goes here.
