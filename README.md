# Project: Detecting safe/unsafe Wikipedia comments

This project was completed over the space of one week as a takehome assignment for a job interview. Since it's on my public github, I'm a) also using it as a public demo of my ML skills, and b) not mentioning the interviewing startup by name so that they can reuse the assignment for other people too.

## Task Description

The task is:
- Use the [Wikipedia Talk Labels: Personal Attacks dataset](https://www.kaggle.com/datasets/jigsaw-team/wikipedia-talk-labels-personal-attacks/data) to train a model to distinguish between "safe" and "unsafe" comments.
- Build an end-to-end pipeline that handles dataset preprocessing, feature selection, and model training.
- Ideally train several models, both "traditional" ML and deep models, and compare and contrast their performance.
- Produce a report describing the results, choices made, etc.

## Deliverables

- The report is presented in the form of a notebook: see [report.ipynb](report.ipynb).
- The bulk of the pipeline is consolidated in two files that can each be run as scripts or imported into a notebook for interactive use:
  - `data_preparation.py` downloads the dataset, cleans it up, and processes it into a format that's ready for use in downstream training. 
  - `nn_classifier.py` trains a deep learning model on this dataset, consisting of a RoBERTa backbone and a dense classifier head.
- Both scripts are controlled by config files, namely `data_prep_config.yaml` and `train_config.yaml` respectively.

## Libraries

- The data wrangling is all handled by `polars`, which is essentially pandas-done-right. 
- The model training loop is written in JAX/Flax, with dataset/dataloaders subclassed from PyTorch and the pretrained RoBERTa model + tokenizer from HuggingFace.

Since part of the goal here was to demonstrate my understanding of custom deep learning models and training loops, I've deliberately avoided using HuggingFace's interface for model training. Instead I've essentially just used HuggingFace as a model hub, extracted the model definition and pretrained variables I needed, and written a flexible+scalable custom training loop in pure JAX/Flax.

## About Me

I'm a math PhD turned ML engineer and former Google AI Resident. I can work with pretty much any portion of the ML pipeline (or quickly learn to do anything I haven't seen before!) but my primary interests are in the dataset and model evaluation portions.

If you like what you see here and could use a competent and enthusiastic problem-solver on your team:
  - My website (with resume, publication list, etc) is [here](https://zeefryer.github.io/).
  - You can reach out to me directly at <fryer.zee@gmail.com>.

