# Superhero Intelligence System — AI-03

This repository contains a reproducible implementation harness for Assignment 03.

## Required data

Place the instructor-provided files here:

```text
data/train.csv
data/test.csv
```

The runner intentionally does not create a replacement test set. The assignment requires the supplied `train.csv` and `test.csv`; results cannot be finalized without both files.

## Run the tests

```powershell
python -m pytest -q
```

## Run the complete assignment

Install dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Then run:

```powershell
python run_assignment.py
```

The runner creates `outputs/` containing the required graphs, tables, a generated report, and validation results. It uses the exact six permitted score features and keeps test data out of fitting, cross-validation, and model selection.

## Notebook

`superhero_project.ipynb` is the submission-oriented entry point. It calls the same runner used by the command line so the notebook and scripted run cannot silently diverge.

## Repository workflow

Changes are developed on a feature branch, tested locally, reviewed through a pull request, and merged into `main` only after the acceptance checks pass. Dataset files are intentionally ignored by Git and must be supplied separately according to the course instructions.

