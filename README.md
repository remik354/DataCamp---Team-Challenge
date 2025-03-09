# RAMP challenge - Portugese grade estimation

[![Build status](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml/badge.svg)](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml)

## Introduction

This challenge is based on a real-world dataset of academic and non-academic information collected on students from two Portuguese secondary schools in 2005-2006. The data includes a variety of student features (numerical and categorical) such as demographics, social factors, school-related characteristics, and academic performance in the subjects of Portuguese language and Mathematics.

This challenge is a regression task aiming to leverage the links between non-academic information, academic performance in Mathematics, and academic performance in Portuguese.
Participants will predict students' Portuguese grades using all other available (non-academic) information, and their Mathematics grades. 


## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Before all, you need to load the training and testing data :

```bash
python download_data.py
```

To get started on this RAMP challenge with the
[dedicated notebook](student_grades_estimation_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
