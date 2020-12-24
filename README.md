# Surrogate-assisTed bayEsiAn Differential Evolution (STEADE)

This repository contains code for STEADE algorithm desgined for the [black box optimization challenge](https://bbochallenge.com/) at [NeurIPS 2020](https://neurips.cc/Conferences/2020/CompetitionTrack).
Kindly refer to [the starter kit](https://github.com/rdturnermtl/bbo_challenge_starter_kit/) to know more about the competition details and the blackbox optimization problem.

The code for `STEADE` along with some baseline optimizers is provided in the folder `example_submissions`:

```console
hyperopt/
nevergrad/
opentuner/
pysot/
random_search/
skopt/
steade/
turbo/
```

## Execution instructions

Local experimentation/benchmarking on publicly available problems can be done using Bayesmark, which is extensively [documented](https://bayesmark.readthedocs.io/en/latest/index.html).
Note, these are not the same problems as on the leaderboard when submitting on the website, but may be useful for local iteration.

For convenience the script `run_local` can do local benchmarking in a single command:

```console
> ./run_local.sh ./example_submissions/steade 3
...
--------------------
Final score `100 x (1-loss)` for leaderboard:
optimizer
steade_0.3.3_8cae841    98.72426
```

It produces a lot of log output as it runs, that is normal.
The first argument gives the *folder of the optimizer* to run, while the second argument gives the number of *repeated trials* for each problem.
Set the repeated trials as large as possible within your computational budget.
For finer grain control over which experiments to run, use the Bayesmark commands directly.
See the [documentation](https://bayesmark.readthedocs.io/en/latest/readme.html#example) for details.

More explanation on how the mean score formula works can be found [here](https://bayesmark.readthedocs.io/en/latest/scoring.html#mean-scores).


### Evaluation budget

The optimizer has a total of 640 seconds compute time for making suggestions on each problem (16 iterations with batch size of 8); or 40 seconds per iteration.
Optimizers exceeding the time limits will be cut off from making further suggestions and the best optima found before being killed will be used.
The optimizer evaluation terminates after 16 iterations (batch size 8) or 640 seconds, whichever happens earlier.
There is no way to get more iterations by being faster.


### Execution environment

The code was run in `Ubuntu 20.04.1 LTS (Focal Fossa)` with `Python 3.6.10` and the pre-installed packages are listed in `environment.txt`.
It is recommended to use a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/) to install these packages as:

```bash
pip install -r environment.txt
```

### Configuration space

The search space is defined in the `api_config` dictionary in the constructor to the optimizer (see template above).
For example, if we are optimizing the hyper-parameters for the scikit-learn neural network with ADAM `sklearn.neural_network.MLPClassifier` then we could use the following configuration for `api_config`:

```python
api_config = \
    {'hidden_layer_sizes': {'type': 'int', 'space': 'linear', 'range': (50, 200)},
     'alpha': {'type': 'real', 'space': 'log', 'range': (1e-5, 1e1)},
     'batch_size': {'type': 'int', 'space': 'linear', 'range': (10, 250)},
     'learning_rate_init': {'type': 'real', 'space': 'log', 'range': (1e-5, 1e-1)},
     'tol': {'type': 'real', 'space': 'log', 'range': (1e-5, 1e-1)},
     'validation_fraction': {'type': 'real', 'space': 'logit', 'range': (0.1, 0.9)},
     'beta_1': {'type': 'real', 'space': 'logit', 'range': (0.5, 0.99)},
     'beta_2': {'type': 'real', 'space': 'logit', 'range': (0.9, 1.0 - 1e-6)},
     'epsilon': {'type': 'real', 'space': 'log', 'range': (1e-9, 1e-6)}}
```

Each key in `api_config` is a variable to optimize and its description is itself a dictionary with the following entries:

* `type`: `{'real', 'int', 'cat', 'bool'}`
* `space`: `{'linear', 'log', 'logit', 'bilog'}`
* `values`: array-like of same data type as `type` to specify a whitelist of guesses
* `range`: `(lower, upper)` tuple to specify a range of allowed guesses

One can also make the following assumption on the configurations:

* `space` will only be specified for types `int` and `real`
* `range` will only be specified for types `int` and `real`
* We will not specify both `range` and `values`
* `bool` does not take anything extra (`space`, `range`, or `values`)
* The `values` for `cat` will be strings

For `observe`, `X` is a (length `n`) list of dictionaries with places where the objective function has already been evaluated.
Each suggestion is a dictionary where each key corresponds to a parameter being optimized.
Likewise, `y` is a length `n` list of floats of corresponding objective values.
The observations `y` can take on `inf` values if the objective function crashes, however, it should never be `nan`.

For `suggest`, `n_suggestions` is simply the desired number of parallel suggestions for each guess.
Also, `next_guess` will be a length `n_suggestions` array of dictionaries of guesses, in the same format as `X`.


## Terms of use

If you use the code of STEADE, please consider citing our paper:
```
@misc{biswas2020better,
      title={Better call Surrogates: A hybrid Evolutionary Algorithm for Hyperparameter optimization}, 
      author={Subhodip Biswas and Adam D Cobb and Andreea Sistrunk and Naren Ramakrishnan and Brian Jalaian},
      year={2020},
      eprint={2012.06453},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

## Contact

Any questions can be sent to [subhodip@cs.vt.edu](mailto:subhodip@cs.vt.edu).
