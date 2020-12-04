import random
import warnings
import numpy as np
from copy import copy, deepcopy
from poap.strategy import EvalRecord
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import SRBFStrategy, DYCORSStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement, \
    qUpperConfidenceBound, qSimpleRegret
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_cyclic


class steade(AbstractOptimizer):
    primary_import = "pysot"

    def __init__(self, api_config):
        """Build wrapper class to use an optimizer in benchmark.
        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.search_space = JointSpace(api_config)
        self.bounds = self.search_space.get_bounds()
        self.iter = 0
        # Sets up the optimization problem (needs self.bounds)
        self.create_opt_prob()
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []
        self.proposals = []
        # Population-based parameters in DE
        self.population = []
        self.fitness = []
        self.F = 0.7
        self.Cr = 0.7
        # For bayes opt
        self.dim = len(self.search_space.param_list)
        self.torch_bounds = torch.from_numpy(self.search_space.get_bounds().T)
        self.min_max_bounds = torch.from_numpy(np.stack([np.zeros(self.dim), np.ones(self.dim)]))
        self.archive = []
        self.arc_fitness = []

    def create_opt_prob(self):
        """Create an optimization problem object."""
        opt = OptimizationProblem()
        opt.lb = self.bounds[:, 0]  # In warped space
        opt.ub = self.bounds[:, 1]  # In warped space
        opt.dim = len(self.bounds)
        opt.cont_var = np.arange(len(self.bounds))
        opt.int_var = []
        assert len(opt.cont_var) + len(opt.int_var) == opt.dim
        opt.objfun = None
        self.opt = opt

    def start(self, max_evals):
        """Starts a new pySOT run."""
        self.history = []
        self.proposals = []

        # Symmetric Latin hypercube design
        des_pts = max([self.batch_size, 2 * (self.opt.dim + 1)])
        slhd = SymmetricLatinHypercube(dim=self.opt.dim, num_pts=des_pts)

        # Warped RBF interpolant
        rbf = RBFInterpolant(dim=self.opt.dim, lb=self.opt.lb, ub=self.opt.ub,
                             kernel=CubicKernel(), tail=LinearTail(self.opt.dim), eta=1e-4)

        # Optimization strategy
        self.strategy = DYCORSStrategy(
            max_evals=self.max_evals,
            opt_prob=self.opt,
            exp_design=slhd,
            surrogate=rbf,
            asynchronous=True,
            batch_size=1,
            use_restarts=True,
        )

    def _suggest(self, n_suggestions=1):
        """Get a suggestion from the optimizer.
        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output
        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """

        if self.batch_size is None:  # First call to suggest
            self.batch_size = n_suggestions
            self.start(self.max_evals)

        # Set the tolerances pretending like we are running batch
        d, p = float(self.opt.dim), float(n_suggestions)
        self.strategy.failtol = p * int(max(np.ceil(d / p), np.ceil(4 / p)))

        # Now we can make suggestions
        x_w = []
        self.proposals = []
        for _ in range(n_suggestions):
            proposal = self.strategy.propose_action()
            record = EvalRecord(proposal.args, status="pending")
            proposal.record = record
            proposal.accept()  # This triggers all the callbacks

            # It is possible that pySOT proposes a previously evaluated point
            # when all variables are integers, so we just abort in this case
            # since we have likely converged anyway. See PySOT issue #30.
            x = list(proposal.record.params)  # From tuple to list
            x_unwarped, = self.search_space.unwarp(x)
            if x_unwarped in self.history:
                warnings.warn("pySOT proposed the same point twice")
                self.start(self.max_evals)
                return self.suggest(n_suggestions=n_suggestions)

            # NOTE: Append unwarped to avoid rounding issues
            self.history.append(copy(x_unwarped))
            self.proposals.append(proposal)
            x_w.append(copy(x_unwarped))

        return x_w

    @staticmethod
    def make_model(train_x, train_y, state_dict=None):
        """
        Define the models based on the observed data
        :param train_x: The design points/ trial solutions
        :param train_y: The objective functional value of the trial solutions used for model fitting
        :param state_dict: Dictionary storing the parameters of the GP model
        :return:
        """
        try:
            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            # load state dict if it is passed
            if state_dict is not None:
                model.load_state_dict(state_dict)
        except Exception as e:
            print('Exception: {} in make_model()'.format(e))

        return model, mll

    def get_bayes_pop(self, n_suggestions):
        """
        Parameters
        ----------
        n_suggestions: Number of new suggestions/trial solutions to generate using BO
        Returns
        The new set of trial solutions obtained by optimizing the acquisition function
        -------
        """

        try:
            candidates, _ = optimize_acqf(
                acq_function=self.acquisition,
                bounds=self.min_max_bounds,
                q=n_suggestions,
                num_restarts=10,
                raw_samples=512,  # used for initialization heuristic
                sequential=True
            )

            bayes_pop = unnormalize(candidates, self.torch_bounds).numpy()
        except Exception as e:
            print('Error in get_bayes_pop(): {}'.format(e))

        population = self.search_space.unwarp(bayes_pop)  # Translate the solution back to the original space

        return population

    def mutate(self, n_suggestions):
        """

        Parameters
        ----------
        n_suggestions
        Returns
        -------
        """

        parents = self.search_space.warp(self.population)
        surrogates = self.search_space.warp(self._suggest(n_suggestions))

        # Pop out 'n_suggestions' number of solutions from the archives since they will be modified
        for _ in range(n_suggestions):
            self.history.pop()
            # self.proposals.pop()

        # Applying DE mutation, for more details refer to https://ieeexplore.ieee.org/abstract/document/5601760
        a, b = 0, 0
        while a == b:
            a = random.randrange(1, n_suggestions - 1)
            b = random.randrange(1, n_suggestions - 1)

        rand1 = random.sample(range(0, n_suggestions), n_suggestions)
        rand2 = [(r + a) % n_suggestions for r in rand1]
        rand3 = [(r + b) % n_suggestions for r in rand1]

        try:
            bayes_pop = self.search_space.warp(self.get_bayes_pop(n_suggestions))

            # Bayesian mutation inspired from DE/rand/2 mutation
            mutants = bayes_pop[rand1, :] + \
                      self.F * (surrogates[rand2, :] - surrogates[rand3, :]) + \
                      1 / self.iter * np.random.random(parents.shape) * (parents[rand2, :] - parents[rand3, :])

        except Exception as e:
            # DE/rand/2 mutation applied when decomposition error encountered in BO
            mutants = parents[rand1, :] + \
                      self.F * (surrogates[rand2, :] - surrogates[rand3, :]) + \
                      1 / self.iter * np.random.random(parents.shape) * (parents[rand2, :] - parents[rand3, :])

        # Check the bound constraints and do (binomial) crossover to generate offsprings/ donor vectors
        offsprings = deepcopy(parents)
        bounds = self.search_space.get_bounds()
        dims = len(bounds)

        for i in range(n_suggestions):
            j_rand = random.randrange(dims)

            for j in range(dims):
                # Check if the bound-constraints are satisfied or not
                if mutants[i, j] < bounds[j, 0]:
                    mutants[i, j] = bounds[j, 0]  # surrogates[i, j]
                if bounds[j, 1] < mutants[i, j]:
                    mutants[i, j] = bounds[j, 1]  # surrogates[i, j]

                if random.random() <= self.Cr or j == j_rand:
                    offsprings[i, j] = mutants[i, j]

        # Translate the offspring back into the original space
        population = self.search_space.unwarp(offsprings)

        # Now insert the solutions back to the archive.
        for i in range(n_suggestions):
            self.history.append(population[i])

        return population

    def suggest(self, n_suggestions=1):
        """Get a suggestion from the optimizer.
        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output
        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """

        self.iter += 1
        lamda = 10     # defines the transition point in the algorithm

        if self.iter < lamda:
            population = self._suggest(n_suggestions)
        else:
            population = self.mutate(n_suggestions)

        return population

    def _observe(self, x, y):
        # Find the matching proposal and execute its callbacks
        idx = [x == xx for xx in self.history]
        i = np.argwhere(idx)[0].item()  # Pick the first index if there are ties
        proposal = self.proposals[i]
        proposal.record.complete(y)
        self.proposals.pop(i)
        self.history.pop(i)

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.
        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        try:
            assert len(X) == len(y)
            c = 0

            for x_, y_ in zip(X, y):
                # Archive stores all the solutions
                self.archive.append(x_)
                self.arc_fitness.append(-y_)  # As BoTorch solves a maximization problem

                if self.iter == 1:
                    self.population.append(x_)
                    self.fitness.append(y_)
                else:
                    if y_ <= self.fitness[c]:
                        self.population[c] = x_
                        self.fitness[c] = y_

                    c += 1

                # Just ignore, any inf observations we got, unclear if right thing
                if np.isfinite(y_):
                    self._observe(x_, y_)

            # Transform the data (seen till now) into tensors and train the model
            train_x = normalize(
                torch.from_numpy(self.search_space.warp(self.archive)),
                bounds=self.torch_bounds
            )
            train_y = standardize(
                torch.from_numpy(np.array(self.arc_fitness).reshape(len(self.arc_fitness), 1))
            )
            # Fit the GP based on the actual observed values
            if self.iter == 1:
                self.model, mll = self.make_model(train_x, train_y)
            else:
                self.model, mll = self.make_model(train_x, train_y, self.model.state_dict())

            # mll.train()
            fit_gpytorch_model(mll)

            # define the sampler
            sampler = SobolQMCNormalSampler(num_samples=512)

            # define the acquisition function
            self.acquisition = qExpectedImprovement(model=self.model, best_f=train_y.max(), sampler=sampler)

        except Exception as e:
            print('Error: {} in observe()'.format(e))


if __name__ == "__main__":
    experiment_main(steade)