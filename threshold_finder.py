import numpy as np
from scipy.stats import beta, norm  # type: ignore


class BayesianRootFinder:
    """
    Implementation of the Probabilistic Bisection Algorithm for stochastic root finding
    based on Waeber, Frazier, and Henderson (2011), with support for Gaussian priors.
    """

    def __init__(
        self,
        lower_bound=0.0,
        upper_bound=1.0,
        n_grid_points=101,
        prior_mean=None,
        prior_std=None,
    ):
        """
        Initialize the Bayesian root finder.

        Parameters:
        -----------
        lower_bound : float
            Lower bound of the search space
        upper_bound : float
            Upper bound of the search space
        n_grid_points : int
            Number of grid points to discretize the posterior density
        prior_mean : float, optional
            Mean of the Gaussian prior. If None, uses a uniform prior.
        prior_std : float, optional
            Standard deviation of the Gaussian prior. Required if prior_mean is provided.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_grid_points = n_grid_points

        # Initialize grid points
        self.grid = np.linspace(lower_bound, upper_bound, n_grid_points)
        self.dx = (upper_bound - lower_bound) / (n_grid_points - 1)

        # Initialize prior density
        if prior_mean is not None and prior_std is not None:
            # Use Gaussian prior
            self.log_density = np.log(norm.pdf(self.grid, prior_mean, prior_std))
            # Normalize the log density
            self.log_density = self.log_density - np.max(self.log_density)
            density = np.exp(self.log_density)
            self.log_density = np.log(density / np.sum(density))
        else:
            # Use uniform prior
            self.log_density = np.zeros(n_grid_points) - np.log(n_grid_points)

        # Initialize history tracking
        self.history = {"x": [], "y": [], "p": [], "median_estimates": []}

    def get_median(self):
        """Find the median of the current posterior density."""
        # Convert log probabilities to regular probabilities for median calculation
        density = np.exp(self.log_density - np.max(self.log_density))
        density = density / np.sum(density)

        # Compute CDF
        cdf = np.cumsum(density)
        # Find the point where CDF crosses 0.5
        idx = np.searchsorted(cdf, 0.5)
        return self.grid[idx]

    def update_density(self, x, y, p):
        """
        Update the posterior density using Bayes rule given an observation.

        Parameters:
        -----------
        x : float
            The location that was sampled
        y : int
            The outcome: +1 if root is to the right, -1 if root is to the left
        p : float
            The probability of correct response (between 0.5 and 1.0)
        """
        assert y == -1 or y == 1
        assert p >= 0.5 and p <= 1.0

        # Find the grid point closest to x
        idx = np.searchsorted(self.grid, x)

        # Calculate the log likelihood for each grid point
        log_likelihood = np.zeros_like(self.log_density)

        # Update log likelihood based on the sign
        if y == -1:  # Response suggests root is to the left
            log_likelihood[:idx] = np.log(p)  # Points to left are more likely
            log_likelihood[idx] = (np.log(p) + np.log(1 - p)) / 2
            log_likelihood[idx + 1 :] = np.log(1 - p)  # Points to right are less likely
        else:  # y == 1, response suggests root is to the right
            log_likelihood[:idx] = np.log(1 - p)  # Points to left are less likely
            log_likelihood[idx] = (np.log(p) + np.log(1 - p)) / 2
            log_likelihood[idx + 1 :] = np.log(p)  # Points to right are more likely

        # Update log density using Bayes rule (addition in log space)
        self.log_density = self.log_density + log_likelihood

        # Normalize the log density (subtract the maximum to prevent overflow)
        self.log_density = self.log_density - np.max(self.log_density)
        density = np.exp(self.log_density)
        density = density / np.sum(density)

        # Update history
        self.history["x"].append(x)
        self.history["y"].append(y)
        self.history["p"].append(p)
        self.history["median_estimates"].append(self.get_median())

    def next_sample_point(self):
        """Return the next point to sample"""
        return self.get_median()


async def find_threshold(
    target_score,
    get_score_fn,
    n_trials=10,
    update_weight=5,
    show_progress=False,
    prior_mean=1.0,
    prior_std=0.25,
    lower_bound=0.0,
    upper_bound=5.0,
):
    """
    Find the threshold for a feature (some adjustable parameter) that results in a target score.

    Parameters:
    -----------
    target_score : float
        The target score to achieve (between 0 and 1)
    get_score_fn : callable
        Function that takes a boost value and returns a score ()
    n_trials : int
        Number of trials to run
    update_weight : float
        Weight for updating the beta distribution
    show_progress : bool
        Whether to show progress during the search
    prior_mean : float
        Mean of the Gaussian prior for the threshold
    prior_std : float
        Standard deviation of the Gaussian prior for the threshold
    lower_bound : float
        Lower bound of the search space
    upper_bound : float
        Upper bound of the search space

    Returns:
    --------
    float
        The estimated threshold
    """
    root_finder = BayesianRootFinder(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_grid_points=10_000,  # Fine grid for better precision
        prior_mean=prior_mean,
        prior_std=prior_std,
    )

    for _ in range(n_trials):
        boost = root_finder.next_sample_point()
        if show_progress:
            print(f"boost={boost.item():.2f}")

        # Get response with feature boosted
        score = await get_score_fn(boost.item())

        if show_progress:
            print(f"boost={boost.item():.2f} score={score:.2f}")

        y = 1 if score >= target_score else -1

        # Model score as beta distribution that starts as Jeffreys prior and then updates with weight
        alpha = 0.5 + update_weight * score
        beta_param = 0.5 + update_weight * (1 - score)

        # p is probability that sign of (score - target_score) is correct
        p_below = beta.cdf(target_score, alpha, beta_param)
        if y == 1:  # We predicted score >= target_score
            p = 1 - p_below
        else:  # y == -1, we predicted score < target_score
            p = p_below

        # Ensure p is in [0.5, 1.0] range required by update_density
        if p < 0.5:
            raise Exception("got p < 0.5, check math")
        if p > 1.0:
            raise Exception("got p > 1.0, check math")

        root_finder.update_density(boost, y, p)

    # Ensure callers don't accidentally pass numpy scalars into downstream systems (e.g. vLLM serialization).
    return float(root_finder.get_median())
