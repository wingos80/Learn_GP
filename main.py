import time

timer = time.perf_counter_ns

start = timer()
import logging
from typing import Tuple, List
from collections.abc import Callable
import jax, jax.numpy as jnp

logger = logging.getLogger(__name__)


# visualize cov matrix for verification
def visualize_cov(domain, K):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    pcm = ax.pcolor(domain.flatten(), jnp.flip(domain.flatten()), K)
    cbar = fig.colorbar(pcm, ax=ax, extend="max")
    plt.show()


# Plot the GP
def plot_gp(GP: Tuple[Callable, jax.Array], dataset: Tuple[jax.Array, jax.Array]):
    kernel, K_y = GP[0], GP[1]
    K_y_inv = jnp.linalg.inv(K_y)

    # evaluating the gp
    plot_domain = jnp.linspace(-0.7, 1, 403)
    gp_evaluated = jnp.array(
        [
            evaluate_gp(x_star, kernel, K_y_inv=K_y_inv, dataset=dataset)
            for x_star in plot_domain
        ]
    )
    plot_values = gp_evaluated[:, 0]
    std = gp_evaluated[:, 1]

    import matplotlib.pyplot as plt

    plt.plot(plot_domain, plot_values)
    plt.fill_between(
        plot_domain, plot_values + std, plot_values - std, color="C0", alpha=0.1
    )
    plt.show()


# evaluate a Gaussian Process at the point x_star, given the GP's K inverse K_y_inv, and the pre-existing dataset x,y
def evaluate_gp(
    x_star: float,
    kernel: Callable,
    K_y_inv: Callable,
    dataset: Tuple[jax.Array, jax.Array],
) -> jax.Array:
    domain, values = dataset[0], dataset[1]

    k_star_row = jnp.array([kernel(x_star, x_j) for x_j in domain]).reshape(
        1, -1
    )  # create a row vector k*
    k_star_K_inv = k_star_row @ K_y_inv

    evaluant = (k_star_K_inv @ values)[0, 0]
    var = kernel(x_star, x_star) - k_star_K_inv @ k_star_row.T
    return jnp.array([evaluant, jnp.sqrt(var[0, 0])])


# Parameterize the GP
def define_gp(
    dataset: Tuple[jax.Array, jax.Array], RBF_theta: float
) -> Tuple[Callable, jax.Array]:
    train_domain = dataset[0]

    # mu = jnp.zeros_like(train_domain)  # IMO not neccessary to parameterize a GP
    kernel_function: Callable = lambda x_i, x_j: jnp.exp(
        -jnp.linalg.norm(x_i - x_j) / (2 * RBF_theta**2)
    )  # RBF
    K: jax.Array = jnp.array(
        [kernel_function(x_j, x_i) for x_i in train_domain for x_j in train_domain]
    ).reshape(
        len(train_domain), -1
    )  # covariance function, TODO refactor away double list comprehension
    K_y = K  # (optional) incorporate noise here
    return kernel_function, K_y


# # Update the "trained" (ppl call it a-priori sometimes) GP with new data points
# def update_gp(post_domain: jax.Array, prior_domain: jax.Array, GP: Tuple[Callable, jax.Array]) -> Tuple[Callable, jax.Array]:
#     """
#     TODO, verify i can actually add new data pts?
#     post_domain:
#         domain over which the "posterior" dataset is defined
#     prior_domain:
#         domain over which the "prior" GP's cov matrix is defined
#     GP:
#         A Gassuian Process, defined by it's Kernel function and
#         the Covariance matrix (1st & 2nd elem of respectively).
#     """
#     kernel_function, K = GP

#     # construct the new covariance vector k*
#     k_star: jax.Array = jnp.array(
#         [kernel_function(x_j, x_i)
#          for x_i in post_domain
#          for x_j in prior_domain]
#     ).reshape(
#         len(post_domain), len(prior_domain)
#     )  # double for loop in list comprehension, shape is latter then former

#     # augment the prior cov matrix with k*
#     K_post_1 = jnp.concatenate((K, k_star), axis=0)  # first add k*, the posterior-information as new columns
#     k_star = jnp.concatenate((k_star,jnp.zeros((len(post_domain), len(post_domain)))),axis=1)  # assume posterior has 0 noise
#     K_post_2 = jnp.concatenate((K_post_1, k_star.T), axis=1)  # then add k* as new rows

#     posterior_GP = (kernel_function, K_post_2)
#     return posterior_GP


# Negative of a minimal version of the original log likelihood of a GP
def negative_log_likelihood(K_y: jax.Array, train_values: jax.Array):
    return (
        jnp.log(jnp.linalg.norm(K_y, ord="fro"))
        + train_values.T @ jnp.linalg.inv(K_y) @ train_values
    )


def determine_costs(
    dataset: Tuple[jax.Array, jax.Array], test_pts: List[float]
) -> List[float]:
    pt_costs = [1e9 for _ in test_pts]
    for i, L in enumerate(test_pts):
        _, K_y = define_gp(dataset, RBF_theta=L)
        pt_costs[i] = negative_log_likelihood(K_y, dataset[1])

    return pt_costs


def find_optimal_hyperparameter(
    dataset: Tuple[jax.Array, jax.Array],
    itrs: int = 10,
    ub_lb: List[float] = [0.25, 0.75],
) -> float:
    # Optimizing hyperparameter for GP
    bounds = ub_lb
    bound_costs = determine_costs(dataset, bounds)
    # Iteratively update the length scale (GP's hyperparameter)
    for itr in range(itrs):
        print(f"---------{itr=}---------")

        step = (bounds[1] - bounds[0]) / 3
        # test_pts = [step + bounds[0] for _ in range(3)]
        test_pts = [bounds[0] + step, (bounds[1] - bounds[0]) / 2, bounds[1] - step]

        costs = determine_costs(dataset, test_pts)

        L, L_cost = test_pts[1], costs[1]
        print(f"log(p) of L={L}: {L_cost}")
        if L_cost == min(bound_costs):
            print(f"optimal hyperparameter found, terminating search")
            print(f"---------Bisection Terminated---------")
            break
        if L_cost > min(bound_costs):
            bounds[0], bounds[1] = test_pts[0], test_pts[2]
            print(f"optimal hyperparameter within bounds, shrinking search")
            continue
        else:
            print(f"optimal hyperparameter not within bounds, expanding search")
            bounds[0] -= step * 2
            bounds[1] += step * 2

    return L


def main():
    ## Single dimensional GP
    # The dataset
    train_domain = jnp.array([-0.5, 0, 0.2, 0.8]).reshape(
        -1, 1
    )  # column vector of observation locations
    train_values = jnp.array([1, 2, 0.5, 0.9]).reshape(
        -1, 1
    )  # column vector of observations
    dataset = (train_domain, train_values)
    new_domain = jnp.array([0.3, 0.4]).reshape(-1, 1)
    new_values = jnp.array([0.55, 0.65]).reshape(-1, 1)
    new_dataset = (new_domain, new_values)

    L = find_optimal_hyperparameter(dataset=dataset, itrs=10, ub_lb=[0.01, 1])
    kernel_function, K_y = define_gp(dataset, RBF_theta=0.25)

    print("\n\nTrained one GP!\n\n")
    plot_gp(GP=(kernel_function, K_y), dataset=dataset)

    posterior_data = (
        jnp.concatenate((dataset[0], new_dataset[0])),
        jnp.concatenate((dataset[1], new_dataset[1])),
    )

    L = find_optimal_hyperparameter(dataset=posterior_data, ub_lb=[0.01, 1])
    kernel_function, K_y = define_gp(posterior_data, RBF_theta=0.6)

    print("\n\nTrained the GP again!\n\n")
    plot_gp(GP=(kernel_function, K_y), dataset=posterior_data)


if __name__ == "__main__":
    print(f"Starting GP script*_-+_*_...._8___")
    main()
    end = timer()
    print(f"time: {(end-start)/1e9} s")
