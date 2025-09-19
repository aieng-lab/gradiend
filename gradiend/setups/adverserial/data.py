import numpy as np


def generate_synthetic_data(n_samples=1000, random_seed=None):
    """
    Generate synthetic data for binary classification with 4 input features:

    - Feature 0: Informative for class 0, non-informative for class 1.
    - Feature 1: Informative for class 1, non-informative for class 0.
    - Feature 2: Informative for both classes.
    - Feature 3: Non-informative for both classes.

    Returns:
        X: numpy array of shape (n_samples, 4)
        y: numpy array of shape (n_samples,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate labels
    y = np.random.randint(0, 2, size=n_samples)

    # Initialize feature matrix
    X = np.zeros((n_samples, 4))

    # Feature 0: informative for class 0
    #   - class 0: drawn from N(μ=1, σ=1)
    #   - class 1: drawn from background N(μ=0, σ=1) (non-informative)
    X[:, 0] = np.where(y == 0,
                       np.random.normal(loc=1.0, scale=1.0, size=n_samples),
                       np.random.normal(loc=0.0, scale=1.0, size=n_samples))

    # Feature 1: informative for class 1
    #   - class 1: drawn from N(μ=1, σ=1)
    #   - class 0: drawn from background N(μ=0, σ=1)
    X[:, 1] = np.where(y == 1,
                       np.random.normal(loc=1.0, scale=1.0, size=n_samples),
                       np.random.normal(loc=0.0, scale=1.0, size=n_samples))

    # Feature 2: informative for both classes
    #   - class 0: drawn from N(μ=1, σ=1)
    #   - class 1: drawn from N(μ=-1, σ=1)
    X[:, 2] = np.where(y == 0,
                       np.random.normal(loc=1.0, scale=1.0, size=n_samples),
                       np.random.normal(loc=-1.0, scale=1.0, size=n_samples))

    # Feature 3: non-informative for both classes
    #   - both classes: drawn from N(μ=0, σ=1)
    X[:, 3] = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

    return X, y


import numpy as np


def generate_regression_data_structured(n_samples=1000, random_seed=None):
    """
    Generate synthetic regression data with 4 input features and 2 output targets:

    - Input 0: influences Output 0 only
    - Input 1: influences Output 1 only
    - Input 2: influences both Output 0 and Output 1
    - Input 3: non-informative (noise only)

    Returns:
        X: numpy array of shape (n_samples, 4)
        y: numpy array of shape (n_samples, 2)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.random.normal(0, 1, size=(n_samples, 4))

    # Define outputs
    y0 = 2.0 * X[:, 0] + 1.0 * X[:, 2] + np.random.normal(0, 0.1, size=n_samples)  # influenced by x0 and x2
    y1 = -1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, size=n_samples)  # influenced by x1 and x2

    y = np.stack([y0, y1], axis=1).astype(np.float32)
    X = X.astype(np.float32)

    return X, y

if __name__ == '__main__':
    # Generate a small example
    X_example, y_example = generate_regression_data_structured(n_samples=5, random_seed=42)

    # Example usage
    X, y = generate_synthetic_data(n_samples=10, random_seed=42)
    print("X sample:\n", X)
    print("y sample:\n", y)
