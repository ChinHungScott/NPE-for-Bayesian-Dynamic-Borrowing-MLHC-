"""
Simulation Design for Bayesian Dynamic Borrowing
=================================================

Based on the simulation setup with:
- Single arm: Nc = 50, 100, Ne = 200
- Covariates (p = 6)
- Concurrent and External data with controlled exchangeability

Six simulation scenarios test different combinations of X and Y exchangeability.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, Union, Sequence


class SimulationConfig:
    """Configuration for the simulation design."""
    
    def __init__(
        self,
        n_concurrent: int = 50,
        n_external: int = 200,
        p_covariates: int = 6,
        mu_c: float = 1.0,
        beta: Optional[np.ndarray] = None,
        theta_true: float = 1.0,
        sigma: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize simulation configuration.
        
        Args:
            n_concurrent: Sample size for concurrent trial
            n_external: Sample size for external (historical) trial
            p_covariates: Number of covariates
            mu_c: Mean vector for concurrent covariates
            beta: Regression coefficients (if None, will be generated)
            theta_true: True treatment effect
            sigma: Noise standard deviation
            random_seed: Random seed for reproducibility
        """
        self.n_concurrent = n_concurrent
        self.n_external = n_external
        self.p_covariates = p_covariates
        self.mu_c = mu_c
        self.theta_true = theta_true
        self.sigma = sigma
        
        # Covariance matrix: diagonal = 1, off-diagonal = 0.5
        self.Sigma = np.ones((p_covariates, p_covariates)) * 0.5
        np.fill_diagonal(self.Sigma, 1.0)
        
        # Beta coefficients (regression weights for covariates)
        if beta is None:
            # Default: small positive coefficients
            self.beta = np.array([0.5, 0.3, 0.2, 0.4, 0.3, 0.25])
        else:
            self.beta = beta
            
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)


def generate_concurrent_data(
    config: SimulationConfig,
    include_treatment: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate concurrent trial data.
    
    Xc ~ MVN(mu_c, Sigma)
    Yc ~ MVN(X*beta + Z*theta, sigma^2) where Z is treatment indicator
    
    Args:
        config: SimulationConfig object
        include_treatment: Whether to include treatment arm (Z)
        
    Returns:
        X_c: Covariate matrix (n_concurrent x p_covariates)
        Y_c: Outcome vector (n_concurrent,)
        Z_c: Treatment indicator (n_concurrent,) or None if include_treatment=False
    """
    n = config.n_concurrent
    p = config.p_covariates
    
    # Generate covariates
    mu_c_vec = np.ones(p) * config.mu_c
    X_c = np.random.multivariate_normal(mu_c_vec, config.Sigma, size=n)
    
    # Generate treatment indicator if requested
    if include_treatment:
        Z_c = np.ones(n)  # Single arm: all treated
    else:
        Z_c = None
    
    # Generate outcome
    # Y = X*beta + Z*theta + epsilon
    linear_pred = X_c @ config.beta
    if include_treatment:
        linear_pred += Z_c * config.theta_true
    
    epsilon = np.random.randn(n) * config.sigma
    Y_c = linear_pred + epsilon
    
    return X_c, Y_c, Z_c


def generate_external_data(
    config: SimulationConfig,
    mu_e: float,
    delta_e: float,
    include_treatment: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate external (historical) trial data.
    
    Xe ~ MVN(mu_e, Sigma)
    Ye ~ MVN(X*beta + delta_e, sigma^2)
    
    Args:
        config: SimulationConfig object
        mu_e: Mean for external covariates
        delta_e: Mean shift in outcome (bias/heterogeneity)
        include_treatment: Whether to include treatment arm indicator
        
    Returns:
        X_e: Covariate matrix (n_external x p_covariates)
        Y_e: Outcome vector (n_external,)
        Z_e: Treatment indicator (n_external,) or None
    """
    n = config.n_external
    p = config.p_covariates
    
    # Generate covariates with potentially different mean
    mu_e_vec = np.ones(p) * mu_e
    X_e = np.random.multivariate_normal(mu_e_vec, config.Sigma, size=n)
    
    # Generate treatment indicator if requested
    if include_treatment:
        Z_e = np.ones(n)  # Single arm: all treated
    else:
        Z_e = None
    
    # Generate outcome with mean shift delta_e
    # Y = X*beta + theta + delta_e + epsilon
    # External trial has treatment effect (theta_true) plus potential shift (delta_e)
    linear_pred = X_e @ config.beta + config.theta_true + delta_e
    
    epsilon = np.random.randn(n) * config.sigma
    Y_e = linear_pred + epsilon
    
    return X_e, Y_e, Z_e


class SimulationScenario:
    """
    Represents one of the six simulation scenarios with specific
    exchangeability properties.
    """
    
    SCENARIOS = {
        'sc1': {
            'name': 'All exchangeable',
            'X_exchangeable': 'Yes',
            'Y_exchangeable': 'Yes',
            'mu_e': lambda mu_c: mu_c,
            'delta_e': 0.0,
            'expectation': 'all good'
        },
        'sc2': {
            'name': 'X partially exchangeable, Y exchangeable',
            'X_exchangeable': 'partially Yes',
            'Y_exchangeable': 'Yes',
            'mu_e': lambda mu_c: [mu_c, 2 * mu_c],  # Two sub-scenarios
            'delta_e': 0.0,
            'expectation': 'PS good, Bayes poor'
        },
        'sc3': {
            'name': 'X exchangeable, Y partially exchangeable',
            'X_exchangeable': 'Yes',
            'Y_exchangeable': 'partially Yes',
            'mu_e': lambda mu_c: mu_c,
            'delta_e': [0.0, 1.5],  # Two sub-scenarios
            'expectation': 'PS poor, Bayes good'
        },
        'sc4': {
            'name': 'X not exchangeable, Y exchangeable',
            'X_exchangeable': 'No',
            'Y_exchangeable': 'Yes',
            'mu_e': lambda mu_c: 2 * mu_c,
            'delta_e': 0.0,
            'expectation': 'PS good, Bayes poor'
        },
        'sc5': {
            'name': 'X exchangeable, Y not exchangeable',
            'X_exchangeable': 'Yes',
            'Y_exchangeable': 'No',
            'mu_e': lambda mu_c: mu_c,
            'delta_e': 1.5,
            'expectation': 'PS poor, Bayes good'
        },
        'sc6': {
            'name': 'Nothing exchangeable',
            'X_exchangeable': 'No',
            'Y_exchangeable': 'No',
            'mu_e': lambda mu_c: 2 * mu_c,
            'delta_e': 1.5,
            'expectation': 'PS poor, Bayes poor'
        }
    }
    
    def __init__(self, scenario_id: str, config: SimulationConfig):
        """
        Initialize a simulation scenario.
        
        Args:
            scenario_id: One of 'sc1', 'sc2', ..., 'sc6'
            config: SimulationConfig object
        """
        if scenario_id not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        self.scenario_id = scenario_id
        self.config = config
        self.spec = self.SCENARIOS[scenario_id]
        
    def generate_data(
        self, 
        sub_scenario_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Generate one dataset for this scenario.
        
        Args:
            sub_scenario_idx: For scenarios with multiple sub-cases (sc2, sc3),
                            select which variant (0 or 1)
        
        Returns:
            Dictionary with keys:
                - X_concurrent: Concurrent covariates
                - Y_concurrent: Concurrent outcomes
                - Z_concurrent: Concurrent treatment indicators
                - X_external: External covariates
                - Y_external: External outcomes
                - Z_external: External treatment indicators
                - mu_e: Actual mu_e used
                - delta_e: Actual delta_e used
        """
        # Determine mu_e
        mu_e_spec = self.spec['mu_e'](self.config.mu_c)
        if isinstance(mu_e_spec, list):
            mu_e = mu_e_spec[sub_scenario_idx]
        else:
            mu_e = mu_e_spec
        
        # Determine delta_e
        delta_e_spec = self.spec['delta_e']
        if isinstance(delta_e_spec, list):
            delta_e = delta_e_spec[sub_scenario_idx]
        else:
            delta_e = delta_e_spec
        
        # Generate concurrent data
        X_c, Y_c, Z_c = generate_concurrent_data(self.config)
        
        # Generate external data
        X_e, Y_e, Z_e = generate_external_data(self.config, mu_e, delta_e)
        
        return {
            'X_concurrent': X_c,
            'Y_concurrent': Y_c,
            'Z_concurrent': Z_c,
            'X_external': X_e,
            'Y_external': Y_e,
            'Z_external': Z_e,
            'n_concurrent': self.config.n_concurrent,
            'n_external': self.config.n_external,
            'mu_e': mu_e,
            'delta_e': delta_e,
            'scenario_id': self.scenario_id,
            'scenario_name': self.spec['name'],
            'expectation': self.spec['expectation']
        }
    
    def n_sub_scenarios(self) -> int:
        """Return number of sub-scenarios (1 or 2)."""
        mu_e_spec = self.spec['mu_e'](self.config.mu_c)
        delta_e_spec = self.spec['delta_e']
        
        n_mu = len(mu_e_spec) if isinstance(mu_e_spec, list) else 1
        n_delta = len(delta_e_spec) if isinstance(delta_e_spec, list) else 1
        
        return max(n_mu, n_delta)


def convert_to_torch(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Convert numpy arrays in data dictionary to torch tensors.
    
    Args:
        data: Dictionary from generate_data()
        
    Returns:
        Same dictionary with torch tensors instead of numpy arrays
    """
    torch_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            torch_data[key] = torch.from_numpy(value).float()
        else:
            torch_data[key] = value
    return torch_data


def generate_all_scenarios(
    n_concurrent: Union[int, Sequence[int]] = 100,
    n_external: int = 200,
    random_seed: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Generate data for all six scenarios.
    
    Args:
        n_concurrent: Concurrent trial size(s). Supports int or sequence, e.g. (50, 100)
        n_external: Sample size for external trial
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping scenario_id to generated data
    """
    all_data = {}

    if isinstance(n_concurrent, int):
        nc_values = [n_concurrent]
    else:
        nc_values = list(n_concurrent)

    for nc in nc_values:
        seed_for_nc = random_seed + int(nc) if random_seed is not None else None
        config = SimulationConfig(
            n_concurrent=int(nc),
            n_external=n_external,
            random_seed=seed_for_nc
        )

        for scenario_id in ['sc1', 'sc2', 'sc3', 'sc4', 'sc5', 'sc6']:
            scenario = SimulationScenario(scenario_id, config)
            n_sub = scenario.n_sub_scenarios()

            if n_sub == 1:
                key = f"Nc{int(nc)}_{scenario_id}"
                all_data[key] = scenario.generate_data()
            else:
                for sub_idx in range(n_sub):
                    key = f"Nc{int(nc)}_{scenario_id}_{sub_idx}"
                    all_data[key] = scenario.generate_data(sub_idx)
    
    return all_data


def print_scenario_summary(data: Dict[str, np.ndarray]):
    """Print a summary of generated scenario data."""
    print(f"\nScenario: {data['scenario_id']} - {data['scenario_name']}")
    print(f"Expectation: {data['expectation']}")
    print(f"Nc = {data['n_concurrent']}, Ne = {data['n_external']}")
    print(f"mu_e = {data['mu_e']:.2f}, delta_e = {data['delta_e']:.2f}")
    print(f"Concurrent: n={len(data['Y_concurrent'])}, X shape={data['X_concurrent'].shape}")
    print(f"External:   n={len(data['Y_external'])}, X shape={data['X_external'].shape}")
    print(f"Concurrent Y: mean={data['Y_concurrent'].mean():.3f}, std={data['Y_concurrent'].std():.3f}")
    print(f"External Y:   mean={data['Y_external'].mean():.3f}, std={data['Y_external'].std():.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Simulation Design for Bayesian Dynamic Borrowing")
    print("=" * 60)
    
    # Generate all scenarios
    print("\nGenerating all scenarios with Nc=100, Ne=200...")
    all_data = generate_all_scenarios(n_concurrent=100, n_external=200, random_seed=42)
    
    # Print summaries
    for scenario_key in sorted(all_data.keys()):
        print_scenario_summary(all_data[scenario_key])
    
    # Example: Generate single scenario
    print("\n" + "=" * 60)
    print("Example: Single scenario generation (sc1)")
    print("=" * 60)
    
    config = SimulationConfig(n_concurrent=100, random_seed=123)
    scenario = SimulationScenario('sc1', config)
    data = scenario.generate_data()
    print_scenario_summary(data)
    
    # Convert to torch
    torch_data = convert_to_torch(data)
    print(f"\nConverted to torch tensors:")
    print(f"X_concurrent type: {type(torch_data['X_concurrent'])}")
    print(f"Y_concurrent shape: {torch_data['Y_concurrent'].shape}")
