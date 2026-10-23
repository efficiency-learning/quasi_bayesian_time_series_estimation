import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
import ot  # For Wasserstein distance calculation
from tqdm import tqdm
import os
from scipy.stats import norm

class QuasiBayesianClustering:
    def __init__(self, p, d, lambda_param=1.0, lambda_adaptive_fn=None, 
                 prior_type='uniform', prior_params=None, proposal_sigma=1.0,
                 ignore_normalization=False):
        self.p = p
        self.d = d
        self.current_lambda = lambda_param
        self.lambda_adaptive_fn = lambda_adaptive_fn
        self.lambda_history = [lambda_param]
        self.ignore_normalization = ignore_normalization
        
        self.prior_type = prior_type
        self.prior_params = prior_params or {}
        self._setup_prior()
        
        self.centers = self._initialize_centers()
        self.center_history = [self.centers.copy()]
        
        self.proposal_sigma = proposal_sigma
        
        self.cumulative_loss_value = 0
        self.loss_history = []
        self.regret_history = []
        self.mean_regret_history = []
        
        if self.lambda_adaptive_fn is not None:
            self.t = 0
            self.update_lambda(0)
    
    def _initialize_centers(self):
        if self.prior_type == 'uniform':
            return np.random.uniform(-5, 5, size=(self.p, self.d))
            
        elif self.prior_type == 'gaussian':
            mean = self.prior_params.get('mean', np.zeros((self.p, self.d)))
            variance = self.prior_params.get('variance', 1.0)
            return mean + np.random.normal(0, np.sqrt(variance), size=(self.p, self.d))
            
        elif self.prior_type == 'custom':
            if 'init_fn' in self.prior_params:
                return self.prior_params['init_fn'](self.p, self.d)
            else:
                return np.random.uniform(-5, 5, size=(self.p, self.d))
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
    
    def _setup_prior(self):
        if self.prior_type == "uniform":
            self.log_prior = lambda c: 0.0
            
        elif self.prior_type == "gaussian":
            mean = self.prior_params.get("mean", np.zeros((self.p, self.d)))
            variance = self.prior_params.get("variance", 1.0)
            
            def gaussian_log_prior(c):
                diff = c - mean
                return -0.5 * np.sum(diff**2) / variance
                
            self.log_prior = gaussian_log_prior
            
        elif self.prior_type == "custom":
            if "log_prior_fn" not in self.prior_params:
                raise ValueError("For 'custom' prior_type, you must provide a 'log_prior_fn' in prior_params")
            
            self.log_prior = self.prior_params["log_prior_fn"]
            
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type}")
    
    def quasi_posterior(self, c, data, prev_loss=None):
        if prev_loss is None:
            loss_value = self.cumulative_loss(c, data)
        else:
            loss_value = prev_loss + self.loss(c, data[-1])
            
            prev_centers = self.center_history[-1]
            variance_term = (self.current_lambda / 2) * (self.loss(c, data[-1]) - self.loss(prev_centers, data[-1]))**2
            loss_value += variance_term
        
        log_likelihood = -self.current_lambda * loss_value
        log_prior_val = self.log_prior(c)
        log_posterior = log_likelihood + log_prior_val
        
        return np.exp(log_posterior)
        
    def loss(self, c, x):
        x = np.array(x).reshape(-1, self.d)
        distances = np.sum((c - x)**2, axis=1)
        return np.min(distances)
    
    def cumulative_loss(self, c, data):
        return sum(self.loss(c, x) for x in data)
    
    def quasi_posterior_measure(self, c, data, prev_loss=None):
        if prev_loss is None:
            loss_value = self.cumulative_loss(c, data)
        else:
            loss_value = prev_loss + self.loss(c, data[-1])
            
            prev_centers = self.center_history[-1]
            variance_term = (self.current_lambda / 2) * (self.loss(c, data[-1]) - self.loss(prev_centers, data[-1]))**2
            loss_value += variance_term
        
        prior_measure = self.prior_density(c)
        
        return np.exp(-self.current_lambda * loss_value) * prior_measure
    
    def integrate_quasi_posterior(self, data, num_samples=1000, method='monte_carlo'):
        if method == 'monte_carlo':
            samples = [self._initialize_centers() for _ in range(num_samples)]
            
            integrand_values = np.array([self.quasi_posterior_measure(c, data) for c in samples])
            
            integral = np.mean(integrand_values)
            
        elif method == 'importance':
            samples = []
            weights = []
            
            for _ in range(num_samples):
                c = self._initialize_centers()
                samples.append(c)
                
                proposal_density = self.proposal_density(c)
                target_density = self.quasi_posterior_measure(c, data)
                weights.append(target_density / proposal_density)
            
            weights = np.array(weights)
            weights /= np.sum(weights)
            
            integral = np.mean(weights)
            
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        return integral
    
    def normalized_quasi_posterior(self, c, data, integration_method='monte_carlo'):
        unnormalized = self.quasi_posterior_measure(c, data)
        normalization = self.integrate_quasi_posterior(data, method=integration_method)
        return unnormalized / normalization
    
    def _propose_new_centers(self, current_centers):
        noise = np.random.normal(0, self.proposal_sigma, size=current_centers.shape)
        proposed_centers = current_centers + noise
        
        return proposed_centers

    def sample_from_quasi_posterior(self, data, num_samples=1, num_parallel=2, n_init=3):
        """
        Optimized version of PACBO sampling with fewer evaluations and simpler acquisition.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        import warnings
        warnings.filterwarnings('ignore')

        # Define the search space for cluster centers
        d = self.d  # dimension of the space
        p = self.p  # number of clusters
        bounds = np.array([[-5, 5]] * (d * p))  # bounds for each coordinate of each center
        
        def objective(x):
            centers = x.reshape(p, d)
            return -self.quasi_posterior_measure(centers, data)
        
        # Initialize with fewer random configurations
        X = np.random.uniform(
            low=bounds[:, 0], 
            high=bounds[:, 1], 
            size=(n_init, d * p)
        )
        y = np.array([objective(x) for x in X])
        
        # Simpler GP kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        
        # Fewer PACBO iterations
        n_iter = 5  # Reduced from 10
        for _ in range(n_iter):
            gp.fit(X, y)
            
            # Generate fewer candidate points
            X_candidates = np.random.uniform(
                low=bounds[:, 0],
                high=bounds[:, 1],
                size=(100, d * p)  # Reduced from 1000
            )
            
            # Simple mean prediction for speed
            mean = gp.predict(X_candidates)
            
            # Select top configurations
            top_indices = np.argsort(mean)[:num_parallel]
            X_new = X_candidates[top_indices]
            y_new = np.array([objective(x) for x in X_new])
            
            # Update dataset
            X = np.vstack((X, X_new))
            y = np.append(y, y_new)
        
        # Select best configurations
        best_indices = np.argsort(y)[:num_samples]
        best_samples = [X[i].reshape(p, d) for i in best_indices]
        
        return best_samples

    def prior_density(self, c):
        if self.prior_type == 'uniform':
            return 1.0
        elif self.prior_type == 'gaussian':
            return np.exp(-0.5 * np.sum(c**2) / self.prior_params['sigma']**2)
        elif self.prior_type == 'custom':
            return self.prior_params['density_fn'](c)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
    
    def proposal_density(self, c):
        if len(self.center_history) > 0:
            historical_centers = np.array(self.center_history)
            n_historical = len(historical_centers)
            
            densities = []
            for center in historical_centers:
                diff = c - center
                density = np.exp(-0.5 * np.sum(diff**2) / self.proposal_sigma**2)
                densities.append(density)
            
            return np.mean(densities) / n_historical
        else:
            return self.prior_density(c)
    
    def pacbo_sample_centers(self, data, num_parallel=4, iterations=20):
        candidates = [self.centers.copy() + np.random.normal(0, 0.2, size=self.centers.shape) 
                      for _ in range(num_parallel)]
        
        best_centers = self.centers.copy()
        best_prob = self.quasi_posterior(best_centers, data)
        
        for _ in range(iterations):
            probs = [self.quasi_posterior(c, data) for c in candidates]
            
            max_idx = np.argmax(probs)
            if probs[max_idx] > best_prob:
                best_centers = candidates[max_idx].copy()
                best_prob = probs[max_idx]
            
            for i in range(num_parallel):
                if np.random.rand() < 0.7:
                    candidates[i] = 0.8 * candidates[i] + 0.2 * best_centers
                    candidates[i] += np.random.normal(0, 0.05, size=candidates[i].shape)
            
            if len(data) > 0:
                for i in range(num_parallel):
                    if np.random.rand() < 0.3:
                        random_point = data[np.random.randint(0, len(data))]
                        random_center_idx = np.random.randint(0, self.p)
                        candidates[i][random_center_idx] = 0.7 * candidates[i][random_center_idx] + 0.3 * random_point
        
        return best_centers
        
    def sample_centers(self, data, num_samples=10):
        return self.pacbo_sample_centers(data)
    
    def predict_distribution(self, prev_data):
        def quasi_posterior_pdf(x):
            num_samples = 20  # Reduced from 100
            centers_samples = self.sample_from_quasi_posterior(
                prev_data, 
                num_samples=num_samples,
                num_parallel=2,  # Reduced from 4
                n_init=3  # Reduced from 5
            )
            
            if self.ignore_normalization:
                N_c = 1.0
            else:
                def calc_normalization(c):
                    x_samples = np.random.uniform(-5, 5, size=(50, self.d))  # Reduced from 100
                    exp_terms = np.array([np.exp(-self.current_lambda * self.loss(c, x_i)) for x_i in x_samples])
                    return np.mean(exp_terms) * 10**self.d
            
            probs = []
            for c in centers_samples:
                if not self.ignore_normalization:
                    N_c = calc_normalization(c)
                likelihood = np.exp(-self.current_lambda * self.loss(c, x)) / N_c
                probs.append(likelihood)
            
            return np.mean(probs)
        
        return quasi_posterior_pdf
    
    def update_lambda(self, t=None):
        if t is not None:
            self.t = t
        else:
            self.t += 1
            
        if self.lambda_adaptive_fn is not None:
            self.current_lambda = self.lambda_adaptive_fn(self.t, self.current_lambda)
        self.lambda_history.append(self.current_lambda)
        return self.current_lambda
        
    def update(self, x_t):
        """Optimized update method"""
        self.update_lambda()
        self.loss_history.append(self.loss(self.centers, x_t))
        self.cumulative_loss_value = sum(self.loss_history)
        
        # Sample new centers with reduced parameters
        new_centers = self.sample_from_quasi_posterior(
            data=[x_t],
            num_samples=1,
            num_parallel=2,  # Reduced from 4
            n_init=3  # Reduced from 5
        )[0]
        
        self.centers = new_centers
        self.center_history.append(self.centers.copy())
        
        return self.centers


def generate_data(p, d, T, distribution_type="gaussian_mixture", 
                cluster_variance=0.5, mixing_prob=None, distribution_params=None,
                add_noise=False, noise_epsilon=0.1):
    if distribution_params is None:
        distribution_params = {}
    
    if mixing_prob is None:
        mixing_prob = np.ones(p) / p
    
    if distribution_type == "gaussian_mixture":
        means = 3 * np.random.randn(p, d)
        covs = [cluster_variance * np.eye(d) for _ in range(p)]
        
        true_dists = [multivariate_normal(mean=means[i], cov=covs[i]) for i in range(p)]
        
        data = []
        dist_indices = []
        
        for _ in range(T):
            dist_idx = np.random.choice(p, p=mixing_prob)
            dist_indices.append(dist_idx)
            
            x_t = true_dists[dist_idx].rvs()
            
            if add_noise:
                noise = np.random.uniform(-noise_epsilon, noise_epsilon, size=d)
                x_t = x_t + noise
            
            data.append(x_t)
    
    elif distribution_type == "t_mixture":
        means = 3 * np.random.randn(p, d)
        df = distribution_params.get("degrees_of_freedom", 3)
        
        class TDistribution:
            def __init__(self, mean, df, d):
                self.mean = mean
                self.df = df
                self.d = d
            
            def pdf(self, x):
                delta = x - self.mean
                return (1 + (1/self.df) * np.sum(delta**2))**(-(self.df + self.d)/2)
            
            def rvs(self):
                scale = np.sqrt(self.df / np.random.chisquare(self.df))
                return self.mean + scale * np.random.normal(0, 1, self.d)
        
        true_dists = [TDistribution(means[i], df, d) for i in range(p)]
        
        data = []
        dist_indices = []
        
        for _ in range(T):
            dist_idx = np.random.choice(p, p=mixing_prob)
            dist_indices.append(dist_idx)
            x_t = true_dists[dist_idx].rvs()
            
            if add_noise:
                noise = np.random.uniform(-noise_epsilon, noise_epsilon, size=d)
                x_t = x_t + noise
            
            data.append(x_t)
    
    elif distribution_type == "gmm_timevarying":
        base_means = 3 * np.random.randn(p, d)
        drift_scale = distribution_params.get("drift_scale", 0.05)
        mean_trajectories = []
        
        true_dists = []
        data = []
        dist_indices = []
        
        current_means = base_means.copy()
        mean_trajectories.append(current_means.copy())
        
        for t in range(T):
            current_means += drift_scale * np.random.randn(p, d)
            mean_trajectories.append(current_means.copy())
            
            current_dists = [multivariate_normal(mean=current_means[i], cov=cluster_variance * np.eye(d)) 
                            for i in range(p)]
            
            dist_idx = np.random.choice(p, p=mixing_prob)
            dist_indices.append(dist_idx)
            x_t = current_dists[dist_idx].rvs()
            
            if add_noise:
                noise = np.random.uniform(-noise_epsilon, noise_epsilon, size=d)
                x_t = x_t + noise
            
            data.append(x_t)
            true_dists.append(current_dists)
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return np.array(data), true_dists, dist_indices


def calculate_wasserstein_distance(pred_pdf, true_pdf, d, grid_size=10, grid_range=(-5, 5)):
    """Optimized Wasserstein distance calculation with smaller grid"""
    if d == 1:
        grid_points = np.linspace(grid_range[0], grid_range[1], grid_size)
        
        pred_values = np.array([pred_pdf(np.array([x])) for x in grid_points])
        true_values = np.array([true_pdf.pdf(np.array([x])) for x in grid_points])
        
        pred_values = pred_values / np.sum(pred_values)
        true_values = true_values / np.sum(true_values)
        
        return ot.emd2_1d(grid_points, grid_points, pred_values, true_values)
    
    elif d == 2:
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        
        pred_values = np.array([pred_pdf(point) for point in grid_points])
        true_values = np.array([true_pdf.pdf(point) for point in grid_points])
        
        pred_values = pred_values / np.sum(pred_values)
        true_values = true_values / np.sum(true_values)
        
        M = ot.dist(grid_points, grid_points)
        
        return ot.emd2(pred_values, true_values, M)
    
    else:
        raise ValueError("This function only supports d=1 or d=2 for visualization simplicity")


def calculate_distribution_means(pred_pdf, true_pdf, d, grid_size=20, grid_range=(-5, 5)):
    if d == 1:
        grid_points = np.linspace(grid_range[0], grid_range[1], grid_size)
        
        pred_values = np.array([pred_pdf(np.array([x])) for x in grid_points])
        true_values = np.array([true_pdf.pdf(np.array([x])) for x in grid_points])
        
        pred_values = pred_values / np.sum(pred_values)
        true_values = true_values / np.sum(true_values)
        
        pred_mean = np.sum(grid_points * pred_values)
        true_mean = np.sum(grid_points * true_values)
        
        mean_distance = np.abs(pred_mean - true_mean)
        
        return mean_distance, np.array([pred_mean]), np.array([true_mean])
    
    elif d == 2:
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        
        pred_values = np.array([pred_pdf(point) for point in grid_points])
        true_values = np.array([true_pdf.pdf(point) for point in grid_points])
        
        pred_values = pred_values / np.sum(pred_values)
        true_values = true_values / np.sum(true_values)
        
        pred_mean = np.zeros(d)
        true_mean = np.zeros(d)
        
        for i in range(grid_points.shape[0]):
            pred_mean += grid_points[i] * pred_values[i]
            true_mean += grid_points[i] * true_values[i]
        
        mean_distance = np.sqrt(np.sum((pred_mean - true_mean)**2))
        
        return mean_distance, pred_mean, true_mean
    
    else:
        raise ValueError("This function only supports d=1 or d=2 for visualization simplicity")


def run_experiment(p, d, T, lambda_param, lambda_adaptive_fn=None,
                  distribution_type="gaussian_mixture", distribution_params=None,
                  add_noise=False, noise_epsilon=0.1, ignore_normalization=False):
    data, true_dists, dist_indices = generate_data(
        p, d, T, distribution_type=distribution_type, distribution_params=distribution_params,
        add_noise=add_noise, noise_epsilon=noise_epsilon
    )
    
    model = QuasiBayesianClustering(p, d, lambda_param, lambda_adaptive_fn, 
                                   ignore_normalization=ignore_normalization)
    
    wasserstein_distances = []
    cumulative_regret = []
    mean_distances = []
    mean_cumulative_regret = []
    
    print(f"Running experiment with {distribution_type} distributions, p={p}, lambda={lambda_param}" + 
          (", with noise ε=" + str(noise_epsilon) if add_noise else ""))
    
    for t in tqdm(range(1, T)):
        prev_data = data[:t]
        pred_dist = model.predict_distribution(prev_data)
        
        if distribution_type == "gmm_timevarying":
            true_dist = true_dists[t][dist_indices[t]]
        else:
            true_dist = true_dists[dist_indices[t]]
        
        w_dist = calculate_wasserstein_distance(pred_dist, true_dist, d)
        wasserstein_distances.append(w_dist)
        
        mean_dist, _, _ = calculate_distribution_means(pred_dist, true_dist, d)
        mean_distances.append(mean_dist)
        
        model.update(data[t])
        
        if len(cumulative_regret) == 0:
            cumulative_regret.append(w_dist)
            mean_cumulative_regret.append(mean_dist)
        else:
            cumulative_regret.append(cumulative_regret[-1] + w_dist)
            mean_cumulative_regret.append(mean_cumulative_regret[-1] + mean_dist)
    
    return wasserstein_distances, cumulative_regret, mean_distances, mean_cumulative_regret, model.lambda_history


def plot_results(wasserstein_distances, cumulative_regret, mean_distances=None, mean_cumulative_regret=None):
    if mean_distances is None or mean_cumulative_regret is None:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(wasserstein_distances, '-o')
        plt.title('Wasserstein Distance over Time')
        plt.xlabel('Time step t')
        plt.ylabel('Wasserstein Distance')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(cumulative_regret, '-o')
        plt.title('Cumulative Wasserstein Regret')
        plt.xlabel('Time step t')
        plt.ylabel('Cumulative Regret')
        plt.grid(True)
    else:
        plt.figure(figsize=(14, 12))
        
        plt.subplot(2, 2, 1)
        plt.plot(wasserstein_distances, '-o')
        plt.title('Wasserstein Distance over Time')
        plt.xlabel('Time step t')
        plt.ylabel('Wasserstein Distance')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(cumulative_regret, '-o')
        plt.title('Cumulative Wasserstein Regret')
        plt.xlabel('Time step t')
        plt.ylabel('Cumulative Regret')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(mean_distances, '-o')
        plt.title('Mean Distance over Time')
        plt.xlabel('Time step t')
        plt.ylabel('Euclidean Distance between Means')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(mean_cumulative_regret, '-o')
        plt.title('Cumulative Mean Regret')
        plt.xlabel('Time step t')
        plt.ylabel('Cumulative Mean Regret')
        plt.grid(True)
    
    plt.tight_layout()
    #plt.show()


def plot_results_log_scale(wasserstein_distances, cumulative_regret, mean_distances=None, mean_cumulative_regret=None):
    if mean_distances is None or mean_cumulative_regret is None:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(wasserstein_distances, '-o')
        plt.yscale('log')
        plt.title('Wasserstein Distance over Time (Log Scale)')
        plt.xlabel('Time step t')
        plt.ylabel('Log(Wasserstein Distance)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.log1p(cumulative_regret), '-o')
        plt.title('Log(1+Cumulative Wasserstein Regret)')
        plt.xlabel('Time step t')
        plt.ylabel('Log(1+Cumulative Regret)')
        plt.grid(True)
    else:
        plt.figure(figsize=(14, 12))
        
        plt.subplot(2, 2, 1)
        plt.plot(wasserstein_distances, '-o')
        plt.yscale('log')
        plt.title('Wasserstein Distance over Time (Log Scale)')
        plt.xlabel('Time step t')
        plt.ylabel('Log(Wasserstein Distance)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(np.log1p(cumulative_regret), '-o')
        plt.title('Log(1+Cumulative Wasserstein Regret)')
        plt.xlabel('Time step t')
        plt.ylabel('Log(1+Cumulative Regret)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(mean_distances, '-o')
        plt.yscale('log')
        plt.title('Mean Distance over Time (Log Scale)')
        plt.xlabel('Time step t')
        plt.ylabel('Log(Mean Distance)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(np.log1p(mean_cumulative_regret), '-o')
        plt.title('Log(1+Cumulative Mean Regret)')
        plt.xlabel('Time step t')
        plt.ylabel('Log(1+Cumulative Mean Regret)')
        plt.grid(True)
    
    plt.tight_layout()
    #plt.show()


def compare_lambda_values(p=3, d=2, T=100, lambda_values=[0.1, 0.5, 1.0, 2.0],
                    distribution_type="gaussian_mixture", distribution_params=None):
    all_regrets = []
    all_mean_regrets = []
    all_lambda_histories = []
    
    for lambda_param in lambda_values:
        print(f"Running experiment with λ = {lambda_param}")
        _, regret, _, mean_regret, model = run_experiment(
            p, d, T, lambda_param, 
            distribution_type=distribution_type, 
            distribution_params=distribution_params
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        all_lambda_histories.append(model.lambda_history)
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    for i, lambda_param in enumerate(lambda_values):
        plt.plot(all_regrets[i], label=f'λ = {lambda_param}')
    
    plt.title('Cumulative Wasserstein Regret for Different λ Values')
    plt.xlabel('Time step t')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for i, lambda_param in enumerate(lambda_values):
        plt.plot(all_mean_regrets[i], label=f'λ = {lambda_param}')
    
    plt.title('Cumulative Mean Regret for Different λ Values')
    plt.xlabel('Time step t')
    plt.ylabel('Cumulative Mean Regret')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for i, lambda_param in enumerate(lambda_values):
        plt.plot(all_lambda_histories[i], label=f'Initial λ = {lambda_param}')
    
    plt.title('Lambda Values Over Time')
    plt.xlabel('Time step t')
    plt.ylabel('λ value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    #plt.show()


def compare_adaptive_lambda(p=3, d=2, T=100, base_lambda=1.0,
                           adaptive_fns=None, adaptive_fn_names=None,
                           distribution_type="gaussian_mixture", distribution_params=None):
    if adaptive_fns is None:
        adaptive_fns = [
            None,
            lambda t, base: base * np.sqrt(t),
            lambda t, base: base * (1 + np.log(1 + t)),
            lambda t, base: base * (1 / (1 + 0.1 * t)),
            lambda t, base: base * (1 + 0.2 * np.sin(0.1 * t))
        ]
    
    if adaptive_fn_names is None:
        adaptive_fn_names = [
            "Constant λ",
            "λ × √t",
            "λ × (1 + log(1+t))",
            "λ × (1 / (1 + 0.1t))",
            "λ × (1 + 0.2sin(0.1t))"
        ]
    
    all_regrets = []
    all_mean_regrets = []
    all_lambda_histories = []
    
    for i, fn in enumerate(adaptive_fns):
        name = adaptive_fn_names[i]
        print(f"Running experiment with {name}")
        _, regret, _, mean_regret, model = run_experiment(
            p, d, T, base_lambda, fn,
            distribution_type=distribution_type, 
            distribution_params=distribution_params
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        all_lambda_histories.append(model.lambda_history)
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 1, 1)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(all_regrets[i], label=name)
    
    plt.title('Cumulative Wasserstein Regret for Different Adaptive λ Functions')
    plt.xlabel('Time step t')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(all_mean_regrets[i], label=name)
    
    plt.title('Cumulative Mean Regret for Different Adaptive λ Functions')
    plt.xlabel('Time step t')
    plt.ylabel('Cumulative Mean Regret')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(all_lambda_histories[i], label=name)
    
    plt.title('λ Values Over Time for Different Adaptive Functions')
    plt.xlabel('Time step t')
    plt.ylabel('λ value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    #plt.show()


def compare_distribution_types(p=3, d=2, T=100, lambda_param=1.0):
    distribution_types = [
        "gaussian_mixture",
        "t_mixture",
        "gmm_timevarying"
    ]
    
    distribution_names = [
        "Gaussian Mixture",
        "t-Distribution Mixture",
        "Time-Varying Gaussian Mixture"
    ]
    
    all_regrets = []
    all_mean_regrets = []
    
    for i, dist_type in enumerate(distribution_types):
        print(f"Running experiment with {distribution_names[i]} distributions...")
        _, regret, _, mean_regret, _ = run_experiment(
            p, d, T, lambda_param, distribution_type=dist_type
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for i, name in enumerate(distribution_names):
        plt.plot(all_regrets[i], label=name)
    
    plt.title('Cumulative Wasserstein Regret for Different Distribution Types')
    plt.xlabel('Time step t')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for i, name in enumerate(distribution_names):
        plt.plot(all_mean_regrets[i], label=name)
    
    plt.title('Cumulative Mean Regret for Different Distribution Types')
    plt.xlabel('Time step t')
    plt.ylabel('Cumulative Mean Regret')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    #plt.show()


def visualize_clustering(model, data, title="Clustering Result"):
    if data.shape[1] != 2:
        print("Visualization is only supported for 2D data.")
        return
    
    plt.figure(figsize=(10, 8))
    
    centers = model.centers
    
    assignments = []
    for x in data:
        distances = np.sum((centers - x.reshape(1, -1))**2, axis=1)
        assignments.append(np.argmin(distances))
    
    for i in range(model.p):
        cluster_points = data[np.array(assignments) == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
    
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, color='black', label='Centers')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    #plt.show()


def visualize_lambda_effect_on_centers(base_lambda=1.0, T=100, p=3, d=2):
    np.random.seed(42)
    
    data, true_dists, dist_indices = generate_data(p, d, T)
    
    lambda_values = [0.1, 1.0, 5.0]
    
    fig, axes = plt.subplots(1, len(lambda_values), figsize=(15, 5))
    
    for i, lambda_param in enumerate(lambda_values):
        model = QuasiBayesianClustering(p, d, lambda_param)
        
        for t in range(T):
            model.update(data[t])
        
        centers = model.centers
        assignments = []
        for x in data:
            distances = np.sum((centers - x.reshape(1, -1))**2, axis=1)
            assignments.append(np.argmin(distances))
        
        ax = axes[i]
        
        for j in range(p):
            cluster_points = data[np.array(assignments) == j]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5)
        
        ax.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, color='red')
        
        if isinstance(true_dists[0], multivariate_normal):
            for j in range(p):
                ax.scatter(true_dists[j].mean[0], true_dists[j].mean[1], 
                          marker='*', s=300, color='green', edgecolors='black')
        
        ax.set_title(f'λ = {lambda_param}')
        ax.grid(True)
    
    plt.tight_layout()
    #plt.show()


def compare_adaptive_lambda_log_scale(p=3, d=2, T=100, base_lambda=1.0,
                           adaptive_fns=None, adaptive_fn_names=None,
                           distribution_type="gaussian_mixture", distribution_params=None):
    if adaptive_fns is None:
        adaptive_fns = [
            None,
            lambda t, base: base * np.sqrt(t),
            lambda t, base: base * (1 + np.log(1 + t)),
            lambda t, base: base * (1 / (1 + 0.1 * t)),
            lambda t, base: base * (1 + 0.2 * np.sin(0.1 * t))
        ]
    
    if adaptive_fn_names is None:
        adaptive_fn_names = [
            "Constant λ",
            "λ × √t",
            "λ × (1 + log(1+t))",
            "λ × (1 / (1 + 0.1t))",
            "λ × (1 + 0.2sin(0.1t))"
        ]
    
    all_regrets = []
    all_mean_regrets = []
    all_lambda_histories = []
    
    for i, fn in enumerate(adaptive_fns):
        name = adaptive_fn_names[i]
        print(f"Running experiment with {name}")
        _, regret, _, mean_regret, model = run_experiment(
            p, d, T, base_lambda, fn,
            distribution_type=distribution_type, 
            distribution_params=distribution_params
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        all_lambda_histories.append(model.lambda_history)
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 1, 1)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(np.log1p(all_regrets[i]), label=name)
    
    plt.title('Log(1+Cumulative Wasserstein Regret) for Different Adaptive λ Functions')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(np.log1p(all_mean_regrets[i]), label=name)
    
    plt.title('Log(1+Cumulative Mean Regret) for Different Adaptive λ Functions')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Mean Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(all_lambda_histories[i], label=name)
    
    plt.title('λ Values Over Time for Different Adaptive Functions')
    plt.xlabel('Time step t')
    plt.ylabel('λ value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    #plt.show()


def run_lambda_experiment(p=3, d=2, T=100, lambda_values=[0.1, 0.5, 1.0, 2.0, 5.0],
                          distribution_type="gaussian_mixture", output_dir='plots'):
    all_regrets = []
    all_mean_regrets = []
    labels = []
    
    for lambda_param in lambda_values:
        print(f"Running experiment with λ = {lambda_param}")
        _, regret, _, mean_regret, _ = run_experiment(
            p, d, T, lambda_param, 
            distribution_type=distribution_type
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        labels.append(f'λ = {lambda_param}')
    
    plot_regret_log_scale(all_regrets, all_mean_regrets, labels, 
                          f"Different Lambda Values (p={p}, {distribution_type})", output_dir)


def run_adaptive_lambda_experiment(p=3, d=2, T=100, base_lambda=1.0, 
                                  adaptive_fns=None, adaptive_fn_names=None,
                                  distribution_type="gaussian_mixture", output_dir='plots'):
    if adaptive_fns is None:
        adaptive_fns = [
            None,
            lambda t, base: base * np.sqrt(t),
            lambda t, base: base * (1 + np.log(1 + t)),
            lambda t, base: base * (1 / (1 + 0.1 * t)),
            lambda t, base: base * (1 + 0.2 * np.sin(0.1 * t))
        ]
    
    if adaptive_fn_names is None:
        adaptive_fn_names = [
            "Constant λ",
            "λ × √t (Increasing)",
            "λ × (1 + log(1+t)) (Slow increasing)",
            "λ × (1 / (1 + 0.1t)) (Slow decreasing)",
            "λ × (1 + 0.2sin(0.1t))"
        ]
    
    all_regrets = []
    all_mean_regrets = []
    all_lambda_histories = []
    
    for i, fn in enumerate(adaptive_fns):
        name = adaptive_fn_names[i]
        print(f"Running experiment with {name}")
        _, regret, _, mean_regret, model = run_experiment(
            p, d, T, base_lambda, fn,
            distribution_type=distribution_type
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        all_lambda_histories.append(model.lambda_history)
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 1, 1)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(np.log1p(all_regrets[i]), label=name)
    
    plt.title(f'Log(1+Cumulative Wasserstein Regret) - Adaptive λ Functions (p={p}, {distribution_type})')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(np.log1p(all_mean_regrets[i]), label=name)
    
    plt.title(f'Log(1+Cumulative Mean Regret) - Adaptive λ Functions (p={p}, {distribution_type})')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Mean Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(all_lambda_histories[i], label=name)
    
    plt.title('λ Values Over Time for Different Adaptive Functions')
    plt.xlabel('Time step t')
    plt.ylabel('λ value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    filename = f"adaptive_lambda_functions_p{p}_{distribution_type}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, filename)}")
    
    #plt.show()


def run_noise_experiment(p=3, d=2, T=100, lambda_param=1.0, 
                        noise_epsilons=[0, 0.1, 0.5, 1.0, 2.0],
                        distribution_type="gaussian_mixture", 
                        lambda_adaptive_fn=None, 
                        output_dir='plots',
                        ignore_normalization=False):
    all_regrets = []
    all_mean_regrets = []
    labels = []
    
    lambda_type = "Adaptive λ" if lambda_adaptive_fn else f"λ = {lambda_param}"
    
    print(f"Running experiment with no noise, {lambda_type}")
    _, regret, _, mean_regret, _ = run_experiment(
        p, d, T, lambda_param, 
        lambda_adaptive_fn=lambda_adaptive_fn,
        distribution_type=distribution_type,
        add_noise=False,
        ignore_normalization=ignore_normalization
    )
    all_regrets.append(regret)
    all_mean_regrets.append(mean_regret)
    labels.append(f'No noise')
    
    for epsilon in noise_epsilons:
        if epsilon == 0:
            continue
            
        print(f"Running experiment with noise ε = {epsilon}, {lambda_type}")
        _, regret, _, mean_regret, _ = run_experiment(
            p, d, T, lambda_param, 
            lambda_adaptive_fn=lambda_adaptive_fn,
            distribution_type=distribution_type,
            add_noise=True,
            noise_epsilon=epsilon,
            ignore_normalization=ignore_normalization
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        labels.append(f'ε = {epsilon}')
    
    plt.figure(figsize=(15, 10))
    
    lambda_title = "Adaptive λ" if lambda_adaptive_fn else f"λ={lambda_param}"
    
    plt.subplot(2, 1, 1)
    for i, label in enumerate(labels):
        plt.plot(np.log1p(all_regrets[i]), label=label)
    
    plt.title(f'Log(1+Cumulative Wasserstein Regret) - Different Noise Levels (p={p}, {lambda_title}, {distribution_type})')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for i, label in enumerate(labels):
        plt.plot(np.log1p(all_mean_regrets[i]), label=label)
    
    plt.title(f'Log(1+Cumulative Mean Regret) - Different Noise Levels (p={p}, {lambda_title}, {distribution_type})')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Mean Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    lambda_str = "adaptive_lambda" if lambda_adaptive_fn else f"lambda{lambda_param}"
    filename = f"noise_experiment_p{p}_{lambda_str}_{distribution_type}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, filename)}")
    
    #plt.show()


def run_noise_adaptive_lambda_experiment(p=3, d=2, T=100, base_lambda=1.0,
                                        noise_epsilon=0.5,
                                        adaptive_fns=None, adaptive_fn_names=None,
                                        distribution_type="gaussian_mixture", 
                                        output_dir='plots'):
    if adaptive_fns is None:
        adaptive_fns = [
            None,
            lambda t, base: base * np.sqrt(t),
            lambda t, base: base * (1 + np.log(1 + t)),
            lambda t, base: base * (1 / (1 + 0.1 * t)),
            lambda t, base: base / t if t > 0 else base
        ]
    
    if adaptive_fn_names is None:
        adaptive_fn_names = [
            "Constant λ",
            "λ × √t (Increasing)",
            "λ × (1 + log(1+t)) (Slow increasing)",
            "λ × (1 / (1 + 0.1t)) (Slow decreasing)",
            "λ / t (Fast decreasing)"
        ]
    
    all_regrets = []
    all_mean_regrets = []
    all_lambda_histories = []
    
    for i, fn in enumerate(adaptive_fns):
        name = adaptive_fn_names[i]
        print(f"Running experiment with noise ε = {noise_epsilon}, {name}")
        _, regret, _, mean_regret, model = run_experiment(
            p, d, T, base_lambda, fn,
            distribution_type=distribution_type,
            add_noise=True,
            noise_epsilon=noise_epsilon
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        all_lambda_histories.append(model.lambda_history)
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 1, 1)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(np.log1p(all_regrets[i]), label=name)
    
    plt.title(f'Log(1+Cumulative Wasserstein Regret) - Adaptive λ with Noise ε={noise_epsilon} (p={p}, {distribution_type})')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(np.log1p(all_mean_regrets[i]), label=name)
    
    plt.title(f'Log(1+Cumulative Mean Regret) - Adaptive λ with Noise ε={noise_epsilon} (p={p}, {distribution_type})')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Mean Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    for i, name in enumerate(adaptive_fn_names):
        plt.plot(all_lambda_histories[i], label=name)
    
    plt.title('λ Values Over Time for Different Adaptive Functions')
    plt.xlabel('Time step t')
    plt.ylabel('λ value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    filename = f"noise_epsilon{noise_epsilon}_adaptive_lambda_p{p}_{distribution_type}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, filename)}")
    
    #plt.show()


def plot_regret_log_scale(all_regrets, all_mean_regrets, labels, title, output_dir='plots'):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for i, label in enumerate(labels):
        plt.plot(np.log1p(all_regrets[i]), label=label)
    
    plt.title(f'Log(1+Cumulative Wasserstein Regret) - {title}')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for i, label in enumerate(labels):
        plt.plot(np.log1p(all_mean_regrets[i]), label=label)
    
    plt.title(f'Log(1+Cumulative Mean Regret) - {title}')
    plt.xlabel('Time step t')
    plt.ylabel('Log(1+Cumulative Mean Regret)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    filename = f"{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, filename)}")
    
    #plt.show()


def run_distribution_experiment(p=3, d=2, T=100, lambda_param=1.0, output_dir='plots'):
    distribution_types = [
        "gaussian_mixture",
        "t_mixture",
        "gmm_timevarying"
    ]
    
    distribution_names = [
        "Gaussian Mixture",
        "t-Distribution Mixture",
        "Time-Varying Gaussian Mixture"
    ]
    
    all_regrets = []
    all_mean_regrets = []
    labels = []
    
    for i, dist_type in enumerate(distribution_types):
        print(f"Running experiment with {distribution_names[i]} distributions...")
        _, regret, _, mean_regret, _ = run_experiment(
            p, d, T, lambda_param, distribution_type=dist_type
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        labels.append(distribution_names[i])
    
    plot_regret_log_scale(all_regrets, all_mean_regrets, labels, 
                         f"Different Distribution Types (p={p}, λ={lambda_param})", output_dir)


def run_p_experiment(p_values=[3, 5, 10, 20, 50], d=2, T=100, lambda_param=1.0,
                    distribution_type="gaussian_mixture", output_dir='plots'):
    all_regrets = []
    all_mean_regrets = []
    labels = []
    
    for p in p_values:
        print(f"Running experiment with p = {p} (clusters)")
        _, regret, _, mean_regret, _ = run_experiment(
            p, d, T, lambda_param, 
            distribution_type=distribution_type
        )
        all_regrets.append(regret)
        all_mean_regrets.append(mean_regret)
        labels.append(f'p = {p}')
    
    plot_regret_log_scale(all_regrets, all_mean_regrets, labels, 
                         f"Different p Values (λ={lambda_param}, {distribution_type})", output_dir)


if __name__ == "__main__":
    np.random.seed(42)
    
    output_dir = "regret_plots"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    d = 2
    T = 100  # Reduced from 1000 for faster experiments
    
    run_noise_with_fixed_lambda = False
    run_noise_with_adaptive_lambda = True

    noise_exp_p = 2  # Reduced from 3
    noise_exp_lambda = 1.0
    noise_epsilons = [0, 1.0]  # Reduced to just two values
    noise_exp_dist = "gaussian_mixture"
    
    lambda_type = "1/t"
    
    if lambda_type == "constant":
        lambda_fn = None
    elif lambda_type == "1/t":
        lambda_fn = lambda t, base: base / t if t > 0 else base
    elif lambda_type == "sqrt_t":
        lambda_fn = lambda t, base: base * np.sqrt(t)
    elif lambda_type == "log_t":
        lambda_fn = lambda t, base: base * (1 + np.log(1 + t))
    elif lambda_type == "slow_decrease":
        lambda_fn = lambda t, base: base * (1 / (1 + 0.1 * t))
    else:
        print(f"Unknown lambda type: {lambda_type}, using constant lambda")
        lambda_fn = None
    
    noise_adaptive_lambda_exp_p = 2  # Reduced from 5
    noise_adaptive_lambda_base = 1.0
    noise_adaptive_lambda_fn = lambda t, base: base / t if t > 0 else base
    noise_adaptive_lambda_epsilons = [0, 1.0]  # Reduced to just two values
    noise_adaptive_lambda_exp_dist = "gmm_timevarying"
    
    ignore_norm = input("Do you want to ignore normalization integrals? (y/n): ").lower() == 'y'
    
    if run_noise_with_fixed_lambda:
        lambda_name = lambda_type if lambda_type else "constant"
        print("\n==================================================")
        print(f"Running noise level comparison experiment with {lambda_name} lambda...")
        print("==================================================")
        run_noise_experiment(
            p=noise_exp_p,
            d=d,
            T=T,
            lambda_param=noise_exp_lambda,
            noise_epsilons=noise_epsilons,
            distribution_type=noise_exp_dist,
            lambda_adaptive_fn=lambda_fn,
            output_dir=output_dir,
            ignore_normalization=ignore_norm
        )
    
    if run_noise_with_adaptive_lambda:
        print("\n==================================================")
        print("Running noise level comparison with 1/t adaptive lambda...")
        print("==================================================")
        run_noise_experiment(
            p=noise_adaptive_lambda_exp_p,
            d=d,
            T=T,
            lambda_param=noise_adaptive_lambda_base,
            noise_epsilons=noise_adaptive_lambda_epsilons,
            distribution_type=noise_adaptive_lambda_exp_dist,
            lambda_adaptive_fn=noise_adaptive_lambda_fn,
            output_dir=output_dir,
            ignore_normalization=ignore_norm
        )
        
    print("\nAll experiments completed!")
    print(f"Plots saved to directory: {output_dir}")