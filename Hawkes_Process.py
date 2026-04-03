# hawkes_process.py
# Self-exciting point process for modeling event arrival intensity.

import numpy as np
from scipy.optimize import minimize


class HawkesProcess:
    """
    Hawkes process with exponential kernel:

        λ(t) = μ + Σ α·exp(-β·(t - tᵢ)),  for tᵢ < t

    Parameters:
        μ     : baseline intensity
        α     : excitation coefficient
        β     : decay rate

    The ratio α/β (branching ratio) indicates the degree of self-excitation.
    """

    def __init__(self):
        self.mu = 1.0
        self.alpha = 0.5
        self.beta = 1.5

        self.event_times = []
        self.current_intensity = self.mu

    def add_event(self, timestamp: float):
        """Add a new event and update intensity."""
        self.event_times.append(timestamp)

        # Keep only recent events for computational efficiency
        if len(self.event_times) > 500:
            self.event_times = self.event_times[-500:]

        self.current_intensity = self._compute_intensity(timestamp)

    def _compute_intensity(self, t: float) -> float:
        """Compute intensity λ(t) at time t."""
        intensity = self.mu

        for ti in self.event_times:
            if ti < t:
                intensity += self.alpha * np.exp(-self.beta * (t - ti))

        return intensity

    def fit(self, event_times: np.ndarray):
        """
        Fit parameters using maximum likelihood estimation.

        Uses Ogata (1981) recursive formulation to reduce complexity
        from O(n²) to O(n) per likelihood evaluation.
        """
        evt = np.asarray(event_times, dtype=float)

        if len(evt) < 10:
            return

        if len(evt) > 500:
            evt = evt[-500:]

        T = evt[-1] - evt[0]
        dt = np.diff(evt)

        def neg_ll(params):
            mu, alpha, beta = params

            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return 1e10

            R = np.zeros(len(evt))
            for i in range(1, len(evt)):
                R[i] = np.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])

            lam = mu + alpha * R
            lam = np.maximum(lam, 1e-12)

            log_sum = np.sum(np.log(lam))

            compensator = mu * T + (alpha / beta) * np.sum(
                1.0 - np.exp(-beta * (T - (evt - evt[0])))
            )

            return -(log_sum - compensator)

        result = minimize(
            neg_ll,
            [self.mu, self.alpha, self.beta],
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, 0.99), (1e-6, None)],
            options={"maxiter": 200, "ftol": 1e-8},
        )

        if result.success and result.x[1] < result.x[2]:
            self.mu, self.alpha, self.beta = result.x

    @property
    def regime(self) -> str:
        """
        Classify activity level based on intensity relative to baseline.
        """
        if self.current_intensity < self.mu * 1.5:
            return "LOW_ACTIVITY"
        elif self.current_intensity < self.mu * 3:
            return "MODERATE"
        elif self.current_intensity < self.mu * 8:
            return "HIGH_ACTIVITY"
        else:
            return "EXTREME"

    @property
    def branching_ratio(self) -> float:
        """
        Ratio α/β indicating degree of self-excitation.
        """
        return self.alpha / self.beta
