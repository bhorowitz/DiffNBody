## NFW-related class functions

import jax.numpy as jnp
import jax.random as random
from utils_ott import lambertw #from ott, but seperated out to make our lives easier in the short term but harder in the long term

class NFW:
    """NFW radial mass distribution.

    This distribution is useful to sample satellite galaxies according to an NFW
    radial profile.

    Implementation follows: https://arxiv.org/abs/1805.09550
    """

    def __init__(self, concentration, Rvir):
        """
        Constructs an NFW profile with specified concentration and virial radius.

        Args:
            concentration: Scalar or array-like, concentration of the NFW profile.
            Rvir: Scalar or array-like, the virial radius of the profile.
        """
        self.concentration = jnp.asarray(concentration, dtype=jnp.float32)
        self.Rvir = jnp.asarray(Rvir, dtype=jnp.float32)

    def _q(self, r):
        """Standardizes input radius `r` to a normalized `q`."""
        return r / self.Rvir

    def pdf(self, r):
        """Probability density function (PDF) of the NFW profile."""
        q = self._q(r)
        c = self.concentration
        
        p = (q * self.concentration**2) / (
            ((q * self.concentration) + 1.0)**2 *
            (1.0 / (self.concentration + 1.0) + jnp.log(self.concentration + 1.0) - 1.0))
        return p

    def cdf_unormalized(self, q):
        """Unnormalized cumulative distribution function (CDF)."""
        x = q * self.concentration
        return jnp.log(1.0 + x) - x / (1.0 + x)

    def cdf(self, r):
        """Cumulative distribution function (CDF) of the NFW profile."""
        q = self._q(r)
        return self.cdf_unormalized(q) / self.cdf_unormalized(1.0)

    def log_cdf(self, r):
        """Log of the cumulative distribution function."""
        q = self._q(r)
        return jnp.log(self.cdf_unormalized(q)) - jnp.log(self.cdf_unormalized(1.0))

    def quantile(self, p):
        """Inverse CDF (quantile function) of the NFW profile."""
        p_scaled = p * self.cdf_unormalized(1.0)  # Scale p
        q = -1.0 / jnp.real(lambertw(-jnp.exp(-p_scaled - 1))) - 1
        return q / self.concentration

    def circular_velocity(self,r):
        return jnp.sqrt(
            self.cdf(r)
        )
    
    def sample(self, key, shape=()):
        """Samples from the NFW profile.

        Args:
            key: PRNG key for JAX random number generation.
            shape: Shape of the output samples.

        Returns:
            Samples from the NFW distribution.
        """
        shape = shape + jnp.broadcast_shapes(self.concentration.shape, self.Rvir.shape)
        uniform_samples = random.uniform(key, shape=shape, minval=jnp.finfo(jnp.float32).tiny, maxval=1.)
        return self.quantile(uniform_samples) * self.Rvir