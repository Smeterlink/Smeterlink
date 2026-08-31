import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate toy data
theta_true = 2.0
sigma = 1.0
N = 200

key = jax.random.PRNGKey(0)
x = theta_true + sigma * jax.random.normal(key, (N,))

# 2. Define model
def model(x):
    theta = numpyro.sample("theta", dist.Normal(0, 5))
    numpyro.sample("obs", dist.Normal(theta, 1), obs=x)

# 3. Run HMC / NUTS
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
mcmc.run(key, x=x)

samples = mcmc.get_samples()
theta_samples = samples["theta"]

# 4. Plot posterior histogram
plt.hist(theta_samples, bins=40, density=True, alpha=0.6)
plt.axvline(theta_true, color="red", label="true theta")
plt.xlabel("theta")
plt.ylabel("density")
plt.title("Posterior distribution of θ")
plt.legend()
plt.show()
