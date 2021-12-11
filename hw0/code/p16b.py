import numpy as np
import matplotlib.pyplot as plt

n = 40000
Z = np.random.randn(n)

# Draw the same graph from a.
plt.step(sorted(Z), np.arange(1, n+1) / float(n), label='Gaussian')

# Draw the step graphs for b.
for k in [1, 8, 64, 512]:
	Z_k = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1.0 / k), axis=1)
	plt.step(sorted(Z_k), np.arange(1, n+1) / float(n), label=f'k={k}')

axes = plt.gca()
axes.set_xlim((-3, 3))

plt.title('Plot of F^hat_n(x) from -3 to 3 with n=40000 samples and k steps')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Probabilities')
plt.savefig('q16b.pdf')
