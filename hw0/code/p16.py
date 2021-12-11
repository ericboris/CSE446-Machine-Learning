 import numpy as np
import matplotlib.pyplot as plt

n = 40000
Z = np.random.randn(n)

# Draw the graph.
plt.step(sorted(Z), np.arange(1, n+1) / float(n), label='Gaussian')

axes = plt.gca()
axes.set_xlim((-3, 3))

plt.title('Plot of F^hat_n(x) from -3 to 3 with n=40000 samples')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Probabilities')
plt.savefig('q16a.pdf')

  