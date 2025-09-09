import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('evaluation_full_summary.csv')

# Pivot: one column per solver
time_pivot = df.pivot(index='n', columns='solver', values='avg_time_ms')

# Extract n and time arrays
n_vals = time_pivot.index.values
dp_time = time_pivot['1D DP (Optimized)'].values
bnb_time = time_pivot['Branch and Bound'].values
fptas_time = time_pivot['FPTAS'].values
greedy_time = time_pivot['Greedy'].values

# Create log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(n_vals, dp_time, 'o-', label='1D DP (Optimized)', color='C0', markersize=4)
plt.loglog(n_vals, bnb_time, 's-', label='Branch & Bound', color='C1', markersize=4)
plt.loglog(n_vals, fptas_time, '^-', label='FPTAS', color='C2', markersize=4)
plt.loglog(n_vals, greedy_time, 'd-', label='Greedy', color='C3', markersize=4)

# Add reference lines (polynomial growth)
n_smooth = np.logspace(np.log10(n_vals.min()), np.log10(n_vals.max()), 100)
plt.loglog(n_smooth, 1e-3 * n_smooth**1.0, '--', color='gray', alpha=0.7, label=r'$O(n)$')
plt.loglog(n_smooth, 1e-5 * n_smooth**2.0, '--', color='gray', alpha=0.7, label=r'$O(n^2)$')
plt.loglog(n_smooth, 1e-7 * n_smooth**3.0, '--', color='gray', alpha=0.7, label=r'$O(n^3)$')

# Styling: high data-ink ratio
plt.xlabel('Problem Size $n$', fontsize=12)
plt.ylabel('Running Time (ms)', fontsize=12)
plt.title('Time Complexity Growth: Log-Log Plot', fontsize=13, pad=20)
plt.legend(frameon=False, fontsize=10, loc='upper left')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.box(False)  # Remove border
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)

# Adjust layout and save
plt.tight_layout()
plt.savefig('knapsack_loglog.pdf', dpi=300, bbox_inches='tight')
plt.savefig('knapsack_loglog.png', dpi=300, bbox_inches='tight')
plt.show()