import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Plot MRE vs Problem Size: Target Solver vs Gurobi.")
parser.add_argument('--csv-path', type=str, required=True, help="Path to the evaluation summary CSV file.")
parser.add_argument('--solver', type=str, required=True, help="Name of the solver to plot (e.g., 'PPO').")
args = parser.parse_args()

csv_path = args.csv_path
target_solver = args.solver
# --- End Argument Parsing ---

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}")

output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
# Let's create a more descriptive filename
output_filename = f'mre_vs_problem_size_{target_solver.lower().replace(" ", "_")}.png'
output_path = os.path.join(output_dir, output_filename)

# Load data
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    raise RuntimeError(f"Failed to read CSV: {e}")

# --- Data Processing Fix ---
# The column name is specific to PPO in the CSV, let's handle that.
mre_col_name = 'PPO_mre' 
if mre_col_name not in df.columns:
    raise ValueError(f"Missing required column for MRE calculation: {mre_col_name}")

target_data = df[df['solver'] == target_solver].copy()
if target_data.empty:
    raise ValueError(f"No data found for solver: {target_solver}")

# 1. Calculate the correct MRE in percent. The column is the approximation ratio.
#    MRE(%) = 100 - ApproximationRatio(%)
target_data['mre_percent'] = 100 - target_data[mre_col_name]

# 2. Convert the MRE percent to a decimal for PercentFormatter.
#    e.g., 0.7% becomes 0.007
target_data['mre_decimal'] = target_data['mre_percent'] / 100.0

# --- Plotting ---
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot the decimal MRE, but the axis will show it as a percentage.
sns.lineplot(data=target_data, x='n', y='mre_decimal', marker='o', label=f'{target_solver} (Instance Solver)')

# Set the formatter for the y-axis
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

# Set titles and labels
ax.set_title(f'MRE vs. Problem Size for {target_solver}', fontsize=16, pad=20)
ax.set_xlabel('Number of Items (n)', fontsize=12)
ax.set_ylabel('Mean Relative Error (MRE)', fontsize=12)
ax.legend(title='Solver')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Optional: Set a sensible y-axis limit, e.g., up to 2%
# This helps focus on the details when the error is low.
max_mre_to_show = target_data['mre_decimal'].max() * 1.2
ax.set_ylim(0, max_mre_to_show)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Plot saved: '{output_path}'")