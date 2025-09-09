import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Setup & Data Loading ---
# Using the EXACT SAME style settings as the MRE plot
sns.set_theme(style="whitegrid", font_scale=1.2)

# Load the dataset from your CSV file
try:
    df = pd.read_csv('evaluation_full_summary.csv')
except FileNotFoundError:
    print("Error: 'evaluation_full_summary.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit()

# --- NEW: Filter and rename solvers for a clean plot ---
# Select only the solvers you want to display
solvers_to_plot = ['PPO', 'PointerNet RL', 'DNN', 'Gurobi']
df_plot = df[df['solver'].isin(solvers_to_plot)].copy()

# Rename solvers for a more descriptive legend
df_plot['solver'] = df_plot['solver'].replace({
    'PPO': 'PPO (Our Work)',
    'PointerNet RL': 'PointerNet (RL)',
    'DNN': 'MLP / DNN',
    'Gurobi': 'Gurobi',
})


# --- 2. Plotting (Now much simpler!) ---
# Using the EXACT SAME figure size
plt.figure(figsize=(10, 6))
ax = plt.gca()

# A SINGLE lineplot command using 'hue' to automatically create different lines for each solver
sns.lineplot(
    data=df_plot, 
    x='n', 
    y='avg_time_ms', 
    hue='solver',      # Use the 'solver' column to differentiate lines by color
    style='solver',      # Use different markers for each solver
    markers=True,        # Show markers on data points
    dashes=False,        # Use solid lines for all
    ax=ax
)

# --- 3. Customization ---
# The y-axis is also logarithmic for the time plot
ax.set_yscale('log')

# Set titles and labels, using the same font sizes as before
ax.set_title('Solver Performance: Time vs. Problem Size', fontsize=16, pad=20)
ax.set_xlabel('Number of Items (n)', fontsize=12)
ax.set_ylabel('Average Time per Instance (ms) - Log Scale', fontsize=12)

# Use the same legend and grid style
ax.legend(title='Model')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- 4. Save the Figure ---
# Save the plot to a new, high-resolution PNG file
output_filename = 'evaluation_times_vs_n_styled.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Plot successfully saved as '{output_filename}'")