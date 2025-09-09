# src/evaluation/plotting.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

def plot_evaluation_errors(results_df: pd.DataFrame, save_path: str, solver_names: list):
    """
    Plots MAE, MRE, and RMSE for a list of solvers against problem size 'n'.
    
    Args:
        results_df (pd.DataFrame): The aggregated dataframe containing error columns.
        save_path (str): The path to save the plot image.
        solver_names (list): A list of solver names to plot errors for.
    """
    logger.info("Generating evaluation error comparison plot...")
    
    # create a figure with 3 subplots, one for each error metric
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Solver Error Metrics vs. Problem Size (n)', fontsize=16, y=0.99)
    
    # define the metrics to plot with their titles
    metrics_to_plot = {
        'mae': "Mean Absolute Error (MAE)",
        'mre': "Mean Relative Error (MRE %)",
        'rmse': "Root Mean Square Error (RMSE)"
    }
    
    # traverse each solver name
    for solver_name in solver_names:
        
        # traverse each metric and its corresponding subplot
        for i, (metric, title) in enumerate(metrics_to_plot.items()):
            # dynamically create the column name based on solver name and metric
            column_name = f"{solver_name}_{metric}"
            
            # check if the column exists in the results dataframe
            if column_name in results_df.columns:
                sns.lineplot(
                    ax=axes[i], 
                    data=results_df, 
                    x='n', 
                    y=column_name, 
                    label=solver_name # use the solver name for the legend
                )
                axes[i].set_title(title)
                axes[i].set_ylabel("Error Value")
                axes[i].legend()

    axes[-1].set_xlabel("Number of Items (n)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    try:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Error comparison plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save error plot: {e}")
    finally:
        plt.close()

def plot_evaluation_times(results_df: pd.DataFrame, save_path: str):
    """Plots a comparison of solve times for all solvers."""
    logger.info("Generating evaluation time comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    sns.lineplot(data=results_df, x='n', y='avg_time_ms', hue='solver', style='solver', markers=True, dashes=False)
    
    plt.title('Solver Performance: Time vs. Problem Size (n)', fontsize=16)
    plt.xlabel('Number of Items (n)', fontsize=12)
    plt.ylabel('Average Time per Instance (ms)', fontsize=12)
    plt.yscale('log') # Use a log scale for time, as it can vary greatly
    plt.legend(title='Solver')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Time comparison plot saved to {save_path}")

def plot_evaluation_errors_DNN(results_df: pd.DataFrame, save_path: str):
    """Plots MAE, MRE, and RMSE for the ML solver against problem size."""
    logger.info("Generating evaluation error plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    # Filter for the ML solver results and required columns
    ml_results = results_df[results_df['solver'] == 'DNN'][['n', 'mae', 'mre', 'rmse']]
    if ml_results.empty:
        logger.warning("No DNN results found to plot errors.")
        return
        
    # Melt the dataframe to make it tidy for seaborn
    df_melted = ml_results.melt(id_vars='n', var_name='Error Type', value_name='Error Value')
    
    sns.lineplot(data=df_melted, x='n', y='Error Value', hue='Error Type', style='Error Type', markers=True, dashes=False)
    
    plt.title('DNN Solver Error vs. Problem Size (n)', fontsize=16)
    plt.xlabel('Number of Items (n)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(title='Error Type')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Error plot saved to {save_path}")

def plot_results(df: pd.DataFrame, save_dir: str):
    """绘制所有性能和误差图表(gym+sb3)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查x轴有多少个独特的点
    unique_n_count = df['n'].nunique()

    # --- 图1: PPO模型性能 ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plot_kind = 'line' if unique_n_count > 1 else 'strip' # 智能选择绘图类型

    if plot_kind == 'line':
        sns.lineplot(data=df, x='n', y='ppo_value', ax=axes[0], marker='o', errorbar='sd', label='PPO Agent')
        if 'baseline_value' in df.columns:
            sns.lineplot(data=df, x='n', y='baseline_value', ax=axes[0], marker='x', linestyle='--', color='gray', label='Baseline (Optimal)')
    else: # 如果只有一个n，画散点图
        sns.stripplot(data=df, x='n', y='ppo_value', ax=axes[0], jitter=True, label='PPO Agent')
        if 'baseline_value' in df.columns:
            sns.stripplot(data=df, x='n', y='baseline_value', ax=axes[0], 
              marker='x', s=10, color='darkorange', linewidth=1.2, 
              label='Baseline (Optimal)', zorder=3)

    axes[0].set_title('PPO Agent Performance vs. Problem Size')
    axes[0].set_ylabel('Total Value')
    axes[0].grid(True)
    axes[0].legend()

    if plot_kind == 'line':
        sns.lineplot(data=df, x='n', y='ppo_time', ax=axes[1], marker='o', color='r', errorbar='sd', label='PPO Agent')
        if 'baseline_time' in df.columns:
            sns.lineplot(data=df, x='n', y='baseline_time', ax=axes[1], marker='x', linestyle='--', color='gray', label='Baseline (Optimal)')
    else:
        sns.stripplot(data=df, x='n', y='ppo_time', ax=axes[1], jitter=True, color='r', label='PPO Agent')
        if 'baseline_time' in df.columns:
            sns.stripplot(data=df, x='n', y='baseline_time', ax=axes[1], 
              marker='x', s=10, color='darkorange', linewidth=1.2, 
              label='Baseline (Optimal)', zorder=3)
    
    axes[1].set_ylabel('Time (seconds)')
    axes[1].grid(True)
    axes[1].legend()

    plt.xlabel('Problem Size (n)')
    fig.suptitle('PPO Agent Performance Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, 'ppo_performance.png'))
    plt.close()
    
    # --- 图2: 优化差距 (Optimality Gap) ---
    if 'optimality_gap' in df.columns:
        plt.figure(figsize=(12, 5))
        if plot_kind == 'line':
            sns.lineplot(data=df, x='n', y='optimality_gap', marker='o', errorbar='sd')
        else:
            sns.stripplot(data=df, x='n', y='optimality_gap', jitter=True)
            
        plt.title('Optimality Gap vs. Problem Size')
        plt.ylabel('Optimality Gap (%)')
        plt.xlabel('Problem Size (n)')
        plt.grid(True)
        # 将y轴格式化为百分比
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'optimality_gap.png'))
        plt.close()
        
    print(f"All plots have been saved to '{save_dir}'")
