# recreate_plot_from_log.py
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- User-configurable parameters ---

# 1. Please update this with the actual path to your log file
#    Based on your previous logs, the path should be similar to this:
LOG_FILE_PATH = r"D:\mycode\UOBCODE\semester2\OR\Mine\artifacts\runs\training\20250622_214750_n1000_lr0.001\training_log_20250622_214750.log"

# 2. This is the name and location for the generated image file
OUTPUT_IMAGE_PATH = r"D:\mycode\UOBCODE\semester2\OR\Mine\artifacts\runs\training\20250622_214750_n1000_lr0.001\recreated_loss_curve.png"

# --- Core logic ---

def parse_log_file(log_path: str) -> pd.DataFrame:
    """
    Parses the specified training log file and extracts epoch, training loss, and validation loss.
    """
    # Regular expression to match lines containing loss information
    # Example: Epoch 1/800, Train Loss: 0.003800, Val Loss: 0.003627
    log_pattern = re.compile(
        r"Epoch (\d+)/\d+, Train Loss: ([\d.]+), Val Loss: ([\d.]+)"
    )
    
    data = []
    print(f"Reading log file: {log_path}")
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))
                    data.append({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    })
    except FileNotFoundError:
        print(f"Error: Log file not found! Please check the path: {log_path}")
        return pd.DataFrame()  # Return an empty DataFrame

    print(f"Successfully parsed {len(data)} epoch records.")
    return pd.DataFrame(data)


def plot_loss_curve(history_df: pd.DataFrame, save_path: str):
    """
    Plots and saves the loss curve based on parsed data.
    This function style references the original _plot_loss_curve method.
    """
    if history_df.empty:
        print("No data available to plot.")
        return

    print("Plotting the loss curve...")
    # Use the same style as in your original script
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    
    # Plot using seaborn
    sns.lineplot(data=history_df, x='epoch', y='train_loss', label='Training Loss')
    sns.lineplot(data=history_df, x='epoch', y='val_loss', label='Validation Loss')
    
    # Find and mark the point with minimum validation loss
    best_epoch_data = history_df.loc[history_df['val_loss'].idxmin()]
    best_epoch = int(best_epoch_data['epoch'])
    min_val_loss = best_epoch_data['val_loss']
    
    plt.axvline(x=best_epoch, color='r', linestyle='--', linewidth=1, label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, min_val_loss, color='red', s=100, zorder=5, marker='*')
    plt.text(best_epoch + 5, min_val_loss, f'Min Val Loss: {min_val_loss:.6f}', color='red')
    
    # Set title and labels
    plt.title('Training & Validation Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    
    # Use logarithmic scale for y-axis, consistent with your original settings
    plt.yscale('log')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"Success! Loss curve saved at: {os.path.abspath(save_path)}")


if __name__ == '__main__':
    # 1. Parse the log file
    df = parse_log_file(LOG_FILE_PATH)
    
    # 2. If parsing was successful, plot the results
    if not df.empty:
        plot_loss_curve(df, OUTPUT_IMAGE_PATH)