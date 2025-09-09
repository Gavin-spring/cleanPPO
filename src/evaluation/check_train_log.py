import numpy as np

# 1. Set the path to your evaluations.npz file
log_file_path = "artifacts_sb3/training/Exp23-proExp13-20250807_202659/logs/evaluations.npz"

print(f"--- Loading data from: {log_file_path} ---\n")

try:
    # 2. Load the .npz file
    data = np.load(log_file_path)

    # 3. List all arrays stored in the file
    print(f"Arrays (keys) found in the file: {data.files}\n")

    # 4. Access and print the data for each array
    
    # Timesteps when evaluations were run
    timesteps = data['timesteps']
    print(f"Evaluation was performed at the following timesteps:")
    print(f"{timesteps}\n")
    print("-" * 50)

    # Results (rewards per episode for each evaluation)
    results = data['results']
    print("Evaluation Results (rewards per episode):")
    print(f"Shape of results array: {results.shape}  (evaluations, episodes_per_eval)\n")
    
    # Process and print summary statistics for each evaluation point
    for i, t in enumerate(timesteps):
        # Get the rewards for the i-th evaluation
        current_rewards = results[i]
        mean_reward = np.mean(current_rewards)
        std_reward = np.std(current_rewards)
        
        print(f"Evaluation at timestep {t}:")
        print(f"  - Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  - Min/Max Reward: {np.min(current_rewards):.2f} / {np.max(current_rewards):.2f}")
        # print(f"  - All rewards: {current_rewards}") # Uncomment to see raw rewards
    print("\n" + "-" * 50)

    # Episode lengths
    ep_lengths = data['ep_lengths']
    print("Episode Lengths:")
    print(f"Shape of ep_lengths array: {ep_lengths.shape}\n")
    
    for i, t in enumerate(timesteps):
        current_lengths = ep_lengths[i]
        mean_length = np.mean(current_lengths)
        
        print(f"Evaluation at timestep {t}:")
        print(f"  - Mean Episode Length: {mean_length:.2f}")
        # print(f"  - All lengths: {current_lengths}") # Uncomment to see raw lengths
    print("\n" + "-" * 50)

    # Check for the optional 'successes' array
    if 'successes' in data.files:
        successes = data['successes']
        print("Success Rate:")
        print(f"Shape of successes array: {successes.shape}\n")
        
        for i, t in enumerate(timesteps):
            current_successes = successes[i]
            # Success rate is the mean of the boolean/binary array
            success_rate = np.mean(current_successes)
            
            print(f"Evaluation at timestep {t}:")
            print(f"  - Success Rate: {success_rate:.2%}") # Format as percentage
        print("\n" + "-" * 50)
    else:
        print("No 'successes' array found in this file.\n")
        
    # 5. Remember to close the file handler if you are done
    data.close()

except FileNotFoundError:
    print(f"ERROR: File not found at '{log_file_path}'. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")