#!/bin/bash
#
# ==============================================================================
#  Automated Environment Setup Script
# ==============================================================================

# --- Configuration ---
set -e

# --- 1. PASTE YOUR RCLONE CONFIGURATION HERE ---
# Get this by running 'cat ~/.config/rclone/rclone.conf' on your LOCAL machine.
# Paste the entire multi-line content between the `EOF` markers.
mkdir -p "$HOME/.config/rclone"
cat << EOF > "$HOME/.config/rclone/rclone.conf"
[gdrive]
type = drive
scope = drive
token = {"access_token":"<YOUR_ACCESS_TOKEN>",
"token_type":"Bearer",
"refresh_token":"<YOUR_REFRESH_TOKEN>",
"expiry":"<YOUR_EXPIRY_TIME>"}
team_drive =
EOF

# --- 2. PASTE YOUR NGROK AUTHTOKEN HERE ---
# IMPORTANT: Paste the token directly, WITHOUT any quotes or brackets.
NGROK_AUTH_TOKEN=<YOUR_NGROK_AUTH_TOKEN>

WORK_DIR="/jupyter/work"
PROJECT_DIR_NAME="ann_kp"
PROJECT_DIR="$WORK_DIR/$PROJECT_DIR_NAME"
ENV_NAME="myenv"
PYTHON_VERSION="3.11"
REQUIREMENTS_FILE="requirements.txt"

# --- Script Start ---
echo "Starting automated environment setup..."

# --- Step 1/5: Configure & Install Tools ---
echo "Step 1/5: Configuring and installing tools..."

# Verify rclone config was pasted
if grep -q "<YOUR_" "$HOME/.config/rclone/rclone.conf"; then
    echo "ERROR: Rclone config placeholder found. Please edit the script and paste your config."
    exit 1
else
    echo "Rclone config file created successfully."
fi

# Install rclone
sudo yum install rclone -y

# Download and configure ngrok
cd "$HOME"
if [ ! -f "ngrok" ]; then
    wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip -O ngrok.zip
    unzip ngrok.zip
    rm ngrok.zip
fi
if [[ "$NGROK_AUTH_TOKEN" != "PASTE_YOUR_NGROK_AUTHTOKEN_HERE" ]]; then
    ./ngrok config add-authtoken "$NGROK_AUTH_TOKEN"
else
    echo "WARNING: ngrok authtoken not set. Please edit the script."
fi

# --- Step 2/5: Installing Miniconda ---
echo "Step 2/5: Installing Miniconda..."
if [ -d "$HOME/miniconda3" ]; then
    echo "Miniconda directory already exists. Skipping installation."
else
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" config --set auto_activate_base false
fi

# --- Step 3/5: Initializing and Fixing Conda ---
echo "Step 3/5: Initializing and fixing Conda configuration..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$HOME/.condarc" ]; then
    sed -i '/auto_activate_base/d' "$HOME/.condarc" 2>/dev/null || true
fi
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Step 4/5: Creating Conda Environment & Installing Dependencies ---
echo "Step 4/5: Creating Conda environment and installing dependencies..."
if ! conda env list | grep -q "$ENV_NAME"; then
    conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y
else
    echo "Conda environment '$ENV_NAME' already exists."
fi
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r "$REQUIREMENTS_FILE"
pip install gymnasium "stable-baselines3[extra]"
echo "Python dependencies have been installed."

# --- Step 5/5: Finished! ---
echo ""
echo "✅ Setup Complete! Your environment is ready."
echo "=============================================================================="
echo "‼️ IMPORTANT: HOW TO USE YOUR ENVIRONMENT ‼️"
echo ""
echo "Your interactive terminal is using a different Conda. To use the environment we just created,"
echo "you MUST run this command first in any new terminal:"
echo ""
echo "    source ~/miniconda3/etc/profile.d/conda.sh"
echo ""
echo "After that, you can activate your environment:"
echo ""
echo "    conda activate $ENV_NAME"
echo ""
echo "------------------------------------------------------------------------------"
echo "Example Workflow:"
echo ""
echo "1. In Terminal 1 (for TensorBoard):"
echo "   source ~/miniconda3/etc/profile.d/conda.sh"
echo "   conda activate $ENV_NAME"
echo "   cd $PROJECT_DIR"
echo "   "
echo ""
echo "2. In Terminal 2 (for ngrok):"
echo "   cd ~"
echo "   ./ngrok http 6006"
echo ""
echo "3. After training, to upload results:"
echo "   source ~/miniconda3/etc/profile.d/conda.sh"
echo "   cd $PROJECT_DIR"
echo "   rclone copy artifacts_sb3 gdrive:backup/ann_kp/artifacts_sb3 -P"
echo "=============================================================================="