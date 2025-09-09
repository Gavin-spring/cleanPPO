import torch

def verify():
    print("--- Verification Started ---")

    # 1. check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    # 2. check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_cuda_available}")

    if not is_cuda_available:
        print("Error: CUDA not available. Please check your NVIDIA driver and PyTorch installation.")
        return

    # 3. check CUDA version and GPU name
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 4. check if torch.compile is available and working
    print("\n--- Testing torch.compile (Triton) ---")
    try:
        # Define a simple model for testing
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        # move model to GPU
        model = SimpleModel().to("cuda")

        # create a random input tensor
        input_tensor = torch.randn(4, 10).to("cuda")

        # compile the model using torch.compile
        compiled_model = torch.compile(model)

        # run the compiled model with the input tensor
        _ = compiled_model(input_tensor)

        print("✅ SUCCESS: torch.compile works correctly! Triton is active.")

    except Exception as e:
        print(f"❌ ERROR: torch.compile failed. Details: {e}")

    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    verify()