
import modal

# Create a Modal application named "metis-gpu-test".
# This name will appear in your Modal dashboard.
app = modal.App("metis-gpu-test")

# Define the container environment used in Modal's cloud.
# Start with a minimal Debian Linux image and install PyTorch.
image = modal.Image.debian_slim().uv_pip_install(
    "torch",
    "numpy",
)

# Register this function to run in Modal's cloud.
# Request an NVIDIA T4 GPU and use the image defined above.
@app.function(gpu="T4", image=image)
def test_gpu():

    # Import PyTorch inside the remote container.
    import torch

    # Create a 1000 x 1000 matrix of random numbers.
    # device="cuda" places the matrix in GPU memory.
    x = torch.randn(1000, 1000, device="cuda")

    # Perform matrix multiplication on the GPU.
    # @ is Python's matrix multiplication operator.
    y = x @ x

    # Return information about the GPU and computation.
    # Modal serializes the result and sends it to your laptop.
    return {
        "gpu": torch.cuda.get_device_name(0),
        "result_shape": list(y.shape),
    }


# Define the entry point that runs on your laptop.
# This function starts when you execute:
# uv run modal run gpu_test.py
@app.local_entrypoint()
def main():

    # Invoke test_gpu() remotely in Modal's cloud.
    # Modal allocates a container with a T4 GPU,
    # executes the function, and returns the result.
    result = test_gpu.remote()

    # Print the returned result on your laptop.
    print(result)
