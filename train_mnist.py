
import io
import time

import modal

# ============================================================
# 1. CREATE THE MODAL APPLICATION
# ============================================================

# Create a named Modal application.
# This name appears in your Modal dashboard.
app = modal.App("metis-mnist-training")

# Define the software environment for the cloud container.
#
# Modal starts with a minimal Debian Linux image.
# We use uv to install the Python libraries needed for training:
#
# - torch: Neural network training and GPU computation.
# - torchvision: Datasets and image transformations.
# - numpy: Numerical computing dependency.
#
# These packages are installed in Modal's cloud container,
# NOT in your Mac's local uv environment.

image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch",
    "torchvision",
    "numpy",
)


# ============================================================
# 2. DEFINE THE CLOUD TRAINING FUNCTION
# ============================================================

# Register train_model() as a cloud function.
#
# gpu="T4": Request one NVIDIA T4 GPU.
# image=image: Use the software environment defined above.
# timeout=600: Allow up to 600 seconds (10 minutes) to execute.
#
# The function executes remotely when train_model.remote()
# is called from the local entry point.

@app.function(gpu="T4", image=image, timeout=600)
def train_model():

    # Import dependencies inside the remote container.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Set the random seed for more reproducible initialization.
    # This does not guarantee fully deterministic GPU results.
    torch.manual_seed(42)

    # Select the GPU for PyTorch computation.
    # "cuda" refers to NVIDIA's GPU computing platform.
    device = torch.device("cuda")

    # ========================================================
    # 3. LOAD THE MNIST DATASET
    # ========================================================

    # MNIST contains 70,000 grayscale images of handwritten
    # digits (0 through 9).
    #
    # Training set: 60,000 images.
    # Test set:     10,000 images.
    #
    # Each image is 28 x 28 pixels.

    # Convert images into PyTorch tensors.
    # Pixel values are scaled from [0, 255] to [0.0, 1.0].
    transform = transforms.ToTensor()

    # Download training data to the remote container.
    train_data = datasets.MNIST(
        root="/tmp/mnist",
        train=True,
        download=True,
        transform=transform,
    )

    # Download the separate test dataset.
    test_data = datasets.MNIST(
        root="/tmp/mnist",
        train=False,
        download=True,
        transform=transform,
    )

    # Divide training data into batches of 256 images.
    #
    # shuffle=True randomizes the training order.
    # This helps avoid learning patterns from fixed ordering.
    train_loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
    )

    # Use larger batches for testing.
    # Shuffling is unnecessary during evaluation.
    test_loader = DataLoader(
        test_data,
        batch_size=512,
        shuffle=False,
    )

    # ========================================================
    # 4. DEFINE THE NEURAL NETWORK
    # ========================================================

    # Architecture:
    #
    # Input image:       28 x 28 = 784 pixels
    #                         |
    #                         v
    # Fully connected:       128 neurons
    #                         |
    #                       ReLU
    #                         |
    #                         v
    # Output layer:           10 neurons
    #                         |
    #                         v
    # Predicted digit:        0 through 9
    #
    # This is a small multilayer perceptron (MLP).

    model = nn.Sequential(

        # Flatten each image from [1, 28, 28] to [784].
        nn.Flatten(),

        # First trainable layer:
        # Maps 784 input features to 128 hidden features.
        nn.Linear(784, 128),

        # Nonlinear activation:
        # ReLU(x) = max(0, x)
        nn.ReLU(),

        # Output layer:
        # Produces 10 scores, one for each digit.
        nn.Linear(128, 10),

    ).to(device)  # Move model parameters to GPU memory.

    # CrossEntropyLoss compares the 10 output scores
    # with the correct digit label.
    #
    # It applies the appropriate softmax-based
    # cross-entropy calculation internally.
    loss_fn = nn.CrossEntropyLoss()

    # Adam adjusts the model parameters using gradients.
    # lr=0.001 is the learning rate.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # Start measuring training and evaluation time.
    # This does not include cloud container startup.
    start = time.perf_counter()

    # ========================================================
    # 5. TRAIN THE NETWORK
    # ========================================================

    # An epoch is one complete pass through the training set.
    #
    # Five epochs means the model sees the entire training
    # dataset five times, in shuffled mini-batches.

    for epoch in range(5):

        # Enable training behavior.
        model.train()

        # Accumulate batch losses for reporting.
        total_loss = 0.0

        # Iterate over training mini-batches.
        for images, labels in train_loader:

            # Transfer images and labels from CPU to GPU.
            images = images.to(device)
            labels = labels.to(device)

            # FORWARD PASS:
            # Compute the model's predicted scores.
            predictions = model(images)

            # Calculate the error against the true labels.
            loss = loss_fn(predictions, labels)

            # Reset gradients from the previous batch.
            # PyTorch otherwise accumulates gradients.
            optimizer.zero_grad()

            # BACKPROPAGATION:
            # Compute gradients using automatic differentiation.
            loss.backward()

            # OPTIMIZATION:
            # Update weights using Adam.
            optimizer.step()

            # Save the batch loss for reporting.
            total_loss += loss.item()

        # Calculate the average loss across training batches.
        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Loss: {average_loss:.4f}"
        )

    # ========================================================
    # 6. EVALUATE ON UNSEEN TEST DATA
    # ========================================================

    # Switch to evaluation mode.
    model.eval()

    correct = 0

    # Disable gradient tracking during evaluation.
    # This reduces memory use and computation overhead.
    with torch.no_grad():

        for images, labels in test_loader:

            # Move the test batch to the GPU.
            images = images.to(device)
            labels = labels.to(device)

            # Generate prediction scores.
            predictions = model(images)

            # argmax(dim=1) selects the highest-scoring
            # digit class for each image.
            predicted_digits = predictions.argmax(dim=1)

            # Count correctly classified images.
            correct += (
                predicted_digits == labels
            ).sum().item()

    # Ensure all queued GPU operations have completed.
    torch.cuda.synchronize()

    # Measure elapsed training + evaluation time.
    elapsed = time.perf_counter() - start

    # Compute classification accuracy:
    #
    # accuracy = correct predictions / total test images
    accuracy = correct / len(test_data)

    # ========================================================
    # 7. SAVE THE TRAINED MODEL WEIGHTS
    # ========================================================

    # A BytesIO buffer stores binary data in memory.
    # This allows us to send model weights back to the Mac
    # without creating a permanent file in the container.
    buffer = io.BytesIO()

    # Move parameters to CPU and serialize the state_dict.
    #
    # state_dict() contains learned weights and biases,
    # not the complete model architecture.
    torch.save(
        model.cpu().state_dict(),
        buffer,
    )

    # Return a dictionary to the local entry point.
    #
    # Modal serializes these values and transfers them
    # from the cloud container back to your Mac.
    return {
        "gpu": torch.cuda.get_device_name(0),
        "accuracy": accuracy,
        "training_seconds": elapsed,
        "weights": buffer.getvalue(),
    }


# ============================================================
# 8. LOCAL ENTRY POINT — RUNS ON YOUR MAC
# ============================================================

# This function starts locally when you execute:
#
# uv run modal run train_mnist.py
#
# Only main() runs locally. train_model() runs on Modal.

@app.local_entrypoint()
def main():

    # Invoke the training function in the cloud.
    #
    # Modal provisions a container with an NVIDIA T4,
    # runs the training function, and returns its output.
    #
    # .remote() blocks until the remote function finishes.
    result = train_model.remote()

    # Save the trained model weights on your Mac.
    #
    # The file is created in your current working directory.
    with open("mnist_model.pt", "wb") as f:
        f.write(result["weights"])

    # Display the experiment results locally.
    print(f"GPU: {result['gpu']}")
    print(f"Test accuracy: {result['accuracy']:.2%}")
    print(
        f"Training + evaluation: "
        f"{result['training_seconds']:.1f}s"
    )
    print("Saved model: mnist_model.pt")
