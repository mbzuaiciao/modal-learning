
import io
import time

import modal

# ============================================================
# 1. MODAL APPLICATION AND ENVIRONMENT
# ============================================================

app = modal.App("metis-mnist-cpu")

# Use a CPU-compatible PyTorch installation.
# Dependencies are installed remotely using uv.
image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch",
    "torchvision",
    "numpy",
    index_url="https://download.pytorch.org/whl/cpu",
)


# ============================================================
# 2. CLOUD FUNCTION — CPU INSTEAD OF GPU
# ============================================================

# No gpu= argument: request CPU compute only.
# Allocate 2 CPU cores and allow up to 10 minutes.
@app.function(image=image, cpu=2, timeout=600)
def train_model():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Match the GPU experiment's random seed.
    torch.manual_seed(42)

    # Key difference: all computation happens on the CPU.
    device = torch.device("cpu")
    torch.set_num_threads(2)

    # ========================================================
    # 3. LOAD MNIST
    # ========================================================

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        "/tmp/mnist",
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.MNIST(
        "/tmp/mnist",
        train=False,
        download=True,
        transform=transform,
    )

    # Same batch sizes as the GPU experiment.
    train_loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=512,
        shuffle=False,
    )

    # ========================================================
    # 4. SAME NEURAL NETWORK
    # ========================================================

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    start = time.perf_counter()

    # ========================================================
    # 5. TRAIN FOR FIVE EPOCHS
    # ========================================================

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            # These tensors stay in CPU memory.
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass.
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            # Backpropagation and parameter update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )

    # ========================================================
    # 6. TEST ACCURACY
    # ========================================================

    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            correct += (
                predictions.argmax(dim=1) == labels
            ).sum().item()

    elapsed = time.perf_counter() - start
    accuracy = correct / len(test_data)

    # ========================================================
    # 7. SERIALIZE MODEL WEIGHTS
    # ========================================================

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    return {
        "device": str(device),
        "accuracy": accuracy,
        "training_seconds": elapsed,
        "weights": buffer.getvalue(),
    }


# ============================================================
# 8. LOCAL ENTRY POINT — RUNS ON YOUR MAC
# ============================================================

@app.local_entrypoint()
def main():
    # Execute the cloud function on a Modal CPU container.
    result = train_model.remote()

    # Keep the CPU checkpoint separate from the GPU version.
    with open("mnist_model_cpu.pt", "wb") as f:
        f.write(result["weights"])

    print(f"Device: {result['device']}")
    print(f"Test accuracy: {result['accuracy']:.2%}")
    print(
        f"Training + evaluation: "
        f"{result['training_seconds']:.1f}s"
    )
    print("Saved model: mnist_model_cpu.pt")
