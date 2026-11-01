
import modal

# 1. Create a Modal application
app = modal.App("hello-metis")

# 2. Define a function that runs in the cloud
@app.function()
def calculate(x: int) -> int:
    return x * x

# 3. Define the entry point that runs locally
@app.local_entrypoint()
def main():
    # 4. Execute calculate() on Modal's cloud
    result = calculate.remote(12)

    # 5. Print the returned result locally
    print("Result:", result)
