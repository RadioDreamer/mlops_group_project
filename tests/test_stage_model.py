import os
import time

import torch
import wandb
from dotenv import load_dotenv

from fakeartdetector.model import FakeArtClassifier

load_dotenv()


def load_model(artifact_path):
    print(f"--- Loading Model from: {artifact_path} ---")
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

    # Access the specific artifact from the registry
    artifact = api.artifact(artifact_path)

    download_dir = "staged_model_dir"
    artifact.download(root=download_dir)

    # Get the checkpoint filename and load the model
    file_name = artifact.files()[0].name
    full_ckpt_path = os.path.join(download_dir, file_name)
    return FakeArtClassifier.load_from_checkpoint(full_ckpt_path)


def test_model_speed():
    model_path = os.getenv("MODEL_NAME")
    if not model_path:
        print("ERROR: MODEL_NAME environment variable is not set!")
        return

    model = load_model(model_path)
    model.eval()

    print("--- Starting Speed Test (100 iterations) ---")
    start = time.time()

    with torch.no_grad():
        for _ in range(100):
            # Fixed input
            model(torch.rand(1, 3, 32, 32))

    end = time.time()
    total_time = end - start

    print(f"Total time for 100 runs: {total_time:.4f}s")
    print(f"Average latency: {(total_time / 100) * 1000:.2f}ms per image")

    assert total_time < 1, f"Model too slow! {total_time:.2f}s"
    print("SUCCESS: Model passed speed requirements.")


if __name__ == "__main__":
    try:
        test_model_speed()
    except Exception as e:
        print(f"SCRIPT CRASHED: {e}")
