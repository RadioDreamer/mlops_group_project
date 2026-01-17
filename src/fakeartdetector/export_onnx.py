import onnx
import torch
import numpy as np
import sys
import time
from statistics import mean, stdev

import onnxruntime as ort
import torch
import torchvision


from fakeartdetector.model import FakeArtClassifier
import typer

app = typer.Typer()
model = FakeArtClassifier()
model.eval()
ort_session = ort.InferenceSession("fakeartclassifier.onnx")
dummy_input = torch.randn(1, 3, 32, 32)



@app.command()
def doruntime():
    import onnxruntime as rt
    ort_session = rt.InferenceSession("fakeartclassifier.onnx")
    input_names = [i.name for i in ort_session.get_inputs()]
    output_names = [i.name for i in ort_session.get_outputs()]
    batch = {input_names[0]: np.random.randn(1, 3, 32, 32).astype(np.float32)}
    out = ort_session.run(output_names, batch)

@app.command()
def exporttoonnx():
    # Initialize model

    # Create dummy input for tracing

    # Export to ONNX using traditional export (compatible with BatchNorm)
    torch.onnx.export(
        model,
        dummy_input,
        "fakeartclassifier.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX model exported successfully to fakeartclassifier.onnx")

@app.command()
def validateonnx():
    import onnx
    model = onnx.load("fakeartclassifier.onnx")
    onnx.checker.check_model(model)
    print(onnx.printer.to_text(model.graph))
    # if the above does not make any sense, we can also use the following package top visualize the model
    # uv run netron fakeartclassifier.onnx

def timing_decorator(func, function_repeat: int = 10, timing_repeat: int = 5):
    """Decorator that times the execution of a function."""

    def wrapper(*args, **kwargs):
        timing_results = []
        for _ in range(timing_repeat):
            start_time = time.time()
            for _ in range(function_repeat):
                result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append(elapsed_time)
        print(f"Avg +- Stddev: {mean(timing_results):0.3f} +- {stdev(timing_results):0.3f} seconds")
        return result

    return wrapper

@timing_decorator
def torch_predict(image) -> None:
    """Predict using PyTorch model."""
    model(image)


@timing_decorator
def onnx_predict(image) -> None:
    """Predict using ONNX model."""
    ort_session.run(None, {"input": image.numpy()})

@app.command()
def comparetiming():
    # for size in [32, 64, 128, 256]:
    size = 32
    print(f"Input size: {size}x{size}")
    dummy_input = torch.randn(1, 3, size, size)
    print("PyTorch inference time:")
    torch_predict(dummy_input)
    print("ONNX inference time:")
    onnx_predict(dummy_input)
    print("-" * 30)

@app.command()
def compgraph():
    import onnxruntime as rt
    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = "optimized_model.onnx"

    session = rt.InferenceSession("fakeartclassifier.onnx", sess_options)

@app.command()
def check_onnx_model(
    onnx_model_file: str,
    pytorch_model: torch.nn.Module,
    random_input: torch.Tensor,
    rtol: float = 1e-03,
    atol: float = 1e-05,
) -> None:
    import onnxruntime as rt
    import numpy as np

    ort_session = rt.InferenceSession(onnx_model_file)
    ort_inputs = {ort_session.get_inputs()[0].name: random_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pytorch_outs = pytorch_model(random_input).detach().numpy()

    assert np.allclose(ort_outs[0], pytorch_outs, rtol=rtol, atol=atol)


if __name__ == "__main__":
    app()