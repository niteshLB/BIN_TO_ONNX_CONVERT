import torch
import torchvision.models as models
import torch.onnx as onnx

# Load a pre-trained AlexNet model from torchvision
model = models.alexnet(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define example input (adjust to match your model's input shape)
example_input = torch.randn(1, 3, 224, 224)

# Convert the PyTorch model to ONNX format
onnx_filename = "alexnet_model.onnx"
onnx.export(model, example_input, onnx_filename, verbose=True)

print(f"AlexNet model successfully converted to ONNX and saved as {onnx_filename}")
