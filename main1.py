import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Define the model and tokenizer
model_name = "gpt2-medium"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Input example
input_text = "Convert this text to ONNX format."

# Tokenize and convert input to tensors
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
attention_mask = tokenizer(input_text, return_tensors="pt")["attention_mask"]

# Export to ONNX format
onnx_model_path = "bert_model.onnx"

# Set the model to evaluation mode
model.eval()

# Ensure the input tensors are on the same device as the model
input_ids = input_ids.to(model.device)
attention_mask = attention_mask.to(model.device)

# Export the model to ONNX format
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"}
        },
        opset_version=11,  # ONNX opset version (adjust if needed)
        verbose=True
    )

print(f"ONNX model saved to {onnx_model_path}")
