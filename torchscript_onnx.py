import torch, timm, pathlib
from models.vit_model import EmotionViTClassifier

model = EmotionViTClassifier(
            model_name="vit_tiny_patch16_224",
            num_classes=7).eval()
model.load_state_dict(torch.load("output/final_model.pth", map_location="cpu"))

example = torch.randn(64, 3, 224, 224)
torch.jit.trace(model, example).save("emotion_vit_tiny.torchscript.pt")
# optional ONNX
torch.onnx.export(model, example, "emotion_vit_tiny.onnx",
                  input_names=["pixel_values"],
                  output_names=["logits"],
                  dynamic_axes={"pixel_values": {0: "batch"},
                                "logits": {0: "batch"}},
                  opset_version=14)
