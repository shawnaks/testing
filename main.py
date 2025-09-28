from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
import torchvision.models as models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # <- allows requests from any domain
    allow_credentials=True,
    allow_methods=["*"],      # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],      # allow all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


# Define the model architecture (must match training)
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(TransferLearningModel, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# Load model checkpoint (adjust path as needed)
checkpoint = torch.load(
    "model.pth", map_location="cpu"
)  # Using the model.pth file in the render directory
num_classes = checkpoint["num_classes"]
class_names = checkpoint["class_names"]
model = TransferLearningModel(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def transform_image(image_bytes):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return {"predicted_class": predicted_class}