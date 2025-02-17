from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from src.machine_learning.ml_model import BeautyScoreModel  # Import your model

app = FastAPI()

# Remove Fly.io backend URL if you don't need it anymore
# Instead, you can leave the frontend domain or any origins that will access your API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aestheticscore.netlify.app"],  # Only allow your frontend if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeautyScoreModel().to(device)
model.load_state_dict(torch.load('src/machine_learning/scut_fbp_model.pt', map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running with HTTPS!"}

@app.post("/predict")
async def predict_beauty_score(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        score = model(image_tensor)
        
    return {"beauty_score": float(score[0][0])}
