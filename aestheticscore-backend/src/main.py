from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from ml.model import BeautyScoreModel  # Import your model
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# Your existing CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeautyScoreModel().to(device)
model.load_state_dict(torch.load('src/ml/best_model.pth'))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Your existing routes
from routes.business import router as business_router
app.include_router(business_router)

# Add new predict endpoint
@app.post("/predict")
async def predict_beauty_score(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        score = model(image_tensor)
        
    return {"beauty_score": float(score[0][0])} 