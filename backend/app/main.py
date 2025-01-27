from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="ReviewScope API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BusinessRequest(BaseModel):
    query: str

class ReviewAnalysis(BaseModel):
    summary: str
    pros: List[str]
    cons: List[str]
    rating: Optional[float]

@app.get("/")
async def root():
    return {"message": "Welcome to ReviewScope API"}

@app.post("/api/analyze-business")
async def analyze_business(request: BusinessRequest):
    try:
        # Simulated analysis result
        analysis = ReviewAnalysis(
            summary="This is a sample business with good service but higher prices.",
            pros=["Excellent customer service", "Clean environment", "Quality products"],
            cons=["Slightly expensive", "Limited parking"],
            rating=4.2
        )
        
        return {
            "business_name": request.query,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
