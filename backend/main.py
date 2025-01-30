from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from services.google_service import GooglePlacesService
from services.analysis_service import ReviewAnalyzer

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
google_service = GooglePlacesService()
review_analyzer = ReviewAnalyzer()

@app.get("/api/analyze/{business_name}")
async def analyze_business(business_name: str):
    try:
        # Get business details and reviews
        business_data = await google_service.search_business(business_name)
        
        if not business_data:
            raise HTTPException(status_code=404, detail="Business not found")

        # Analyze reviews
        analysis = review_analyzer.analyze_reviews(business_data['reviews'])

        return {
            "business": {
                "name": business_data['name'],
                "rating": business_data['rating']
            },
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 