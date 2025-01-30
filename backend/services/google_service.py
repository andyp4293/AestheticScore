from googleapiclient.discovery import build
import os

class GooglePlacesService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_PLACES_API_KEY")
        self.service = build('places', 'v1', developerKey=self.api_key)

    async def search_business(self, query: str):
        try:
            # Search for business
            places_result = self.service.places().findPlaceFromText(
                input_=query,
                inputtype='textquery'
            ).execute()

            if places_result.get('candidates'):
                place_id = places_result['candidates'][0]['place_id']
                
                # Get place details including reviews
                details = self.service.places().details(
                    placeId=place_id,
                    fields=['name', 'rating', 'reviews']
                ).execute()

                return {
                    'name': details.get('name'),
                    'rating': details.get('rating'),
                    'reviews': details.get('reviews', [])
                }
            return None
        except Exception as e:
            print(f"Error searching business: {e}")
            return None 