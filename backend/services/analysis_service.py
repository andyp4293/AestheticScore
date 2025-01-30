from transformers import pipeline
from collections import defaultdict

class ReviewAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.summarizer = pipeline("summarization")

    def analyze_reviews(self, reviews):
        # Analyze sentiment for each review
        analyzed_reviews = []
        pros = defaultdict(int)
        cons = defaultdict(int)

        for review in reviews:
            text = review['text']
            sentiment = self.sentiment_analyzer(text)[0]
            
            # Extract key points based on sentiment
            sentences = text.split('.')
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                    
                if sentiment['label'] == 'POSITIVE':
                    pros[sentence.strip()] += 1
                else:
                    cons[sentence.strip()] += 1

            analyzed_reviews.append({
                'text': text,
                'sentiment': sentiment['label'],
                'score': sentiment['score']
            })

        # Get top pros and cons
        top_pros = sorted(pros.items(), key=lambda x: x[1], reverse=True)[:5]
        top_cons = sorted(cons.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'summary': {
                'pros': [pro[0] for pro in top_pros],
                'cons': [con[0] for con in top_cons]
            },
            'reviews': analyzed_reviews
        } 