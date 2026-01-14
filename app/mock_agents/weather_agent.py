import json
from datetime import datetime

class WeatherAgent:
    """
    Mock agent that returns weather information in a structured JSON format.
    """
    def process(self, query: str) -> dict:
        # Simple keyword detection for location (mock logic)
        location = "Unknown"
        if "london" in query.lower():
            location = "London"
        elif "paris" in query.lower():
            location = "Paris"
        elif "new york" in query.lower():
            location = "New York"
        else:
            location = "Seattle" # Default

        # Mock data response
        return {
            "intent": "weather_report",
            "data": {
                "location": location,
                "temperature": 22 if location == "Seattle" else 15,
                "unit": "C",
                "condition": "Cloudy",
                "humidity": "65%",
                "timestamp": datetime.now().isoformat()
            }
        }
