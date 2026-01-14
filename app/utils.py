import json

def create_text_response(text: str) -> dict:
    """
    Creates a standard text response in the format expected by ag_ui.
    """
    return {
        "type": "text",
        "content": text
    }

def create_weather_widget(data: dict) -> dict:
    """
    Creates an A2UI widget for weather data.
    Structure based on generic declarative UI concepts (Card > Column > Text).
    """
    return {
        "type": "card",
        "title": f"Weather in {data['location']}",
        "content": [
            {
                "type": "row",
                "children": [
                    {
                        "type": "text",
                        "style": "heading",
                        "content": f"{data['temperature']}°{data['unit']}"
                    },
                    {
                        "type": "text",
                        "style": "subheading",
                        "content": data['condition']
                    }
                ]
            },
            {
                "type": "text",
                "content": f"Humidity: {data['humidity']}"
            },
            {
                "type": "text",
                "style": "caption",
                "content": f"Updated: {data['timestamp']}"
            }
        ]
    }

def create_doc_results_widget(data: dict) -> dict:
    """
    Creates an A2UI widget for documentation search results.
    """
    results_ui = []
    for res in data['results']:
        results_ui.append({
            "type": "container",
            "style": "bordered",
            "children": [
                {
                    "type": "link",
                    "text": res['title'],
                    "url": res['url']
                },
                {
                    "type": "text",
                    "content": res['snippet']
                }
            ]
        })

    return {
        "type": "card",
        "title": f"Results for: {data['query']}",
        "subtitle": data['source'],
        "content": results_ui
    }
