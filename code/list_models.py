from google import genai

# Replace with your actual Gemini API Key
API_KEY = "AIzaSyC8bAXRmaN-axvwUEJAlVB6pQEkCwMAsNs"

def list_available_models():
    # Initialize the client
    client = genai.Client(api_key=API_KEY)
    
    print("🔍 Fetching available models...\n")
    print("-" * 50)
    print(f"{'MODEL NAME':<35} | {'VERSION'}")
    print("-" * 50)
    
    try:
        # Call the ListModels endpoint
        for model in client.models.list():
            # We filter for models that support text generation ('generateContent')
            if 'generateContent' in model.supported_actions:
                 print(f"{model.name:<35} | {model.version}")
                 
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_available_models()