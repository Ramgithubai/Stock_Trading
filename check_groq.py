import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("GROQ_API_KEY")

print(f"GROQ API Key: {'Found' if api_key else 'Not found'}")

try:
    # Initialize the ChatGroq model
    chat = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key
    )
    
    # Test the model with a simple query
    result = chat.invoke("Hello, can you tell me what model you are?")
    
    print("\nGroq test successful!")
    print(f"Response: {result.content[:100]}...")
    
except Exception as e:
    print(f"\nError testing Groq: {str(e)}")