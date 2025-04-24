from fastapi import FastAPI, Query
from predict import get_response

# Initialize FastAPI app
app = FastAPI()

# Function to remove duplicate sentences in a response
def remove_duplicates(text):
    sentences = text.split(". ")
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        if sentence.strip() and sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    return ". ".join(unique_sentences)

@app.get("/search")
def search(query: str = Query(..., description="Input text to search for a response")):
    # Get the single most relevant response
    response = get_response(query)

    # Clean response (remove line breaks and duplicates)
    cleaned_response = remove_duplicates(response.replace("\n", " ").replace("\r", " ").strip())

    return {
        "query": query,
        "response": cleaned_response
    }
