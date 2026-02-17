from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from ml_engine import llm_engine
from schemas import GenerationRequest, GenerationResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    llm_engine.load_model()
    yield
    # Shutdown: Clean up resources (if needed)
    print("🛑 Shutting down model engine...")

# Initialize the app with the lifespan
app = FastAPI(title="TinyLlama API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "online", "model": "TinyLlama-1.1B"}

@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    """
    Endpoint to generate text.
    Note: We use a standard 'def' (not async def) here.
    """
    try:
        # Generate the text
        result = llm_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens
        )
        
        return GenerationResponse(
            result=result,
            token_usage=len(result.split()) # Rough estimate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
    )