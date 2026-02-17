## TinyLlama Production LLM API

This project is a **production-style REST API** built with **FastAPI** that exposes a **text generation endpoint** backed by the **TinyLlama-1.1B-Chat** model from Hugging Face.

The API is designed to:
- Load the language model once at startup.
- Expose a simple `/generate` endpoint for text generation.
- Return structured responses with basic token-usage info.

---

## Project structure

- `main.py` – FastAPI app, routes, and server entrypoint.
- `ml_engine.py` – `LLMEngine` wrapper around Hugging Face `pipeline` for TinyLlama.
- `schemas.py` – Pydantic request/response models.
- `requirements.txt` – Python dependencies.

---

## Requirements

- Python **3.10+** (recommended; project currently uses Python 3.13 on your machine).
- Internet access (first model download from Hugging Face).
- A machine with enough RAM/VRAM to host `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
  (CPU-only will work but be slower; GPU will be auto-used if available).

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `torch`
- `transformers`
- `accelerate`
- `fastapi[standard]`
- `uvicorn`
- `pydantic`

---

## How it works

### Model engine (`ml_engine.py`)

- Defines an `LLMEngine` class that:
  - Lazily loads the TinyLlama model via `transformers.pipeline` in `load_model()`.
  - Uses `torch.bfloat16` and `device_map="auto"` to optimize memory and pick GPU if present.
  - Builds a simple chat-style prompt with system + user messages.
  - Calls the pipeline and returns only the generated continuation (prompt is stripped off).
- Exposes a **global instance**:

```python
llm_engine = LLMEngine()
```

This is what the FastAPI app uses.

### Schemas (`schemas.py`)

- `GenerationRequest`
  - `prompt: str` – input text (required).
  - `max_tokens: int = 256` – number of new tokens (10–1024).
  - `temperature: float = 0.7` – sampling temperature (0.0–1.0).
- `GenerationResponse`
  - `result: str` – generated text.
  - `token_usage: int` – rough word-count-based usage estimate.

### API (`main.py`)

- Uses a FastAPI **lifespan** function to:
  - Call `llm_engine.load_model()` once at startup.
- Endpoints:
  - `GET /` – health check.
  - `POST /generate` – text generation.
- Also includes a `__main__` block so you can run:

```bash
python main.py
```

---

## Running the API

From the project root (`Production_LLM_API`), run:

```bash
python main.py
```

You should see Uvicorn logs similar to:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

The first request after startup might be slower while TinyLlama loads.

### Alternative: using `uvicorn` directly

You can also run:

```bash
uvicorn main:app --reload
```

> Note: If you use `fastapi dev`, use:
>
> ```bash
> fastapi dev --app main:app
> ```

---

## API usage

### Health check

**Request**

```http
GET / HTTP/1.1
Host: 127.0.0.1:8000
```

**Response (200)**

```json
{
  "status": "online",
  "model": "TinyLlama-1.1B"
}
```

### Generate text

**Endpoint**

- `POST /generate`
- Request body: `GenerationRequest`
- Response body: `GenerationResponse`

**Example request (JSON)**

```json
{
  "prompt": "Explain quantum physics in 5-year-old terms.",
  "max_tokens": 128,
  "temperature": 0.7
}
```

**Example response (JSON)**

```json
{
  "result": "Quantum physics is like tiny invisible balls that like to play by funny rules...",
  "token_usage": 24
}
```

### Interactive docs

Once the server is running, open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

You can call `/generate` directly from the Swagger UI.

---

## Notes for production hardening

This project is a good starting point, but for real production use you may want to:

- Add **authentication** (API keys, OAuth2, etc.).
- Implement **request limits** and **rate limiting**.
- Add **structured logging** and **tracing**.
- Add timeouts and guardrails around generation (max prompt size, max latency).
- Containerize the app (Docker) and deploy behind a reverse proxy (e.g., Nginx, Traefik).

---

## License

MIT License

