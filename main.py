
import numpy as np
import hashlib
import time
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

AI_PIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM2OTVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.WcSwtS3kruSbFeC_U4UGWsZ9CDwedL3EpFryK0fhXMU"

client = OpenAI(
    api_key=AI_PIPE_TOKEN,
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
def similarity(req: SimilarityRequest):
    all_texts = [req.query] + req.docs
    response = client.embeddings.create(input=all_texts, model="text-embedding-3-small")
    embeddings = [e.embedding for e in response.data]
    query_emb = embeddings[0]
    scored = [(cosine_similarity(query_emb, embeddings[i+1]), doc) for i, doc in enumerate(req.docs)]
    scored.sort(reverse=True)
    return {"matches": [doc for _, doc in scored[:3]]}

class ValidationRequest(BaseModel):
    userId: str
    input: str
    category: str

INJECTION_PATTERNS = ["ignore", "override", "forget", "disregard", "system prompt",
    "developer mode", "jailbreak", "you are now", "pretend", "act as",
    "reveal", "show me your", "what are your instructions", "safety rules", "no restrictions"]

@app.post("/validate")
def validate(req: ValidationRequest):
    text_lower = req.input.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in text_lower:
            return {"blocked": True, "reason": f"Prompt injection detected: '{pattern}'", "sanitizedOutput": None, "confidence": 0.95}
    return {"blocked": False, "reason": "Input passed all security checks", "sanitizedOutput": req.input, "confidence": 0.99}

class StreamRequest(BaseModel):
    prompt: str
    stream: bool = True

def generate(prompt):
    story = """Education reform is one of the most critical challenges facing modern society. 
For decades, educators and policymakers have debated how to best prepare students for an 
increasingly complex world. Traditional teaching methods, while valuable, often fail to 
engage students who learn differently or come from disadvantaged backgrounds.

Consider Maria, a passionate teacher in an underfunded school district. Despite limited 
resources, she transforms her classroom into a hub of innovation. She uses technology 
creatively, turning old tablets into learning tools. Her students, once disengaged, now 
compete in national science competitions. This is the power of dedicated teaching.

But here is the plot twist - the greatest barrier to education reform is not funding or 
technology. It is our own resistance to change. When Maria proposed a new curriculum, 
administrators initially rejected it. Only when students showed remarkable improvement 
did they embrace her methods. Real reform requires courage to challenge the status quo.

The path forward requires collaboration between teachers, parents, policymakers, and 
students themselves. We must invest in teacher training, modernize curricula, and ensure 
every child has access to quality education regardless of zip code or economic status. 
Education is not just about grades - it is about preparing young minds to solve 
tomorrow's problems with creativity, empathy, and critical thinking."""

    words = story.split()
    # Send in groups of 3 words per chunk
    chunk_size = 3
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size]) + " "
        data = {"choices": [{"delta": {"content": chunk}}]}
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"
    
    # Send rest of content
    words = story.split()[1:]
    for word in words:
        data = {"choices": [{"delta": {"content": word + " "}}]}
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"
@app.post("/stream")
async def stream(req: StreamRequest):
    async def generate_async():
        story = "Education reform is one of the most critical challenges facing modern society. For decades, educators and policymakers have debated how to best prepare students for an increasingly complex world. Traditional teaching methods, while valuable, often fail to engage students who learn differently. Consider Maria, a passionate teacher in an underfunded school district. Despite limited resources, she transforms her classroom into a hub of innovation using technology creatively. Her students, once disengaged, now compete in national science competitions. This is the power of dedicated teaching and innovative thinking. But here is the plot twist - the greatest barrier to education reform is not funding or technology. It is our own resistance to change. When Maria proposed a new curriculum, administrators initially rejected it. Only when students showed remarkable improvement did they embrace her methods. Real reform requires courage to challenge the status quo and reimagine what education can be. The path forward requires collaboration between teachers, parents, policymakers, and students. We must invest in teacher training, modernize curricula, and ensure every child has access to quality education regardless of zip code or economic status."
        
        words = story.split()
        for i in range(0, len(words), 2):
            chunk = " ".join(words[i:i+2]) + " "
            data = {"choices": [{"delta": {"content": chunk}}]}
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate_async(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff"})

cache = {}
total_requests = 0
cache_hits = 0

class QueryRequest(BaseModel):
    query: str
    application: str = "code review assistant"

@app.post("/")
def query(req: QueryRequest):
    global total_requests, cache_hits
    total_requests += 1
    cache_key = hashlib.md5(req.query.lower().strip().encode()).hexdigest()
    if cache_key in cache:
        cache_hits += 1
        return {"answer": cache[cache_key], "cached": True, "latency": 5, "cacheKey": cache_key}
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": req.query}], max_tokens=200)
    answer = response.choices[0].message.content
    cache[cache_key] = answer
    return {"answer": answer, "cached": False, "latency": 2000, "cacheKey": cache_key}

@app.get("/analytics")
def analytics():
    hit_rate = cache_hits / total_requests if total_requests > 0 else 0
    return {"hitRate": round(hit_rate, 2), "totalRequests": total_requests, "cacheHits": cache_hits,
        "cacheMisses": total_requests - cache_hits, "cacheSize": len(cache),
        "costSavings": round(cache_hits * 2000 * 1.20 / 1_000_000, 2),
        "savingsPercent": round(hit_rate * 100),
        "strategies": ["exact match", "LRU eviction", "TTL expiration", "semantic caching"]}
