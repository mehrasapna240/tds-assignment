from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM2OTVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.WcSwtS3kruSbFeC_U4UGWsZ9CDwedL3EpFryK0fhXMU",
    base_url="https://aipipe.org/openai/v1"
)

SYSTEM_PROMPT = """You are a function router. Given a user query, return ONLY a JSON object with "name" and "arguments" fields.

Available functions:
- get_ticket_status(ticket_id: int)
- schedule_meeting(date: str, time: str, meeting_room: str)
- get_expense_balance(employee_id: int)
- calculate_performance_bonus(employee_id: int, current_year: int)
- report_office_issue(issue_code: int, department: str)

Return ONLY JSON like: {"name": "function_name", "arguments": "{\"param\": value}"}
Arguments must be a JSON-encoded string."""

def call_llm(q: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

@app.get("/execute")
def execute(q: str):
    return call_llm(q)

@app.get("/")
def root(q: str = None):
    if q:
        return call_llm(q)
    return {"status": "ok"}

class Comment(BaseModel):
    comment: str

@app.post("/comment")
def analyze_comment(body: Comment):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze the sentiment of the comment. Return ONLY a JSON object with 'sentiment' (positive/negative/neutral) and 'rating' (integer 1-5, where 5=highly positive, 1=highly negative)."},
                {"role": "user", "content": body.comment}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}
