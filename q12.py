from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import re

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

@app.get("/execute")
def execute(q: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result
    except Exception as e:
        return {"error": str(e)}
