from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

functions = [
    {"name": "get_ticket_status", "description": "Get status of a support ticket", "parameters": {"type": "object", "properties": {"ticket_id": {"type": "integer"}}, "required": ["ticket_id"]}},
    {"name": "schedule_meeting", "description": "Schedule a meeting", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "meeting_room": {"type": "string"}}, "required": ["date", "time", "meeting_room"]}},
    {"name": "get_expense_balance", "description": "Get expense balance for employee", "parameters": {"type": "object", "properties": {"employee_id": {"type": "integer"}}, "required": ["employee_id"]}},
    {"name": "calculate_performance_bonus", "description": "Calculate performance bonus", "parameters": {"type": "object", "properties": {"employee_id": {"type": "integer"}, "current_year": {"type": "integer"}}, "required": ["employee_id", "current_year"]}},
    {"name": "report_office_issue", "description": "Report an office issue", "parameters": {"type": "object", "properties": {"issue_code": {"type": "integer"}, "department": {"type": "string"}}, "required": ["issue_code", "department"]}},
]

@app.get("/execute")
def execute(q: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}],
            functions=functions,
            function_call="auto"
        )
        message = response.choices[0].message
        if message.function_call:
            return {"name": message.function_call.name, "arguments": message.function_call.arguments}
        else:
            return {"error": "No function call", "content": message.content}
    except Exception as e:
        return {"error": str(e)}
