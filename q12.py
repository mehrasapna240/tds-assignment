from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

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

tools = [
    {"type": "function", "function": {"name": "get_ticket_status", "parameters": {"type": "object", "properties": {"ticket_id": {"type": "integer"}}, "required": ["ticket_id"]}}},
    {"type": "function", "function": {"name": "schedule_meeting", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "meeting_room": {"type": "string"}}, "required": ["date", "time", "meeting_room"]}}},
    {"type": "function", "function": {"name": "get_expense_balance", "parameters": {"type": "object", "properties": {"employee_id": {"type": "integer"}}, "required": ["employee_id"]}}},
    {"type": "function", "function": {"name": "calculate_performance_bonus", "parameters": {"type": "object", "properties": {"employee_id": {"type": "integer"}, "current_year": {"type": "integer"}}, "required": ["employee_id", "current_year"]}}},
    {"type": "function", "function": {"name": "report_office_issue", "parameters": {"type": "object", "properties": {"issue_code": {"type": "integer"}, "department": {"type": "string"}}, "required": ["issue_code", "department"]}}},
]

@app.get("/execute")
def execute(q: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}],
            tools=tools,
            tool_choice="auto"
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
    except Exception as e:
        return {"error": str(e)}
