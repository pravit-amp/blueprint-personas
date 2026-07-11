import os

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Bottle Persona")
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

CHAT_MODEL = "gpt-4o-mini"

PERSONA_PROMPTS = {
    "optimist": (
        "Paraphrase the following message with an overly positive, upbeat spin. "
        "Keep it roughly the same length."
    ),
    "cynic": (
        "Paraphrase the following message with a skeptical, cynical tone. "
        "Keep it roughly the same length."
    ),
    "poet": (
        "Paraphrase the following message in dramatic, flowery, poetic language. "
        "Keep it roughly the same length."
    ),
}


class WebhookRequest(BaseModel):
    text: str


class WebhookResponse(BaseModel):
    text: str


def get_system_prompt() -> str:
    persona = os.environ.get("PERSONA", "").strip().lower()
    if persona not in PERSONA_PROMPTS:
        raise HTTPException(
            status_code=500,
            detail=f"PERSONA must be one of {sorted(PERSONA_PROMPTS)}; got {persona!r}",
        )
    return PERSONA_PROMPTS[persona]


@app.post("/webhook", response_model=WebhookResponse)
def webhook(body: WebhookRequest):
    system_prompt = get_system_prompt()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": body.text},
        ],
    )
    return {"text": response.choices[0].message.content.strip()}
