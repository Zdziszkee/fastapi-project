import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import httpx
from uuid import uuid4


class MessageSource(Enum):
    USER = 1
    BOT = 2


class Message(BaseModel):
    message: str
    date_time: datetime.datetime
    source: MessageSource


class Answer(BaseModel):
    answer: str
    date_time: datetime.datetime


class Conversation(BaseModel):
    messages: List[Message]
    id: uuid4


app = FastAPI()

conversations = {}

# OpenAI API key and endpoint
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_API_URL = "https://api.openai.com/v1/engines/davinci/completions"


async def fetch_openai_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"].strip()


@app.post("/")
async def ask_question(message: str):
    conversation_id = uuid4()

    conversation = Conversation(
        messages=[
            Message(
                message=message,
                date_time=datetime.datetime.utcnow(),
                source=MessageSource.USER
            )
        ],
        id=conversation_id
    )


    conversations[conversation_id] = conversation


    try:
        answer_text = await fetch_openai_response(message)
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail="Error communicating with OpenAI API")

    conversation.messages.append(
        Message(
            message=answer_text,
            date_time=datetime.datetime.utcnow(),
            source=MessageSource.BOT
        )
    )

    conversations[conversation_id] = conversation

    return {"redirect_url": f"/conversation/{conversation_id}"}


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: int):
    conversation = conversations.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation
