import hmac
import logging
import os
import re
from typing import Annotated, Literal

import httpx
import ollama
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
API_KEY = os.getenv("API_KEY")
REQUEST_SIZE_LIMIT = 256 * 1024
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16_384)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=128)
    messages: list[Message] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_request(self) -> "ChatRequest":
        if (
            not MODEL_NAME_PATTERN.fullmatch(self.model)
            or "://" in self.model
        ):
            raise ValueError("model contains unsupported characters")
        if sum(len(message.content) for message in self.messages) > 65_536:
            raise ValueError("combined message content must not exceed 65536 characters")
        return self


class ChatResponse(BaseModel):
    model: str
    message: Message
    done: bool


class RequestSizeLimitMiddleware:
    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for key, value in scope.get("headers", []) if key.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.limit:
                    await self._send_too_large(send)
                    return
            except ValueError:
                await self._send_too_large(send)
                return

        body_chunks = []
        body_size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            body_size += len(chunk)
            if body_size > self.limit:
                await self._send_too_large(send)
                return
            body_chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(body_chunks)
        replayed = False

        async def replay_body():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_body, send)

    @staticmethod
    async def _send_too_large(send):
        body = b'{"detail":"Request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


async def require_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
    if API_KEY is None:
        return
    scheme, separator, supplied_key = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied_key, API_KEY)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )


app = FastAPI(
    title="Ollama API",
    description="A small, validated HTTP API for a locally running Ollama server.",
    version="1.0.0",
)
app.add_middleware(RequestSizeLimitMiddleware, limit=REQUEST_SIZE_LIMIT)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models", dependencies=[Depends(require_api_key)])
async def list_models():
    try:
        result = await ollama.AsyncClient(host=OLLAMA_BASE_URL, timeout=10).list()
    except ollama.ResponseError as exc:
        logger.warning("Ollama rejected model listing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ollama could not list models",
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning("Could not reach Ollama: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to Ollama",
        ) from exc
    return {"models": [{"name": model.model} for model in result.models]}


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
async def chat(request: ChatRequest):
    try:
        result = await ollama.AsyncClient(host=OLLAMA_BASE_URL, timeout=120).chat(
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
        )
    except ollama.ResponseError as exc:
        logger.warning("Ollama rejected chat request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ollama could not complete the chat request",
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning("Could not reach Ollama: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to Ollama",
        ) from exc

    return ChatResponse(
        model=result.model,
        message=Message(role=result.message.role, content=result.message.content),
        done=result.done,
    )
