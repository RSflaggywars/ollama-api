# Ollama API

A small FastAPI service that forwards chat requests to an Ollama server using the official Python `ollama` library. The repository can be public; publishing this source code does **not** host Ollama or make an API endpoint available on the internet. Run this service on a machine that has Ollama installed, or deploy it deliberately to a server with appropriate network security.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- A model downloaded in Ollama, for example `llama3.2`

```powershell
ollama pull llama3.2
```

## Install and run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The API listens only on `127.0.0.1` by default, and connects to Ollama at `http://127.0.0.1:11434`. Set `OLLAMA_BASE_URL` if Ollama is running at a different trusted address. The API address can be changed with Uvicorn's `--host` and `--port` arguments.

Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Endpoints

- `GET /health` — API process health check; does not expose host configuration.
- `GET /api/models` — list model names available to Ollama.
- `POST /api/chat` — send a chat request to the selected local model.

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/chat `
  -ContentType 'application/json' `
  -Body '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'
```

Chat messages accept `system`, `user`, and `assistant` roles. Requests are limited to 256 KiB, at most 50 messages, and 65,536 combined message characters. Model names are validated; clients cannot choose an Ollama server URL.

## Protecting a deployment

An API key is optional for local-only use. Set `API_KEY` to a long, unique secret to require `Authorization: Bearer <key>` on `/api/models` and `/api/chat`. Never commit the key to this repository.

```powershell
$env:API_KEY = 'replace-with-a-long-random-secret'
py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For a remote deployment, configure the API host, TLS termination, firewall, and secret management intentionally. Do not expose Ollama's port directly to untrusted networks. The health endpoint is intentionally unauthenticated and returns only a generic status.

## Tests

```powershell
py -m pip install -r requirements-dev.txt
py -m pytest
```
