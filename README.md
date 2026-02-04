# Fabio Agent API

Production-ready Fastify API wrapper for the OpenAI Agent Builder workflow SDK.

## Requirements

- Node.js 18+
- An OpenAI API key

## Setup

```bash
npm install
cp .env.example .env
```

Set the environment variables in `.env`:

- `OPENAI_API_KEY`
- `APP_API_KEY` (used for `x-api-key` auth)
- `PORT` (optional, default `3000`)

## Run locally

```bash
npm run dev
```

## API

### Health

```
GET /health
```

### Run workflow

```
POST /run
X-API-Key: <APP_API_KEY>
Content-Type: application/json

{
  "input_as_text": "texto do usuário"
}
```

## Render deployment

1. Push this repo to GitHub.
2. In Render, create a **Web Service** from the repo.
3. Set:
   - Build Command: `npm ci && npm run build`
   - Start Command: `npm start`
4. Add environment variables:
   - `OPENAI_API_KEY`
   - `APP_API_KEY`
   - `LOG_LEVEL` (optional)
5. Deploy. Use `GET /health` for health checks.

## Notes

- The API enforces `x-api-key` for the `/run` endpoint.
- Rate limit is set to 60 requests per minute per IP.
