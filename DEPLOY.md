# Deploying SViam to Render

Deploys the **`deploy-render`** branch (Docker) to a Render **web service**, with a
**Metered TURN** relay so WebRTC audio connects in the cloud. `main` + the
`demo-baseline-direct-socratic` tag stay frozen — the interview code is unchanged;
this is additive infra only.

> Why TURN is required: Render web services are HTTP/WebSocket only and route no
> inbound UDP, so the browser and the server-side aiortc can't reach each other
> directly. A TURN relay (both sides connect *outbound* to it) is the only path
> for the audio. Without it the transcript socket works but the voice never connects.

---

## 1. Metered TURN (US West)
1. Create a free account at https://www.metered.ca/ → **Metered TURN**.
2. Create an app; note your **subdomain** (e.g. `your-app.metered.live`) and **API key**.
3. In the Metered dashboard set the TURN region to **US West** (near the California user).
4. The server fetches fresh credentials at
   `https://<METERED_DOMAIN>/api/v1/turn/credentials?apiKey=<METERED_API_KEY>`
   (dynamic method — nothing hardcoded).

## 2. Create the Render service
- Render dashboard → **New → Blueprint** → pick this repo → branch **`deploy-render`**.
  Render reads `render.yaml` (Docker, region `oregon`, plan `pro` = 2 CPU / 4 GB).
- To change the instance size later, edit the **one** `plan:` line in `render.yaml`
  (`standard` = 1 CPU/2 GB cheaper · `pro` = 2 CPU/4 GB · `pro_plus` = 4 CPU/8 GB).

## 3. Environment variables (set in the Render dashboard — never committed)
| Key | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `DEEPGRAM_API_KEY` | your Deepgram key |
| `GEMINI_API_KEY` | your Gemini key |
| `JUDGE_PROVIDER` | `groq` (pinned in render.yaml) |
| `JUDGE_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` (pinned) |
| `METERED_DOMAIN` | `your-app.metered.live` |
| `METERED_API_KEY` | your Metered API key |

`PORT` is injected by Render automatically — **do not set it**.

## 4. Deploy & verify, in order (don't skip a gate)
- **(a) Builds + boots** — deploy succeeds, logs show `SViam WebRTC server on http://0.0.0.0:<PORT>`, health check green. *(No audio yet.)*
- **(b) Audio connects** — open the Render URL in Chrome, Connect, speak; you hear the bot. Check `GET /api/ice` returns STUN **and** a `turn:`/`turns:` entry. If audio fails, TURN env vars are wrong or Metered region/quota is off.
- **(c) Full interview** — natural flow + the wrong-answer cut-in ("That's not right…"), no stalls, no double replies. **Only after (c) do we merge to `main`.**
- **(d) Smoke tests** — you first, then the senior from California.

## Notes
- **Metrics:** `enable_metrics` is on — Render logs show per-turn latency/jitter.
- **Cold starts:** not a concern (Pro workspace, always-on instance).
- **Local dev unchanged:** with no `METERED_*` set, `/api/ice` returns STUN only and the app behaves exactly as before.
