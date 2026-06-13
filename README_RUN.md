# Run SViam (browser, full-duplex)

Windows / PowerShell. The venv already exists at `venv/`.

## 1. Install deps (once)
```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Keys
`.env` must contain (already set up):
```
GROQ_API_KEY=...
DEEPGRAM_API_KEY=...
# optional: GEMINI_API_KEY, OPENAI_API_KEY
# optional overrides: CONV_LLM_PROVIDER, GROQ_CONV_MODEL, JUDGE_PROVIDER, JUDGE_MODEL, SILENCE_SECS
```

## 3. Start the server
```powershell
venv\Scripts\python.exe server.py
```
You should see: `SViam WebRTC server on http://localhost:8000`.

## 4. Open it
- Open **http://localhost:8000** in **Chrome**.
- Click **Connect**, allow the microphone.
- The interviewer greets you. Answer out loud.
- **You can interrupt it mid-sentence** — it goes silent and responds to what you said (full-duplex via the browser's echo cancellation).
- Transcript + a Listening / Speaking / Thinking status update live.
- Click **End** to stop the session. `Ctrl+C` in the terminal stops the server.

## Notes
- Browser AEC means **no headset needed** for barge-in (unlike the deprecated local-mic app).
- One pipeline runs per browser connection.
- Models: conversation + judge = Groq `llama-4-scout` (swappable via `.env`); STT/TTS = Deepgram nova-3 / Aura-2.
- The old local-mic entry point (`voice_agent.py`) still runs but is **deprecated** per `CLAUDE.md`.

## Deprecated (local mic) — for reference only
```powershell
venv\Scripts\python.exe voice_agent.py
```
