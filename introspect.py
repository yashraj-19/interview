import importlib, inspect

import pipecat
print("pipecat version:", getattr(pipecat, "__version__", "unknown"))
print("=" * 60)

# (module, classname) candidates to probe
targets = [
    ("pipecat.transports.local.audio", "LocalAudioTransport"),
    ("pipecat.transports.local.audio", "LocalAudioTransportParams"),
    ("pipecat.services.deepgram.stt", "DeepgramSTTService"),
    ("pipecat.services.deepgram.tts", "DeepgramTTSService"),
    ("pipecat.services.google.llm", "GoogleLLMService"),
    ("pipecat.services.groq.llm", "GroqLLMService"),
    ("pipecat.audio.vad.silero", "SileroVADAnalyzer"),
    ("pipecat.pipeline.pipeline", "Pipeline"),
    ("pipecat.pipeline.task", "PipelineTask"),
    ("pipecat.pipeline.task", "PipelineParams"),
    ("pipecat.pipeline.runner", "PipelineRunner"),
    ("pipecat.processors.aggregators.openai_llm_context", "OpenAILLMContext"),
]

for mod, name in targets:
    try:
        m = importlib.import_module(mod)
        obj = getattr(m, name)
        try:
            sig = str(inspect.signature(obj.__init__))
        except (TypeError, ValueError):
            sig = "(sig n/a)"
        print(f"OK   {mod}.{name}")
        print(f"     __init__{sig}")
    except Exception as e:
        print(f"FAIL {mod}.{name} -> {type(e).__name__}: {e}")

print("=" * 60)
# Frame types relevant to interruptions / context
for mod, names in [
    ("pipecat.frames.frames",
     ["TTSSpeakFrame", "TextFrame", "LLMMessagesFrame", "BotInterruptionFrame",
      "StartInterruptionFrame", "TranscriptionFrame", "InterimTranscriptionFrame",
      "LLMRunFrame", "EndFrame"]),
]:
    m = importlib.import_module(mod)
    for n in names:
        print(("OK   " if hasattr(m, n) else "MISS ") + f"{mod}.{n}")
