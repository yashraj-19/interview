import importlib, inspect, pkgutil

def probe(mod, names):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(f"  [module FAIL] {mod} -> {e}")
        return
    for n in names:
        print(("  OK   " if hasattr(m, n) else "  MISS ") + f"{mod}.{n}")

print("### LLM context / aggregator candidates")
for mod in [
    "pipecat.processors.aggregators.llm_context",
    "pipecat.processors.aggregators.llm_response",
    "pipecat.processors.aggregators.llm_response_universal",
]:
    probe(mod, ["LLMContext", "LLMContextAggregatorPair",
                "LLMUserAggregatorParams", "LLMAssistantAggregatorParams"])

print("\n### interruption-related frame names in pipecat.frames.frames")
import pipecat.frames.frames as F
for n in sorted(dir(F)):
    if "Interrup" in n or "Bot" in n or "Stop" in n:
        print("  ", n)

print("\n### LLMContext signature")
try:
    from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextAggregatorPair
    print("  LLMContext.__init__", inspect.signature(LLMContext.__init__))
    print("  LLMContextAggregatorPair.__init__", inspect.signature(LLMContextAggregatorPair.__init__))
    print("  Pair methods:", [m for m in dir(LLMContextAggregatorPair) if not m.startswith("_")])
except Exception as e:
    print("  ERR", e)

print("\n### pydantic fields: LocalAudioTransportParams")
from pipecat.transports.local.audio import LocalAudioTransportParams
for k, v in LocalAudioTransportParams.model_fields.items():
    print(f"   {k}: default={v.default!r}")

print("\n### pydantic fields: PipelineParams")
from pipecat.pipeline.task import PipelineParams
for k, v in PipelineParams.model_fields.items():
    print(f"   {k}: default={v.default!r}")

print("\n### pydantic fields: VADParams")
from pipecat.audio.vad.vad_analyzer import VADParams
for k, v in VADParams.model_fields.items():
    print(f"   {k}: default={v.default!r}")

print("\n### PipelineTask public methods")
from pipecat.pipeline.task import PipelineTask
print("  ", [m for m in dir(PipelineTask) if not m.startswith("_")])

print("\n### LocalAudioTransport public methods")
from pipecat.transports.local.audio import LocalAudioTransport
print("  ", [m for m in dir(LocalAudioTransport) if not m.startswith("_")])
