"""Non-interactive build check: construct the whole pipeline, don't run audio."""
import voice_agent as va

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

print("[1] imports OK")
vad = SileroVADAnalyzer(); print("[2] SileroVADAnalyzer loaded")
stt = DeepgramSTTService(api_key=va.DEEPGRAM_API_KEY); print("[3] Deepgram STT OK")
tts = DeepgramTTSService(api_key=va.DEEPGRAM_API_KEY, voice=va.TTS_VOICE); print("[4] Deepgram TTS OK")
llm = GoogleLLMService(api_key=va.GEMINI_API_KEY, model=va.GEMINI_MODEL); print("[5] Gemini LLM OK")
ctx = LLMContext([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]); print("[6] LLMContext OK")
ts = UserTurnStrategies(
    start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
    stop=[SpeechTimeoutUserTurnStopStrategy()],
); print("[7] turn strategies OK")
agg = LLMContextAggregatorPair(ctx, user_params=LLMUserAggregatorParams(user_turn_strategies=ts)); print("[8] aggregator OK")
transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)); print("[9] LocalAudioTransport OK")
pipeline = Pipeline([
    transport.input(), VADProcessor(vad_analyzer=vad), stt, agg.user(), llm, tts,
    transport.output(), agg.assistant(),
]); print("[10] Pipeline assembled OK")
worker = PipelineWorker(pipeline, params=PipelineParams()); print("[11] PipelineWorker OK")
print("ALL GOOD — pipeline constructs cleanly")
