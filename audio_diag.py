"""Diagnose local audio devices to fix choppy TTS playback."""
import pyaudio

pa = pyaudio.PyAudio()

print("=== Host APIs ===")
for i in range(pa.get_host_api_count()):
    info = pa.get_host_api_info_by_index(i)
    print(f"  [{i}] {info['name']}  (default in={info['defaultInputDevice']}, out={info['defaultOutputDevice']})")

di = pa.get_default_input_device_info()
do = pa.get_default_output_device_info()
print("\n=== Default INPUT device ===")
print(f"  index={di['index']} name={di['name']!r}")
print(f"  maxInputChannels={di['maxInputChannels']} defaultSampleRate={di['defaultSampleRate']}")
print("\n=== Default OUTPUT device ===")
print(f"  index={do['index']} name={do['name']!r}")
print(f"  maxOutputChannels={do['maxOutputChannels']} defaultSampleRate={do['defaultSampleRate']}")

print("\n=== Output format support (16-bit) ===")
for rate in (16000, 24000, 44100, 48000):
    for ch in (1, 2):
        try:
            ok = pa.is_format_supported(
                rate, output_device=do["index"], output_channels=ch,
                output_format=pyaudio.paInt16,
            )
        except Exception as e:
            ok = f"NO ({type(e).__name__})"
        print(f"  out {rate} Hz x{ch}ch -> {ok}")

print("\n=== Input format support (16-bit) ===")
for rate in (16000, 24000, 44100, 48000):
    try:
        ok = pa.is_format_supported(
            rate, input_device=di["index"], input_channels=1, input_format=pyaudio.paInt16,
        )
    except Exception as e:
        ok = f"NO ({type(e).__name__})"
    print(f"  in {rate} Hz x1ch -> {ok}")

pa.terminate()
