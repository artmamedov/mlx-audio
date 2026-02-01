#!/usr/bin/env python3
"""
Test script to validate the _conds fix in ChatterboxTurboTTS.

Tests:
1. Generate TTS audio with reference voice
2. Verify output is actual speech (not static noise) by checking audio properties
3. Optionally run STT to verify intelligibility
"""

import time
import numpy as np
from pathlib import Path

# Test text
TEST_TEXT = "Hello, this is a test of the text to speech system."
OUTPUT_FILE = Path("/tmp/tts_test_output.wav")
REF_AUDIO = Path("/Users/arturm/Github/mcp-speak/voice/voices/agent.wav")


def test_tts_generation():
    """Test that TTS generates actual speech."""
    from mlx_audio.tts.utils import load_model
    import soundfile as sf

    print("=" * 60)
    print("Testing ChatterboxTurboTTS with _conds fix")
    print("=" * 60)

    # Load model using the same method as CLI
    print("\n1. Loading model...")
    t0 = time.time()
    model = load_model("mlx-community/chatterbox-turbo-fp16")
    print(f"   Model loaded in {time.time() - t0:.2f}s")

    # Check if reference audio exists, if not we'll generate without it
    use_ref = REF_AUDIO.exists()
    if use_ref:
        print(f"\n2. Using reference audio: {REF_AUDIO}")
    else:
        print("\n2. No reference audio found, generating without voice cloning")

    # Generate audio
    print(f"\n3. Generating TTS for: '{TEST_TEXT}'")
    t0 = time.time()

    gen_kwargs = {
        "text": TEST_TEXT,
        "temperature": 0.8,
        "top_k": 1000,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    }

    if use_ref:
        gen_kwargs["ref_audio"] = str(REF_AUDIO)

    # Generate
    result = None
    for gen_result in model.generate(**gen_kwargs):
        result = gen_result

    gen_time = time.time() - t0
    print(f"   Generated in {gen_time:.2f}s")

    if result is None or result.audio is None:
        print("\n   ERROR: No audio generated!")
        return False

    # Get audio array
    audio = result.audio
    if hasattr(audio, 'numpy'):
        audio = audio.numpy()
    audio = np.array(audio).squeeze()

    print(f"\n4. Audio properties:")
    print(f"   - Shape: {audio.shape}")
    print(f"   - Duration: {len(audio) / model.sample_rate:.2f}s")
    print(f"   - Sample rate: {model.sample_rate} Hz")
    print(f"   - Min/Max: {audio.min():.4f} / {audio.max():.4f}")
    print(f"   - RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    # Check if audio is valid (not silence or static)
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    is_valid = True
    issues = []

    if rms < 0.001:
        issues.append("Audio is nearly silent (RMS < 0.001)")
        is_valid = False

    if peak < 0.01:
        issues.append("Audio has very low amplitude (peak < 0.01)")
        is_valid = False

    # Check for static (high frequency noise with no variation)
    # Real speech has varying amplitude over time
    chunk_size = model.sample_rate // 10  # 100ms chunks
    chunk_energies = []
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i + chunk_size]
        chunk_energies.append(np.sqrt(np.mean(chunk**2)))

    if chunk_energies:
        energy_std = np.std(chunk_energies)
        print(f"   - Energy variation (std): {energy_std:.4f}")

        if energy_std < 0.001:
            issues.append("Audio has no energy variation (likely static noise)")
            is_valid = False

    # Save audio
    print(f"\n5. Saving to {OUTPUT_FILE}")
    sf.write(str(OUTPUT_FILE), audio, model.sample_rate)

    if is_valid:
        print("\n✓ Audio appears to be valid speech!")
        print(f"   Play it with: afplay {OUTPUT_FILE}")
    else:
        print("\n✗ Audio validation failed:")
        for issue in issues:
            print(f"   - {issue}")

    return is_valid


def test_with_stt():
    """Optional: Verify TTS output with STT."""
    try:
        import mlx_whisper
    except ImportError:
        print("\nSkipping STT test (mlx-whisper not installed)")
        return None

    if not OUTPUT_FILE.exists():
        print("\nSkipping STT test (no output file)")
        return None

    print("\n" + "=" * 60)
    print("Running STT verification")
    print("=" * 60)

    print("\n1. Transcribing with mlx-whisper...")
    t0 = time.time()
    result = mlx_whisper.transcribe(str(OUTPUT_FILE), path_or_hf_repo="mlx-community/whisper-small")
    print(f"   Transcribed in {time.time() - t0:.2f}s")

    text = result.get("text", "").strip()
    print(f"\n2. Transcription: '{text}'")

    # Check if transcription is somewhat similar to input
    # (allowing for STT variations)
    input_words = set(TEST_TEXT.lower().split())
    output_words = set(text.lower().split())

    overlap = len(input_words & output_words)
    similarity = overlap / len(input_words) if input_words else 0

    print(f"\n3. Word overlap: {overlap}/{len(input_words)} ({similarity*100:.0f}%)")

    if similarity > 0.3:  # At least 30% word overlap
        print("   ✓ Transcription matches expected text reasonably well")
        return True
    else:
        print("   ✗ Transcription doesn't match expected text")
        return False


if __name__ == "__main__":
    # Run TTS test
    tts_ok = test_tts_generation()

    # Run STT verification
    stt_ok = test_with_stt()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"TTS Generation: {'PASS' if tts_ok else 'FAIL'}")
    if stt_ok is not None:
        print(f"STT Verification: {'PASS' if stt_ok else 'FAIL'}")

    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"Play with: afplay {OUTPUT_FILE}")
