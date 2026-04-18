def play_prompt(prompt: str) -> dict:
    """
    Placeholder audio controller.

    Later, replace this with:
    - pygame
    - aplay
    - espeak
    - pyttsx3
    - prerecorded audio files
    """

    print(f"[AUDIO] {prompt}")

    return {
        "audio_played": True,
        "prompt": prompt,
    }