"""Main orchestration loop for the Chef Buddy cooking voice agent."""

import json
import os
import time
from pathlib import Path
from typing import Optional

from agent import CookingAgent
from camera import Camera
from voice_input import listen
from voice_output import speak

RECIPES_DIR = Path(__file__).parent / "recipes"
CAMERA_CHECK_INTERVAL = 15  # seconds between auto camera snapshots


def load_recipes() -> list[dict]:
    recipes = []
    for path in sorted(RECIPES_DIR.glob("*.json")):
        try:
            with open(path) as f:
                recipes.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Warning] Could not load {path.name}: {e}")
    return recipes


def select_recipe_voice(agent: CookingAgent, recipes: list[dict]) -> Optional[dict]:
    names = [r.get("name", "Unknown") for r in recipes]
    options_str = ", ".join(f"{i+1}. {n}" for i, n in enumerate(names))
    speak(f"I have {len(recipes)} recipes available: {options_str}. Which one would you like to cook?")

    for _ in range(3):
        user_input = listen(timeout=10)
        if not user_input:
            speak("I didn't catch that. Which recipe would you like?")
            continue

        response = agent.chat(
            f"The user said: '{user_input}'. Available recipes: {options_str}. "
            f"Reply with ONLY the exact recipe name from the list that best matches, "
            f"or 'unknown' if unclear."
        )
        matched = next(
            (r for r in recipes if r.get("name", "").lower() in response.lower()),
            None,
        )
        if matched:
            return matched
        speak("I didn't quite catch which recipe. Could you say the name again?")

    # Default to first recipe
    speak(f"I'll go with {recipes[0].get('name')} then!")
    return recipes[0]


def cooking_loop(agent: CookingAgent, camera: Camera):
    last_camera_check = time.time()

    speak("I'm ready when you are. Just talk to me as you cook, and I'll guide you along!")

    while True:
        now = time.time()
        time_since_check = now - last_camera_check

        # Auto camera snapshot if user has been quiet
        if camera.available and time_since_check >= CAMERA_CHECK_INTERVAL:
            frame = camera.capture_frame()
            if frame:
                response = agent.chat(
                    "Take a look at what's happening on the camera and give a brief, helpful observation about the cooking progress.",
                    frame,
                )
                speak(response)
            last_camera_check = time.time()
            continue

        user_input = listen(timeout=CAMERA_CHECK_INTERVAL - int(time_since_check))

        if user_input is None:
            continue

        lower = user_input.lower().strip()

        if any(word in lower for word in ("quit", "exit", "done", "finished", "stop", "bye")):
            speak("Great cooking session! Enjoy your meal!")
            break

        if any(phrase in lower for phrase in ("next step", "next", "what's next", "move on")):
            frame = camera.capture_frame() if camera.available else None
            response = agent.next_step(frame)
            speak(response)
            last_camera_check = time.time()
            continue

        frame = camera.capture_frame() if camera.available else None
        response = agent.chat(user_input, frame)
        speak(response)
        if frame:
            last_camera_check = time.time()


def main():
    print("=" * 50)
    print("  Chef Buddy — Cooking Voice Agent")
    print("=" * 50)

    recipes = load_recipes()
    if not recipes:
        print("[Error] No recipes found in recipes/ directory.")
        return

    agent = CookingAgent()

    with Camera() as camera:
        if camera.available:
            print("[Camera] Webcam detected — vision mode active.")
        else:
            print("[Camera] No webcam found — running in audio-only mode.")

        speak("Hey there! I'm Chef Buddy, your cooking assistant. Let's get cooking!")

        selected = select_recipe_voice(agent, recipes)
        if not selected:
            speak("I couldn't find a matching recipe. Let me know when you're ready to try again!")
            return

        agent.load_recipe(selected)
        speak(agent.chat(
            f"The user has selected '{selected.get('name')}'. "
            f"Give them step 1 instructions in a friendly, spoken way."
        ))

        cooking_loop(agent, camera)


if __name__ == "__main__":
    main()
