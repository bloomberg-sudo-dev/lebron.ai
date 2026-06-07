# Chef Buddy — Cooking Voice Agent

An interactive voice agent that watches you cook via your webcam and guides you through recipes in real time. Powered by Claude claude-opus-4-8 with vision.

## Features

- **Voice interaction** — talk naturally while your hands are busy cooking
- **Vision awareness** — Chef Buddy watches your camera and comments on your progress
- **Step-by-step guidance** — works through recipes at your pace
- **Friendly persona** — encouraging, concise, optimized for spoken responses
- **Audio-only fallback** — works without a webcam too

## Requirements

- Python 3.11+
- A microphone
- A webcam (optional — degrades gracefully without one)
- `ANTHROPIC_API_KEY` environment variable set

## Setup

```bash
cd demos/cooking-voice-agent

# Install dependencies
pip install -r requirements.txt

# On Linux, PyAudio may need portaudio first:
# sudo apt-get install portaudio19-dev python3-dev

# Set your API key
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

```bash
python app.py
```

Chef Buddy will:
1. Greet you and ask which recipe you want to cook
2. Walk you through each step
3. Check in on your camera every ~15 seconds
4. Respond to anything you say naturally

**Voice commands:**
- Say `"next step"` or `"next"` to advance to the next recipe step
- Say `"done"`, `"finished"`, or `"quit"` to end the session

## Recipes

Three recipes are included in `recipes/`:

- `spaghetti_carbonara.json` — Classic Roman pasta
- `scrambled_eggs.json` — Creamy soft scrambled eggs
- `stir_fry_vegetables.json` — Quick garlic vegetable stir-fry

### Adding Your Own Recipes

Create a new `.json` file in `recipes/` following this structure:

```json
{
  "name": "Recipe Name",
  "description": "Brief description",
  "servings": 2,
  "prep_time_minutes": 10,
  "cook_time_minutes": 20,
  "ingredients": ["ingredient 1", "ingredient 2"],
  "steps": [
    {
      "number": 1,
      "title": "Step title",
      "instruction": "What to do in this step."
    }
  ]
}
```

## Architecture

```
app.py           — Main loop: voice input → agent → voice output
agent.py         — Claude claude-opus-4-8 with vision, streaming, conversation history
camera.py        — OpenCV webcam capture (JPEG, max 1280px wide)
voice_input.py   — Speech recognition via Google STT
voice_output.py  — Text-to-speech via edge-tts (Microsoft Azure Neural)
recipes/         — JSON recipe files
```
