"""Core cooking agent powered by Claude with vision support."""

import base64
from typing import Optional
import anthropic

SYSTEM_PROMPT = """You are Chef Buddy, a friendly and encouraging cooking assistant. You can see what the user is cooking via their camera and guide them through recipes step by step.

Your style:
- Warm, conversational, and encouraging — like a knowledgeable friend in the kitchen
- Concise responses optimized for being spoken aloud (1-3 sentences max per response)
- Use natural speech patterns, not bullet points or markdown
- Proactively notice what you see on camera and comment on it
- Celebrate progress and gently correct mistakes
- Keep track of recipe steps and guide the user through them

When you see an image, describe what you observe and relate it to the current recipe step.
Always stay focused on helping the user cook successfully and safely."""


class CookingAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.conversation_history = []
        self.current_recipe = None
        self.current_step = 0

    def chat(self, user_message: str, image_data: Optional[bytes] = None) -> str:
        content = []
        if image_data:
            b64_image = base64.standard_b64encode(image_data).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64_image,
                },
            })
        content.append({"type": "text", "text": user_message})

        self.conversation_history.append({"role": "user", "content": content})

        full_response = ""
        with self.client.messages.stream(
            model="claude-opus-4-8",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=self.conversation_history,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                print(text, end="", flush=True)
        print()

        self.conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    def load_recipe(self, recipe: dict) -> str:
        self.current_recipe = recipe
        self.current_step = 0
        name = recipe.get("name", "this recipe")
        steps = recipe.get("steps", [])
        intro = f"I've loaded {name}. It has {len(steps)} steps. Ready to start?"
        return self.chat(
            f"The user selected: {name}. Here's the recipe: {recipe}. "
            f"Greet them warmly and give a brief 1-sentence intro, then ask if they're ready."
        )

    def get_current_step(self) -> Optional[dict]:
        if not self.current_recipe:
            return None
        steps = self.current_recipe.get("steps", [])
        if self.current_step < len(steps):
            return steps[self.current_step]
        return None

    def next_step(self, image_data: Optional[bytes] = None) -> str:
        if not self.current_recipe:
            return self.chat("No recipe loaded yet.", image_data)
        steps = self.current_recipe.get("steps", [])
        self.current_step += 1
        if self.current_step >= len(steps):
            self.current_step = len(steps)
            return self.chat(
                "The user has completed all the steps. Congratulate them warmly!",
                image_data,
            )
        step = steps[self.current_step]
        return self.chat(
            f"Moving to step {self.current_step + 1}: {step}. Guide the user through it.",
            image_data,
        )

    def reset(self):
        self.conversation_history = []
        self.current_recipe = None
        self.current_step = 0
