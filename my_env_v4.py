import os
from dotenv import load_dotenv

# Inject .env variables before inference_given.py initializes OpenAI
load_dotenv()
if not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
    if os.getenv("GROQ_API_KEY"):
        os.environ["API_KEY"] = os.getenv("GROQ_API_KEY")
        # Route the HuggingFace URL to Groq URL so you're using your free key!
        if not os.getenv("API_BASE_URL"):
            os.environ["API_BASE_URL"] = "https://api.groq.com/openai/v1"
            os.environ["MODEL_NAME"] = "llama-3.3-70b-versatile"

from pydantic import BaseModel
from typing import Optional
import httpx
import json

class MyEnvV4Action(BaseModel):
    message: str

class Observation(BaseModel):
    echoed_message: str

class Result(BaseModel):
    observation: Observation
    reward: Optional[float] = 0.0
    done: bool = False

class MyEnvV4Env:
    def __init__(self):
        self.base_url = "http://localhost:7860"

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str]) -> "MyEnvV4Env":
        # Mocking the Docker spinup and binding directly to the local FastAPI
        return cls()

    async def reset(self) -> Result:
        try:
            r = httpx.post(f"{self.base_url}/reset", json={"task": "easy"}, timeout=30)
            r.raise_for_status()
            data = r.json()
            obs_str = json.dumps(data.get("observation", {}))
            return Result(
                observation=Observation(echoed_message=f"Environment initialized. System State: {obs_str[:200]}..."),
                reward=0.0,
                done=False
            )
        except Exception as e:
            return Result(
                observation=Observation(echoed_message=f"Error connecting to backend: {e}"),
                reward=0.0,
                done=True
            )

    async def step(self, action: MyEnvV4Action) -> Result:
        text = action.message
        
        # We need to extract an action (0-17) out of the LLMs response string
        # If the LLM just types English sentences because of the echo prompt,
        # we will mathematically map the length of its sentence to a random action 0-17.
        action_id = 17
        ints = [int(s) for s in text.split() if s.isdigit()]
        if ints and 0 <= ints[0] <= 17:
            action_id = ints[0]
        else:
            action_id = len(text) % 18
            
        try:
            r = httpx.post(f"{self.base_url}/step", json={"action": action_id}, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            
            # Show the actual LLM what its action did to the backend
            echoed = (f"Simulator processed action [{action_id}]. "
                      f"Reward Feedback: {reward:.4f}. "
                      f"Did you finish? {done}.")

            return Result(
                observation=Observation(echoed_message=echoed), 
                reward=reward, 
                done=done
            )
        except Exception as e:
             return Result(
                observation=Observation(echoed_message=f"Error: {e}"), 
                reward=0.0, 
                done=True
            )

    async def close(self) -> None:
        # FastAPI handles its own lifecycle
        pass
