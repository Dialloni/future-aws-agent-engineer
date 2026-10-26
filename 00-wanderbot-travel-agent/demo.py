from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator  # built-in Strands tool

app = BedrockAgentCoreApp()

MODEL_ID = "us.amazon.nova-2-lite-v1:0"
model = BedrockModel(model_id=MODEL_ID)

SYSTEM_PROMPT = """You are WanderBot, the AI travel assistant for Horizon Travel.
When asked to calculate costs, tips, totals, durations, or percentages,
use the calculator tool. Keep answers friendly, concise, travel-focused."""


@app.entrypoint
async def invoke(payload: dict, context=None):
    user_message = payload.get("message", "Hello!")
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[calculator],
    )
    return agent(user_message)


if __name__ == "__main__":
    app.run()