"""WanderBot - AI travel concierge for Horizon Travel.

Runs on Amazon Bedrock AgentCore Runtime, built with the Strands SDK.
"""

# The runtime container. Turns this file into an HTTP service that
# AgentCore Runtime knows how to call. We never write routes ourselves.
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# The agent loop: takes a message, decides whether to call a tool,
# calls the model, repeats until it has a final answer.
from strands import Agent

# Adapter that points a Strands Agent at a model hosted on Amazon Bedrock.
from strands.models import BedrockModel

# A built-in Strands tool. It is an ordinary Python function whose
# docstring Strands converts into a tool schema for the model to read.
from strands_tools import calculator

# Instantiate the container. No arguments. This object is what the
# @app.entrypoint decorator below registers against.
app = BedrockAgentCoreApp()

# Cross-region inference profile, NOT the bare "amazon.nova-2-lite-v1:0".
# The "us." prefix lets Bedrock route the request to whichever US region
# has capacity. Verified present in this account in us-west-2.
MODEL_ID = "us.amazon.nova-2-lite-v1:0"

# Config object only - constructing this makes no network call. A typo in
# MODEL_ID therefore fails at first invoke, not at import time.
model = BedrockModel(model_id=MODEL_ID)

# Sent to the model on EVERY turn, so every token here is billed every
# time. Kept short on purpose. The calculator rule is phrased as an
# absolute because a soft phrasing ("when asked to calculate") lets the
# model do arithmetic in its head, which it does badly.
SYSTEM_PROMPT = """You are WanderBot, the AI travel assistant for Horizon Travel.

You help travellers with baggage allowances, check-in, cabin classes, and
cancellation policy.

Never do arithmetic yourself. For any total, fee, tip, percentage, currency
estimate, or duration, call the calculator tool and report its result.

Be friendly, concise, and stay on travel topics."""


@app.entrypoint
async def invoke(payload: dict, context=None):
    """Called by AgentCore Runtime once per incoming request.

    payload: the JSON body of the request
    context: session id, request id, and other runtime metadata
    """
    # Read the user's text. Default keeps a malformed request from
    # crashing the handler - this is our trust boundary.
    user_message = payload.get("message", "Hello!")

    # Built per request on purpose. An Agent carries conversation history,
    # so a shared module-level Agent would leak one traveller's context
    # into the next one's reply. Stateless is correct here.
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[calculator],
    )

    # invoke_async, not agent(...). Calling the agent directly is a
    # blocking call; inside an async function it stalls the event loop
    # and the server stops serving other requests while it waits.
    result = await agent.invoke_async(user_message)

    # result is an AgentResult object, not a string. AgentCore has to
    # serialize the return value to JSON, so hand it plain text.
    return {"result": str(result)}


if __name__ == "__main__":
    # Starts the local HTTP server. This is the branch `agentcore dev` uses.
    app.run()
