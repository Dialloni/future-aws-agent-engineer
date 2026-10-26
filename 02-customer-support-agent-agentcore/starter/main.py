"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-rkjdxsma5f.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "1WPFY1ARI8"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-Nqj7Vb4AVY"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id, region_name=REGION)

memory_client = MemoryClient(region_name=REGION)

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)

SYSTEM_PROMPT = """You are a friendly, accurate customer support assistant for an online Amazon store.
The current customer's ID is {customer_id}.

Use your tools instead of guessing:
- Order status, tracking, customer profile or order history: use the order-tracker tools
  (order-tracker___get_order, order-tracker___get_customer_orders, order-tracker___get_customer).
- Refunds, refund status and return labels: use the refund-processor tools
  (refund-processor___initiate_refund, refund-processor___check_refund_status,
  refund-processor___get_return_label). Look up the order first so you know the refund amount.
- Product details, return policies, warranties, loyalty tiers and benefits: use search_knowledge_base.
- Loyalty discount maths: always use calculate_loyalty_discount, never compute it yourself.
- Live web pages: use the browser tool (init_session, then navigate, then get_text or evaluate).

If a message starts with "Customer Context:", those are facts and preferences remembered from
earlier sessions with this customer. Use them to personalise your answer (e.g. their name and
preferred response style). Report tool results faithfully, including IDs, tracking numbers,
carriers, dates and amounts. Never invent data a tool did not return."""


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Return a dict: { strategy["type"]: strategy["namespaces"][0] for each strategy }
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}
    for strategy in strategies:
        # Newer API responses use "namespaceTemplates"; older ones use "namespaces".
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []
        if templates:
            namespaces[strategy["type"]] = templates[0]
    return namespaces


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_id = memory_id
        self.memory_client = memory_client
        self.namespaces = get_namespaces(memory_client, memory_id)
        # Prefix added to the user message, so it can be stripped before saving.
        self.context_prefix = ""

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return
        message = messages[-1]
        content = message.get("content", [])
        # Only plain-text user messages; tool results are also sent with role "user".
        if message.get("role") != "user" or not content:
            return
        if any("toolResult" in block for block in content) or "text" not in content[0]:
            return

        user_query = content[0]["text"]
        try:
            context_lines = []
            for strategy_type, template in self.namespaces.items():
                namespace = template.replace("{actorId}", self.actor_id)
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
                for memory in memories:
                    text = memory.get("content", {}).get("text", "").strip()
                    if text:
                        context_lines.append(f"[{strategy_type}] {text}")

            if context_lines:
                self.context_prefix = "Customer Context:\n" + "\n".join(context_lines) + "\n\n"
                content[0]["text"] = self.context_prefix + user_query
                logger.info("Retrieved %d memories for %s", len(context_lines), self.actor_id)
        except Exception as e:
            logger.error("Memory retrieval failed: %s", e)

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages
        customer_query = agent_response = None
        for message in reversed(messages):
            content = message.get("content", [])
            texts = [block["text"] for block in content if "text" in block]
            if not texts:
                continue
            if message["role"] == "assistant" and agent_response is None:
                agent_response = "\n".join(texts)
            elif (
                message["role"] == "user"
                and agent_response is not None
                and not any("toolResult" in block for block in content)
            ):
                customer_query = texts[0].removeprefix(self.context_prefix)
                break

        if not customer_query or not agent_response:
            return
        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
            )
            logger.info("Saved interaction for %s", self.actor_id)
        except Exception as e:
            logger.error("Memory save failed: %s", e)

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base not configured. Set KB_ID in main.py to enable product and policy search."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        logger.error("Knowledge base retrieval failed: %s", e)
        return f"Knowledge base search failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return f"No knowledge base results found for: {query}"

    chunks = [r["content"]["text"] for r in results if r.get("content", {}).get("text")]
    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    tier = tier.strip().capitalize()
    product_category = product_category.strip().lower()

    code = f"""
import json, math

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {int(loyalty_points)}
tier = {tier!r}
order_total = round({float(order_total)}, 2)
product_category = {product_category!r}

# Points: 100 points = $1, minimum 500, redeemed in blocks of 500,
# and the points discount may cover at most 50% of the order.
max_points_by_cap = int(order_total * 0.5 * 100)
points_redeemed = min(loyalty_points, max_points_by_cap) // 500 * 500
if points_redeemed < 500:
    points_redeemed = 0
points_discount = points_redeemed / 100

# Tier discount applies to the subtotal after points.
subtotal_after_points = order_total - points_discount
tier_discount_pct = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal_after_points * tier_discount_pct, 2)

final_total = round(subtotal_after_points - tier_discount, 2)
total_savings = round(order_total - final_total, 2)
points_earned = math.floor(final_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed

print(json.dumps({{
    "tier": tier,
    "order_total": order_total,
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier_discount_pct": int(tier_discount_pct * 100),
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
    "calculated_by": "AgentCore Code Interpreter",
}}))
"""

    try:
        with code_session(REGION) as client:
            response = client.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": True,
            })
            for event in response["stream"]:
                if "result" in event:
                    result = event["result"]
                    stdout = result.get("structuredContent", {}).get("stdout") or \
                        result["content"][0]["text"]
                    return json.dumps(json.loads(stdout))
        raise RuntimeError("Code Interpreter returned no result")

    except Exception as e:
        # Fallback: tier discount only, no points redemption.
        logger.error("Code Interpreter unavailable, using fallback: %s", e)
        tier_discount_pct = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}.get(tier, 0.0)
        tier_discount = round(order_total * tier_discount_pct, 2)
        final_total = round(order_total - tier_discount, 2)
        return json.dumps({
            "tier": tier,
            "order_total": round(order_total, 2),
            "points_redeemed": 0,
            "tier_discount_pct": int(tier_discount_pct * 100),
            "tier_discount": tier_discount,
            "final_total": final_total,
            "remaining_points": loyalty_points,
            "calculated_by": "fallback (tier discount only)",
            "note": f"Code Interpreter unavailable: {e}",
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    user_input = (payload or {}).get("prompt", "").strip()
    if not user_input:
        return "Please provide a 'prompt' in the request payload."
    actor_id = payload.get("customer_id") or "anonymous-customer"
    session_id = payload.get("session_id") or str(uuid.uuid4())

    try:
        memory_hook = MemoryHook(actor_id, session_id, memory_client, MEMORY_ID)
        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with mcp_client:
            gateway_tools, token = [], None
            while True:
                page = mcp_client.list_tools_sync(pagination_token=token)
                gateway_tools.extend(page)
                token = page.pagination_token
                if not token:
                    break
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT.format(customer_id=actor_id),
            )
            response = await agent.invoke_async(user_input)

        for block in response.message.get("content", []):
            if "text" in block:
                return block["text"]
        return str(response)

    except Exception as e:
        logger.exception("Agent invocation failed")
        return f"Sorry, I ran into a problem handling your request: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
