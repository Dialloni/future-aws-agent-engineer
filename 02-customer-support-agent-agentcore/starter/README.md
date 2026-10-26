# Customer Support Agent on Amazon Bedrock AgentCore

An AI customer support assistant for a fictional online store. It tracks orders, processes refunds, answers product and policy questions, remembers customers across sessions, calculates loyalty discounts, and reads live web pages.

Built with [Strands Agents](https://strandsagents.com) and deployed to Amazon Bedrock AgentCore Runtime. All agent code is in [`main.py`](main.py).

## Capabilities

| Capability | How it works |
| --- | --- |
| Order tracking | `order-tracker` Lambda behind an API Gateway REST API (`get_order`, `get_customer_orders`, `get_customer`), exposed as MCP tools by AgentCore Gateway |
| Refunds and return labels | `refund-processor` Lambda added to the same Gateway as a Lambda target, using [`lambda/lambda_schema`](lambda/lambda_schema) |
| Product and policy answers (RAG) | `search_knowledge_base` tool calls the Bedrock Knowledge Base Retrieve API over [`product_catalog.txt`](product_catalog.txt) |
| Cross-session memory | `MemoryHook` loads saved facts and preferences before each message and saves each finished turn to AgentCore Memory |
| Loyalty discount | `calculate_loyalty_discount` runs the pricing rules in the AgentCore Code Interpreter sandbox, with a tier-only fallback |
| Web browsing | `AgentCoreBrowser` tool reads live pages |

Model: Amazon Nova 2 Lite (`global.amazon.nova-2-lite-v1:0`), region `us-east-1`.

## AWS resources

| Resource | Name |
| --- | --- |
| Lambda functions | `order-tracker`, `refund-processor` (Python 3.12) |
| REST API | `CustomerSupportAPI`, stage `prod` |
| AgentCore Gateway | `CustomerSupportGateway` (NONE authorizer, targets `order-tracker` and `refund-processor`) |
| Knowledge Base | `CustomerSupportKB` (Titan Embeddings v2) |
| AgentCore Memory | `CustomerSupportMemory`: `customer_facts` → `cs_agent/{actorId}/facts`, `customer_preferences` → `cs_agent/{actorId}/preferences` |
| Runtime role policies | [`kb-retrieve-policy.json`](kb-retrieve-policy.json), [`browser-tool-policy.json`](browser-tool-policy.json) |

The Udacity sandbox blocks OpenSearch Serverless, so the Knowledge Base uses Bedrock's managed vector store instead.

## Run and deploy

```bash
uv sync
aws configure                     # region us-east-1
# Set GATEWAY_URL, KB_ID, REGION and MEMORY_ID at the top of main.py
uv run agentcore configure --entrypoint main.py --name customer_support_agent
uv run agentcore deploy
uv run agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```

## Test results

All six scenarios passed against the deployed agent. Screenshots are in [`screenshot/`](screenshot/).

| Test | Result | Evidence |
| --- | --- | --- |
| 1. Order tracking | SHIPPED, UPS, TRK987654321, estimated delivery date | [test1_order_tracking.png](screenshot/test1_order_tracking.png) |
| 2. Refund | Refund ID, Approved, $139.99, 3-5 business days | [test2_refund_processing.png](screenshot/test2_refund_processing.png) |
| 3. Knowledge Base | Platinum: free same-day shipping, 15% discount, priority support | [test3_knowledge_base_rag.png](screenshot/test3_knowledge_base_rag.png) |
| 4. Memory | Session `s-B` recalls the name "Jane" and the concise-response preference from session `s-A` | [test4_memory_sessions_A_and_B.png](screenshot/test4_memory_sessions_A_and_B.png) |
| 5. Loyalty discount | Gold, 4,250 points, $150: 4,000 points redeemed, 10% tier discount, $99.00 final, 250 points left | [test5_loyalty_discount.png](screenshot/test5_loyalty_discount.png) |
| 6. Browser | Page title "Learn the Latest Tech Skills; Advance Your Career \| Udacity" | [test6_browser_tool.png](screenshot/test6_browser_tool.png) |

Design choices, challenges and production considerations are in [`REFLECTION.md`](REFLECTION.md).

## Cleanup

`uv run agentcore destroy`, then delete the Gateway, Memory, Knowledge Base, both S3 buckets, the REST API and both Lambda functions. The Gateway has no authorizer, so delete it first.
