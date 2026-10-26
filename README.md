# Future AWS Agent Engineer

Everything I built in the Udacity **Future AWS Agent Engineer** program: four
AI agent projects on **Amazon Bedrock AgentCore**. They start with a single
travel agent and finish with a production-style multi-agent system. Each project
was deployed and tested on real AWS infrastructure.

| # | Project | What it is | Result |
|---|---|---|---|
| 00 | [WanderBot Travel Agent](00-wanderbot-travel-agent/) | Travel concierge agent: custom tools, local RAG, Gateway + Lambda over MCP, short and long-term memory | Deployed to AgentCore Runtime |
| 01 | [Bedrock AgentCore Support Chatbot](01-bedrock-agentcore-support-chatbot/) | Online-shop support bot that classifies messages, answers only from an FAQ, and files bug tickets to DynamoDB | **Passed**: Udacity completion badge, Faithfulness 1.00 |
| 02 | [Customer Support Agent with AgentCore](02-customer-support-agent-agentcore/) | Amazon-store support agent with Gateway tools, Knowledge Base RAG, memory, Code Interpreter, and Browser | All 6 test scenarios passing |
| 03 | [NovaMart Multi-Agent Capstone](03-novamart-multi-agent-capstone/project/starter/) | Five-agent orchestrator/worker system with parallel multi-agent RAG, guardrails, shared DynamoDB state, and X-Ray | **120/120** on the automated test suite |

**Stack:** Amazon Bedrock AgentCore (Runtime, Gateway, Memory, Code
Interpreter, Browser) · Strands Agents SDK · Amazon Nova and Claude models ·
Bedrock Knowledge Bases · Bedrock Guardrails · Lambda · API Gateway · DynamoDB ·
S3 / S3 Vectors · CloudFormation / CDK · CloudWatch · X-Ray · MCP · Python

---

## 00 · WanderBot Travel Agent

Horizon Travel's concierge agent, built module by module on the Strands Agents
SDK and deployed to AgentCore Runtime.

- **Custom tools:** `search_flights`, `search_hotels`, `get_exchange_rate`,
  and `calculate_trip_cost`, which runs real arithmetic so the model never does
  money math in its head.
- **RAG:** `search_knowledge_base` uses Bedrock embeddings and cosine similarity
  over the destination guides and travel policies.
- **Gateway + Lambda over MCP:** booking and loyalty-points Lambdas sit behind
  AgentCore Gateway. The agent discovers those tools at runtime and has no
  hard-coded booking code.
- **Memory:** a short-term hook stores every turn in AgentCore Memory, and a
  traveler-preference hook keeps preferences across sessions.

**What I worked through:** the default container deploy failed in the lab
account because CodeBuild compute was blocked. I switched to
`direct_code_deploy` and then fixed a toolkit bug (`runtime_type` unset), and
after that the deploy went through. In the cloud the agent returned correct
totals ($1,049 and $270), and CloudWatch logs confirmed the calculator tool was
called on the server side.

## 01 · Bedrock AgentCore Support Chatbot

Customer-support chatbot built on the AgentCore managed harness with Amazon
Nova Pro. Every message gets routed down one of three paths:

- **Bug report:** collects description, steps to reproduce, and environment,
  then calls a Gateway → Lambda tool that writes a ticket to DynamoDB. It refuses
  to file a ticket until all three fields have actually been given.
- **Platform question:** answers **only** from the shop FAQ.
- **Other:** hands the customer off to the human support line.

The infrastructure is defined in CloudFormation. I evaluated the bot with Bedrock
Evaluations (LLM-as-a-judge) on an 8-case test suite: **Faithfulness 1.00**,
Correctness 0.75. Submitted, passed, and awarded the Udacity Project Completion
badge. → [Full write-up](01-bedrock-agentcore-support-chatbot/README.md)

## 02 · Customer Support Agent with AgentCore

Production-grade support agent for a fictional Amazon store, deployed to
AgentCore Runtime.

- **Gateway (MCP):** Lambda tools for order tracking and refunds, served through
  an API Gateway OpenAPI target.
- **RAG:** a Bedrock Knowledge Base over the product catalog and policies.
- **Memory:** short-term memory within a session and long-term memory across
  sessions. The agent recalled "Jane, prefers concise responses" in a brand-new
  session.
- **Code Interpreter:** computes loyalty discounts deterministically (Gold, 4,250
  pts, $150 order → $99.00, 250 pts left).
- **Browser:** fetches live web pages.

**What I worked through:** a Gateway target failed because the CLI-built REST
API had no `200` method responses. The runtime role also lacked permissions, so
I added least-privilege policies for KB `Retrieve` and the Browser tool.
Screenshots of every test run are in
[`starter/screenshot/`](02-customer-support-agent-agentcore/starter/screenshot/).
→ [Reflection](02-customer-support-agent-agentcore/starter/REFLECTION.md)

## 03 · NovaMart Multi-Agent Capstone

Enterprise multi-agent customer support system, and the final project of the
program.

```
OrchestratorAgent (Claude Haiku 4.5)
 ├─ InventoryAgent       → DynamoDB orders / customer tier
 ├─ PolicyAgent          → 3 retriever sub-agents run in PARALLEL
 │                          (Returns · Shipping · Warranty Knowledge Bases)
 ├─ RefundAgent          → eligibility decision + refund
 └─ CommunicationAgent   → writes the final customer reply
Shared state: DynamoDB WorkflowStateTable with optimistic locking
```

- I implemented all five Strands agents and the orchestrator's routing rules in
  [`src/agent_orchestrator.py`](03-novamart-multi-agent-capstone/project/starter/src/agent_orchestrator.py).
- **Multi-agent RAG:** a ThreadPoolExecutor fans out to three Knowledge Bases
  backed by S3 Vectors, then merges and deduplicates the results.
- **Bedrock Guardrails** cover harmful content, PII, and off-topic requests.
- The system is deployed to AgentCore Runtime through the AgentCore CLI and CDK.
- **Observability:** CloudWatch logs and an X-Ray service map cover the
  orchestrator → worker → KB call chain.
- **Score: 120/120** on the automated test suite. Evidence is in
  [`screenshots/`](03-novamart-multi-agent-capstone/project/starter/screenshots/),
  and the architecture diagrams are in
  [`diagrams/`](03-novamart-multi-agent-capstone/project/starter/diagrams/).

The `lesson-01` through `lesson-11` folders next to the project hold the
course's exercises for that module (multi-agent architecture, routing, sagas,
shared state, RAG, guardrails, deployment, and Gateway communication). They are
Udacity course material, licensed under that folder's
[LICENSE](03-novamart-multi-agent-capstone/LICENSE.md).

---

## Running a project

Each project folder has its own README with setup steps. All of them need AWS
credentials with Bedrock access. Never commit `.env` files or keys; this repo's
`.gitignore` excludes them. Delete the cloud resources when you are done
(`agentcore destroy`, `aws cloudformation delete-stack`, or the capstone's
`infrastructure/cleanup.py`) so they stop costing money.

**Author:** Abubakar Diallo ([@Dialloni](https://github.com/Dialloni))
