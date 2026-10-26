# Bedrock AgentCore Support Chatbot

A customer-support chatbot for an online shop, built on **Amazon Bedrock
AgentCore** (managed harness + Gateway) with **Amazon Nova Pro**. It classifies
every incoming message into one of three paths, answers shop questions strictly
from a provided FAQ, files bug tickets into DynamoDB through a Gateway-invoked
Lambda tool, and hands anything else off to a human support line.

Built for the Udacity AWS AI Engineer project; evaluated with Bedrock
Evaluations (LLM-as-a-judge). **Submitted and passed** — Udacity Project
Completion badge awarded.

<p align="center">
  <img src="screenshots/udacity-completion-badge.png" alt="Udacity Project Completion badge — Customer Support Chatbot with Amazon Bedrock" width="380">
</p>

## What it does

Each customer message is classified into exactly one category, and the response
rules differ per category:

| Category | Trigger | Behavior |
|---|---|---|
| **Bug report** | Something broken, crashing, not loading | Collects `description`, `stepsToReproduce`, `environment` — then calls the `create_bug_report` tool and returns a ticket ID |
| **Platform question** | Orders, shipping, returns, payments, products, account, privacy | Answers **only** from `online_shop_faq.md`; no outside knowledge |
| **Other** | Complaints, manager requests, off-topic | Redirects to the human support phone line |

The bug-report path is deliberately strict: the model may not infer a field the
customer never stated. Two of three fields is not enough — it asks follow-up
questions until all three are explicitly provided, and refuses to file early
even when the customer insists. That rule lives in `system_prompt.txt`.

## Architecture

```
  customer (chat.py)
        |
        v
  AgentCore Harness  ──  system_prompt.txt + online_shop_faq.md
  (Nova Pro, temp 0)     stateful per runtimeSessionId
        |
        | tool call: bugreports___create_bug_report
        v
  AgentCore Gateway (MCP, AWS_IAM auth)
        |
        v
  Lambda: create_bug_report  ──>  DynamoDB (ticket record)
```

Routing lives entirely in the system prompt — the managed harness has no
separate condition or classifier node. Conversation state is held by the
harness: reusing one `runtimeSessionId` is what lets it collect bug details
across several turns.

Model is pinned to `us.amazon.nova-pro-v1:0` with greedy decoding
(temperature 0, topK 1), AWS's recommendation for reliable Nova tool calling.
Region: `us-east-1`.

## Repository layout

| File | Purpose |
|---|---|
| `system_prompt.txt` | Classification + routing rules; `{{FAQ}}` placeholder gets the FAQ injected |
| `online_shop_faq.md` | The knowledge base the bot is allowed to answer from |
| `cloudformation-tool.yaml` | DynamoDB table, Lambda, Gateway role, harness execution role |
| `setup_gateway.py` | Creates the Gateway and registers the Lambda as a tool target |
| `create_harness.py` | Creates/updates the harness, injects the FAQ, waits for READY |
| `chat.py` | Terminal chat client; one conversation per run |
| `create_bug_report.py` | Standalone copy of the Lambda handler (mirrors the CFN-embedded code) |
| `cloudformation-testing.yaml` | S3 bucket + IAM role for Bedrock Evaluations |
| `harness-tests.json` | 8 test cases across all three categories |
| `output_eval_dataset.jsonl` | Per-example evaluation output |
| `screenshots/` | Console evidence: gateway, harness, DynamoDB, eval scores |

## Setup

Requires AWS credentials for `us-east-1` and `boto3`.

```bash
# 1. Deploy the tool stack (DynamoDB + Lambda + IAM roles)
aws cloudformation deploy \
  --template-file cloudformation-tool.yaml \
  --stack-name support-chatbot-tool \
  --capabilities CAPABILITY_IAM

# 2. Create the Gateway and register the Lambda tool (run once)
python setup_gateway.py

# 3. Create the harness (injects the FAQ into the system prompt)
python create_harness.py

# 4. Chat
python chat.py
```

Steps 2 and 3 write their ARNs to `agentcore_config.json`, so nothing needs to
be copy-pasted between scripts. Iterating on behavior is: edit
`system_prompt.txt`, re-run `create_harness.py`.

Gateway target names may only contain letters, digits, and underscores — a dash
breaks Nova tool calling with `Model produced invalid sequence as part of
ToolUse`.

## Evaluation

Bedrock Evaluations, LLM-as-a-judge, Nova Pro as evaluator, over the 8-case
suite in `harness-tests.json`:

| Metric | Score | Reading |
|---|---|---|
| Faithfulness | **1.00** | No response fabricated information outside the FAQ or tool results |
| Correctness | 0.75 | |
| Helpfulness | 0.665 | |
| Refusal | 0.25 | Appropriately low — redirected only on genuinely out-of-scope requests |

Faithfulness at 1.00 is the headline result: the FAQ-grounding constraint held
across every case. Per-example breakdown is in `output_eval_dataset.jsonl`;
console screenshots are in `screenshots/pics/`.

## Note on project architecture

This submission uses the **AgentCore managed harness** rather than **Bedrock
Flows**, per the course's own updated Project Instructions and Testing and
Evaluation pages.

Bedrock Agents Classic closed to new customers on July 30, 2026. Because Bedrock
Flows' "Agent node" depends on Agents Classic, the course replaced this
project's architecture with the AgentCore harness + Gateway stack. The rubric
text on the submission page still references the older Flows terminology (flow
diagram, Condition node, `flow-tests.json`), which no longer applies.

Equivalent evidence for this architecture:

- **Routing logic** (classification + path selection): implemented in
  `system_prompt.txt`, since the harness has no separate condition or classifier
  node — matching the Project Instructions: *"there are no condition nodes or
  classifiers, just instructions."*
- **Test suite**: `harness-tests.json`, the current filename per the Testing and
  Evaluation guide (renamed from `flow-tests-template.json`).
- **All other requirements** — Gateway-invoked Lambda tool, DynamoDB
  persistence, FAQ-grounded answers, human hand-off, automated evaluation with
  Bedrock Evaluations — are implemented and verified in the screenshots and
  output files.

This architecture was accepted and the project passed review.
