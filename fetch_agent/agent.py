"""
AegisCoord Fetch.ai uAgent — ACP-compliant chat interface for the
disaster relief multi-agent simulation.

Run:  pip install -r fetch_agent/requirements.txt && python fetch_agent/agent.py
"""

import os
import sys
from datetime import datetime
from uuid import uuid4

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    StartSessionContent,
    EndSessionContent,
    chat_protocol_spec,
)

from fetch_agent.scenario_parser import parse_scenario
from fetch_agent.sim_adapter import run_scenario
from fetch_agent.formatter import format_results


# ── agent init ───────────────────────────────────────────────────────────

AGENT_SEED = os.getenv("AEGISCOORD_AGENT_SEED", "aegiscoord-disaster-relief-agent-seed-v1")
AGENTVERSE_KEY = os.getenv("AGENTVERSE_KEY") or os.getenv("AGENTVERSE_API_KEY")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT")

agent_kwargs = {
    "name": "aegiscoord",
    "seed": AGENT_SEED,
    "port": int(os.getenv("AGENT_PORT", "8001")),
    "report_events": False,
}

if AGENT_ENDPOINT:
    agent_kwargs["endpoint"] = [AGENT_ENDPOINT]
else:
    agent_kwargs["mailbox"] = f"{AGENTVERSE_KEY}@https://agentverse.ai" if AGENTVERSE_KEY else True

agent = Agent(**agent_kwargs)


import asyncio
import re

# ── ACP chat protocol ───────────────────────────────────────────────────

chat_proto = Protocol(spec=chat_protocol_spec)


@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """
    Main handler: parse scenario → run simulation → reply with results.
    Always acknowledge first, then reply.
    """
    # 1. Acknowledge immediately with proper field
    try:
        await ctx.send(
            sender,
            ChatAcknowledgement(
                acknowledged_msg_id=msg.msg_id,
            ),
        )
        ctx.logger.info(f"Sent ACK for {msg.msg_id} to {sender}")
    except Exception as exc:
        ctx.logger.error(f"Failed to send acknowledgement: {exc}")

    # 2. Extract and clean user text (strip @mention)
    raw_text = msg.text()
    user_text = re.sub(r"@agent1[a-z0-9]+\s*", "", raw_text, flags=re.IGNORECASE).strip()

    ctx.logger.info(f"Query from {sender}: {user_text[:120]}")

    # Greeting / Help fast-path (instant reply without heavy simulation)
    if not user_text or re.match(r"^(hi+|hello+|hey+|help|start|info|who are you)[\s!.]*$", user_text.lower()):
        welcome = (
            "Hello! I am **AegisCoord**, an AI multi-agent disaster relief coordinator.\n\n"
            "I simulate and compare emergency rescue strategies:\n"
            "- **Greedy heuristic**: Nearest available agent dispatch\n"
            "- **Coordinated**: Hungarian optimal bipartite assignment\n"
            "- **LLM-assisted**: Dynamic AI triage and coordination\n\n"
            "**Try sending a scenario like:**\n"
            "> *3 victims at (2,3) critical, (5,5) serious, (7,1) stable with 2 rescue teams. Compare strategies.*"
        )
        await _reply(ctx, sender, welcome)
        return

    # 3. Parse scenario (ASI:One LLM → regex → defaults)
    try:
        params = parse_scenario(user_text)
    except Exception as exc:
        ctx.logger.error(f"Parse error: {exc}")
        params = {
            "num_agents": 3,
            "num_victims": 6,
            "modes": ["greedy", "coordinated", "llm"],
            "steps": 15,
            "seed": 42,
        }

    ctx.logger.info(f"Parsed params: {params}")

    # 4. Run simulation in background thread so event loop is not blocked
    try:
        sim_steps = min(params.get("steps", 15), 20)
        results = await asyncio.to_thread(
            run_scenario,
            num_agents=params.get("num_agents", 3),
            num_victims=params.get("num_victims", 6),
            modes=params.get("modes", ["greedy", "coordinated", "llm"]),
            steps=sim_steps,
            seed=params.get("seed", 42),
        )
    except Exception as exc:
        ctx.logger.error(f"Simulation error: {exc}")
        await _reply(ctx, sender, f"Simulation failed: {exc}")
        return

    # 5. Format and reply
    try:
        reply_text = format_results(results)
    except Exception as exc:
        ctx.logger.error(f"Formatter error: {exc}")
        reply_text = f"Simulation ran but formatting failed: {exc}"

    await _reply(ctx, sender, reply_text)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"ACK for msg {msg.acknowledged_msg_id} from {sender}")


# ── helpers ──────────────────────────────────────────────────────────────

async def _reply(ctx: Context, recipient: str, text: str):
    """Send a ChatMessage reply with text content."""
    try:
        await ctx.send(
            recipient,
            ChatMessage(
                content=[
                    TextContent(text=text),
                ],
            ),
        )
    except Exception as exc:
        ctx.logger.error(f"Reply send failed: {exc}")


# ── register protocol ───────────────────────────────────────────────────

agent.include(chat_proto, publish_manifest=True)


# ── startup logging ─────────────────────────────────────────────────────

@agent.on_event("startup")
async def on_startup(ctx: Context):
    ctx.logger.info(f"AegisCoord agent started")
    ctx.logger.info(f"Address: {agent.address}")
    ctx.logger.info(f"Mailbox: enabled")
    ctx.logger.info("Waiting for chat messages via ACP...")


# ── entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if AGENTVERSE_KEY:
        try:
            from uagents_core.utils.registration import (
                register_chat_agent,
                RegistrationRequestCredentials,
            )
            endpoint_url = AGENT_ENDPOINT or "https://agentverse.ai/v2/agents/mailbox/submit"
            register_chat_agent(
                name="aegiscoord",
                endpoint=endpoint_url,
                active=True,
                credentials=RegistrationRequestCredentials(
                    agentverse_api_key=AGENTVERSE_KEY,
                    agent_seed_phrase=AGENT_SEED,
                ),
            )
            print("Successfully registered and activated on Agentverse.")
        except Exception as e:
            print(f"Agentverse registration warning: {e}")

    agent.run()
