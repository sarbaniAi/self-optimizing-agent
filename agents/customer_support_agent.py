"""
Customer Support Agent -- TechStore
RAG + tool-calling agent with search_knowledge_base, lookup_order,
process_return, and escalate_to_human capabilities.
"""
import json
import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the TechStore knowledge base for product info, policies, "
                "FAQs, and troubleshooting guides. Use this before answering any "
                "factual question about TechStore."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up a customer order by order ID. Returns order status, "
                "items, shipping info, and payment details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID (e.g. ORD-123456).",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_return",
            "description": (
                "Initiate a return or refund for a given order. Requires the "
                "order ID and a reason. This action needs human approval before "
                "it is finalized."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to process a return for.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Customer-stated reason for the return.",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Escalate the conversation to a human support agent. Use this "
                "when the customer issue cannot be resolved automatically or "
                "when the customer explicitly requests a human."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the issue for the human agent.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Escalation priority level.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Stub tool implementations (replace with real backends)
# ---------------------------------------------------------------------------
def _search_knowledge_base(query: str) -> str:
    """Stub: replace with a real vector-search retriever call."""
    return json.dumps(
        {
            "results": [
                {
                    "title": "Return Policy",
                    "content": (
                        "TechStore allows returns within 30 days of purchase "
                        "for most items. Electronics with opened packaging may "
                        "be subject to a 15% restocking fee. Defective items "
                        "can be returned at any time within the warranty period."
                    ),
                },
                {
                    "title": "Shipping Info",
                    "content": (
                        "Standard shipping takes 5-7 business days. Express "
                        "shipping (2-3 business days) is available for an "
                        "additional fee. Free shipping on orders over $50."
                    ),
                },
            ]
        }
    )


def _lookup_order(order_id: str) -> str:
    """Stub: replace with a real order-lookup call."""
    return json.dumps(
        {
            "order_id": order_id,
            "status": "delivered",
            "items": [
                {"name": "Wireless Mouse", "qty": 1, "price": 29.99},
                {"name": "USB-C Hub", "qty": 1, "price": 49.99},
            ],
            "shipping": {"method": "standard", "delivered_on": "2026-05-05"},
            "total": 79.98,
        }
    )


def _process_return(order_id: str, reason: str) -> str:
    """Stub: replace with a real return-processing call."""
    return json.dumps(
        {
            "return_id": f"RET-{order_id.replace('ORD-', '')}",
            "order_id": order_id,
            "status": "pending_approval",
            "reason": reason,
            "message": "Return request submitted. Awaiting human approval.",
        }
    )


def _escalate_to_human(summary: str, priority: str = "medium") -> str:
    """Stub: replace with a real escalation call."""
    return json.dumps(
        {
            "ticket_id": "ESC-78901",
            "status": "created",
            "priority": priority,
            "summary": summary,
            "message": "Your case has been escalated. A human agent will contact you shortly.",
        }
    )


TOOL_DISPATCH: dict[str, Any] = {
    "search_knowledge_base": lambda args: _search_knowledge_base(args["query"]),
    "lookup_order": lambda args: _lookup_order(args["order_id"]),
    "process_return": lambda args: _process_return(args["order_id"], args["reason"]),
    "escalate_to_human": lambda args: _escalate_to_human(
        args["summary"], args.get("priority", "medium")
    ),
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class CustomerSupportAgent:
    """TechStore customer-support agent with RAG + tool calling."""

    MAX_TOOL_ROUNDS = 5  # safety cap on tool-call loops

    def __init__(
        self,
        system_prompt: str,
        endpoint: str = "databricks-claude-sonnet-4-6",
        tool_dispatch: dict[str, Any] | None = None,
    ):
        self.system_prompt = system_prompt
        self.endpoint = endpoint
        self.tool_dispatch = tool_dispatch or TOOL_DISPATCH
        self.client = OpenAI(
            api_key=os.environ.get("DATABRICKS_TOKEN", ""),
            base_url=os.environ.get("DATABRICKS_HOST", "") + "/serving-endpoints",
        )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def predict(self, inputs: dict) -> dict:
        """
        Run the agent on a single input.

        Parameters
        ----------
        inputs : dict
            Must contain a ``question`` key (str).

        Returns
        -------
        dict
            ``response``  -- final assistant text
            ``tool_calls`` -- list of tool invocations made during the turn
        """
        question = inputs.get("question", str(inputs))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        all_tool_calls: list[dict] = []

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.endpoint,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=1500,
                temperature=0.1,
            )

            assistant_msg = response.choices[0].message

            # If no tool calls, we have the final answer.
            if not assistant_msg.tool_calls:
                messages.append({"role": "assistant", "content": assistant_msg.content})
                break

            # Append assistant message (with tool_calls) to history.
            messages.append(assistant_msg.model_dump())

            # Execute each tool call and append results.
            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                logger.info("Tool call: %s(%s)", fn_name, fn_args)

                handler = self.tool_dispatch.get(fn_name)
                if handler is None:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    try:
                        result = handler(fn_args)
                    except Exception as exc:
                        logger.exception("Tool %s failed", fn_name)
                        result = json.dumps({"error": str(exc)})

                all_tool_calls.append(
                    {"tool": fn_name, "arguments": fn_args, "result": json.loads(result)}
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            # Reached MAX_TOOL_ROUNDS without a final text response.
            logger.warning("Agent hit max tool-call rounds (%d)", self.MAX_TOOL_ROUNDS)

        final_text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                final_text = msg["content"]
                break
            if hasattr(msg, "role") and msg.role == "assistant" and msg.content:
                final_text = msg.content
                break

        return {"response": final_text, "tool_calls": all_tool_calls}

    # ------------------------------------------------------------------
    # Convenience: batch predict
    # ------------------------------------------------------------------
    def predict_batch(self, inputs_list: list[dict]) -> list[dict]:
        """Run predict over a list of inputs sequentially."""
        return [self.predict(inp) for inp in inputs_list]
