"""Horizon Travel booking backend, exposed to WanderBot through AgentCore Gateway.

The agent owns no booking logic. Gateway calls this Lambda with the tool input
as the event, and names the tool in the client context.
"""

import json

# Stands in for the bookings database. Keyed by reference so get_booking is a
# dict lookup; list_bookings_by_email scans, which is fine at this size.
BOOKINGS = {
    "BK-1001": {
        "booking_reference": "BK-1001",
        "customer_email": "alice@example.com",
        "customer_name": "Alice Moreau",
        "flight_number": "HZ-204",
        "origin": "BCN",
        "destination": "FCO",
        "date": "2026-03-20",
        "cabin": "Economy",
        "passengers": 2,
        "total_paid_usd": 418.0,
        "status": "CONFIRMED",
    },
    "BK-1002": {
        "booking_reference": "BK-1002",
        "customer_email": "alice@example.com",
        "customer_name": "Alice Moreau",
        "flight_number": "HZ-310",
        "origin": "FCO",
        "destination": "BCN",
        "date": "2026-03-27",
        "cabin": "Economy",
        "passengers": 2,
        "total_paid_usd": 396.0,
        "status": "CONFIRMED",
    },
    "BK-1003": {
        "booking_reference": "BK-1003",
        "customer_email": "ben@example.com",
        "customer_name": "Ben Okafor",
        "flight_number": "HZ-512",
        "origin": "LHR",
        "destination": "CDG",
        "date": "2026-04-02",
        "cabin": "Business",
        "passengers": 1,
        "total_paid_usd": 742.0,
        "status": "CANCELLED",
    },
}


def get_booking(booking_reference: str) -> dict:
    """Look up one booking by its reference."""
    if not booking_reference:
        return {"error": "booking_reference is required"}

    booking = BOOKINGS.get(booking_reference.strip().upper())
    if not booking:
        return {
            "error": f"No booking found with reference {booking_reference}.",
            "known_references": sorted(BOOKINGS),
        }
    return booking


def list_bookings_by_email(customer_email: str) -> dict:
    """Return every booking belonging to one customer."""
    if not customer_email:
        return {"error": "customer_email is required"}

    email = customer_email.strip().lower()
    matches = [b for b in BOOKINGS.values() if b["customer_email"].lower() == email]
    return {"bookings": matches, "total": len(matches), "customer_email": email}


def handler(event, context):
    """Gateway entrypoint.

    Gateway puts the tool name in the client context, prefixed with the target
    name as `TargetName___toolName`. Strip the prefix before dispatching, or
    every call falls through to the unknown-tool branch.
    """
    tool_name = ""
    client_context = getattr(context, "client_context", None)
    if client_context and getattr(client_context, "custom", None):
        tool_name = client_context.custom.get("bedrockAgentCoreToolName", "")

    if "___" in tool_name:
        tool_name = tool_name.split("___", 1)[1]

    if tool_name == "get_booking":
        result = get_booking(event.get("booking_reference", ""))
    elif tool_name == "list_bookings_by_email":
        result = list_bookings_by_email(event.get("customer_email", ""))
    else:
        result = {
            "error": f"Unknown tool: {tool_name or '(none supplied)'}",
            "supported": ["get_booking", "list_bookings_by_email"],
        }

    # Gateway expects a normal Lambda proxy-style response; the agent sees body.
    return {"statusCode": 200, "body": json.dumps(result)}
