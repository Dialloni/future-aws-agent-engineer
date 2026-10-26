"""Horizon Rewards loyalty API, sitting behind API Gateway with a required key.

This Lambda knows nothing about the key. API Gateway enforces it; the Lambda
only ever sees requests that already passed that check.
"""

import json

MEMBERS = {
    "hz-001": {
        "member_id": "hz-001",
        "member_name": "Alice Moreau",
        "email": "alice@example.com",
        "tier": "Gold",
        "points_balance": 48250,
        "points_to_next_tier": 1750,
        "next_tier": "Platinum",
        "member_since": "2021-06-14",
        "lifetime_flights": 37,
    },
    "hz-002": {
        "member_id": "hz-002",
        "member_name": "Ben Okafor",
        "email": "ben@example.com",
        "tier": "Silver",
        "points_balance": 12480,
        "points_to_next_tier": 7520,
        "next_tier": "Gold",
        "member_since": "2023-02-08",
        "lifetime_flights": 11,
    },
    "hz-003": {
        "member_id": "hz-003",
        "member_name": "Chiara Rossi",
        "email": "chiara@example.com",
        "tier": "Platinum",
        "points_balance": 103900,
        "points_to_next_tier": 0,
        "next_tier": None,
        "member_since": "2018-11-30",
        "lifetime_flights": 92,
    },
}


def handler(event, context):
    """API Gateway proxy integration entrypoint."""
    path_params = event.get("pathParameters") or {}
    member_id = (path_params.get("memberId") or "").strip().lower()

    if not member_id:
        return _response(400, {"error": "memberId path parameter is required"})

    member = MEMBERS.get(member_id)
    if not member:
        return _response(
            404,
            {
                "error": f"No Horizon Rewards member with id {member_id}.",
                "known_members": sorted(MEMBERS),
            },
        )

    return _response(200, member)


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
