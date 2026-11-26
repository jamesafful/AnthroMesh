"""Make one preference consultation over loopback MCP."""
import argparse
import json
import os

from anthromesh.client import DecisionClient
from anthromesh.models import CandidateOption, QueryContext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/mcp")
    parser.add_argument("--subject", default="demo-owner")
    parser.add_argument("--context", default="Keep the Friday team sync at 10 a.m. Eastern?")
    args = parser.parse_args()
    client = DecisionClient(args.endpoint, token=os.environ.get("ANTHROMESH_MCP_TOKEN"))
    response = client.consult(QueryContext(
        subject_id=args.subject, domain="calendar", context=args.context,
        candidate_options=[CandidateOption(option_id="keep", label="Keep"),
                           CandidateOption(option_id="review", label="Change")],
    ))
    print(json.dumps(response, indent=2))
