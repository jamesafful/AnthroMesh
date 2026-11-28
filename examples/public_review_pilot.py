"""Run an exploratory replay using point-in-time features on public GitHub reviews."""
import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from examples.replay_public_reviews import Review, replay


def fetch(path):
    request = Request("https://api.github.com" + path, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "AnthroMesh-public-research",
    })
    with urlopen(request, timeout=20) as response:
        if 'rel="next"' in response.headers.get("Link", "") and "per_page=100" in path:
            raise ValueError("Historical source is incomplete.")
        return json.load(response)


def collect(repo, pull_numbers):
    rows = []
    counts = Counter()
    for number in pull_numbers:
        seen = set()
        try:
            reviews = fetch(f"/repos/{repo}/pulls/{number}/reviews?per_page=100")
        except ValueError:
            counts["incomplete_review_lists"] += 1
            continue
        except HTTPError as exc:
            if exc.code != 404:
                raise
            counts["unavailable_pull_numbers"] += 1
            continue
        for event in reviews:
            counts["reviews_seen"] += 1
            user = event.get("user") or {}
            if (user.get("type") != "User" or "id" not in user
                    or event.get("author_association") not in {"MEMBER", "OWNER", "COLLABORATOR"}
                    or event.get("state") not in {"APPROVED", "CHANGES_REQUESTED"}
                    or not event.get("commit_id") or not event.get("submitted_at")):
                counts["ineligible_reviews"] += 1
                continue
            if user["id"] in seen:
                counts["later_reviews_on_same_pr"] += 1
                continue
            seen.add(user["id"])
            stamp = datetime.fromisoformat(event["submitted_at"])
            head = event["commit_id"]
            try:
                commit = fetch(f"/repos/{repo}/commits/{head}?per_page=100")
                runs = fetch(f"/repos/{repo}/commits/{head}/check-runs?filter=all&per_page=100")
            except ValueError:
                counts["incomplete_commit_or_checks"] += 1
                continue
            except HTTPError as exc:
                if exc.code not in {404, 422}:
                    raise
                counts["missing_commit_or_checks"] += 1
                continue
            files = commit.get("files", [])
            if len(commit.get("parents", [])) != 1 or not files:
                counts["unsupported_commit"] += 1
                continue
            finished = [run["conclusion"] for run in runs["check_runs"]
                        if run.get("completed_at") and
                        datetime.fromisoformat(run["completed_at"]) < stamp]
            if not finished:
                counts["no_completed_checks_before_review"] += 1
                continue
            # Commit delta relative to its parent, with CI conclusions observed
            # before submission. This is a coarse context, not a historical PR diff.
            context = json.dumps({
                "repo": repo,
                "files": min(len(files), 5),
                "tests_touched": any("tests" in f["filename"].split("/") for f in files),
                "check_conclusions": sorted(set(finished)),
            }, sort_keys=True)
            rows.append(Review(
                event_id=str(event["id"]),
                subject_id=sha256(f"{repo}:{user['id']}".encode()).hexdigest(),
                reviewed_at=stamp, context=context, action=event["state"],
            ))
    return {"collection": dict(counts), "runtime": replay(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("pulls", nargs="+", type=int)
    args = parser.parse_args()
    print(json.dumps(collect(args.repo, sorted(set(args.pulls))), indent=2))
