from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


LABEL_TEMPLATES = {
    "primary": {
        "subjects": ["Project follow up", "Meeting notes", "Lunch plan", "Document review", "Weekly sync"],
        "senders": ["manager@example.com", "teammate@example.com", "friend@example.com"],
        "bodies": [
            "Please review the notes and share your feedback before end of day.",
            "Can we move our meeting to tomorrow afternoon after the planning call?",
            "I added comments to the document and assigned the open action items.",
            "Thanks for the update. Let us discuss the remaining blockers in standup.",
        ],
    },
    "updates": {
        "subjects": ["Invoice available", "Delivery status", "Password changed", "Build completed", "Statement ready"],
        "senders": ["billing@service.example", "no-reply@updates.example", "ci@example.com"],
        "bodies": [
            "Your monthly invoice is ready and can be viewed in your account dashboard.",
            "Your package has shipped and the tracking details are now available.",
            "The deployment completed successfully and logs are attached to the build.",
            "Your account statement for this month is ready for download.",
        ],
    },
    "promotion": {
        "subjects": ["Weekend sale", "Exclusive discount", "Limited offer", "Cashback deal", "Coupon inside"],
        "senders": ["deals@shop.example", "promo@store.example", "offers@market.example"],
        "bodies": [
            "Get a special discount and free shipping on selected products this weekend.",
            "Use this coupon code to unlock cashback on your next purchase.",
            "Limited time sale with exclusive deals for members only.",
            "Buy today and save more with our festival offer.",
        ],
    },
    "social": {
        "subjects": ["New connection request", "Profile activity", "New follower", "Comment notification", "Group invite"],
        "senders": ["notifications@social.example", "alerts@network.example", "updates@community.example"],
        "bodies": [
            "A colleague sent you a connection request and viewed your profile.",
            "You have new followers and reactions on your recent post.",
            "Someone commented on your update in the project group.",
            "You were invited to join a community discussion.",
        ],
    },
    "spam": {
        "subjects": ["You won cash", "Cheap medicine offer", "Prize claim", "Free gift now", "Lottery winner"],
        "senders": ["winner@unknown.example", "sales@random.example", "claim@prize.example"],
        "bodies": [
            "Congratulations!!! You won $5000. Click the URL to claim your prize immediately.",
            "Buy cheap pills with no prescription and get a free bonus offer today.",
            "You have been selected for a lottery reward. Act now to receive cash.",
            "Free gift waiting for you. Click now before the offer expires!!!",
        ],
    },
    "phishing": {
        "subjects": ["Account suspended", "Verify password now", "Urgent security alert", "OTP required", "Final login notice"],
        "senders": ["security-alert@paypa1.example", "support@secure-login.example", "verify@bank-check.example"],
        "bodies": [
            "URGENT: your account will expire. Verify your password at http://secure-login.example/verify.",
            "Final notice: account suspended. Confirm OTP and login credential immediately.",
            "Unusual activity detected. Update your PIN and password now to restore access.",
            "Your bank login is locked. Visit www.account-verify.example and confirm details.",
        ],
    },
}

NOISE_PHRASES = [
    "Reference id {number}.",
    "This message was generated for account {number}.",
    "Please keep this email for your records.",
    "Reply is not required for this notification.",
    "Sent from automated mail service.",
]


def build_rows(rows_per_label: int, seed: int) -> list[dict[str, str]]:
    random.seed(seed)
    rows: list[dict[str, str]] = []
    for label, parts in LABEL_TEMPLATES.items():
        for index in range(rows_per_label):
            number = random.randint(10000, 99999)
            rows.append(
                {
                    "subject": f"{random.choice(parts['subjects'])} #{index + 1}",
                    "sender": random.choice(parts["senders"]),
                    "text": f"{random.choice(parts['bodies'])} {random.choice(NOISE_PHRASES).format(number=number)}",
                    "label": label,
                }
            )
    random.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic email data.")
    parser.add_argument("--output", default="data/synthetic_emails.csv")
    parser.add_argument("--rows-per-label", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(rows_per_label=args.rows_per_label, seed=args.seed)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["subject", "sender", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

