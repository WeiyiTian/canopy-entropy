import json
import os
import urllib.request

from dotenv import load_dotenv

KEY_ENDPOINT = "https://openrouter.ai/api/v1/key"

load_dotenv()


def main() -> None:
    request = urllib.request.Request(
        KEY_ENDPOINT,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        usage = json.load(response)["data"]

    print(f"spent today  ${usage['usage_daily']:.4f}")
    print(f"spent total  ${usage['usage']:.4f}")
    print(f"remaining    ${usage['limit_remaining']:.4f} of ${usage['limit']:.2f}")


if __name__ == "__main__":
    main()
