from google import genai

from core.config import settings


def main() -> None:
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.gemini_api_key)
    for model in client.models.list():
        actions = getattr(model, "supported_actions", []) or []
        supports_generation = any("generate" in str(action).lower() for action in actions)
        print(model.name, " | supports generation:", supports_generation)


if __name__ == "__main__":
    main()
