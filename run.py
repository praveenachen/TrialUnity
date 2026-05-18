import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8001"))
    reload_enabled = os.getenv("RELOAD", "true").lower() in {"1", "true", "yes"}
    app_url = f"http://{host}:{port}"

    print()
    print("TrialUnity is starting...")
    print(f"Frontend: {app_url}")
    print(f"API docs: {app_url}/docs")
    print("Press Ctrl+C to stop the server.")
    print()

    uvicorn.run("backend.app.main:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
