# Aster Voice API

Isolated Render experiment for Pocket TTS Indonesian 6L. This repository is separate from the main Aster application.

Endpoints:
- `GET /health`
- `POST /tts` with JSON `{ "text": "Selamat pagi" }`

The model is loaded lazily on the first `/tts` request so the service can boot before model inference is tested.
