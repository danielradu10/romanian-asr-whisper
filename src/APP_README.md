# Romanian ASR + LLM Web App

This document explains how to run and test `src/app.py`.

The app is a local Gradio web interface for demonstrating the Romanian ASR model from this project. It supports microphone recording or audio upload, transcribes speech with the fine-tuned Whisper checkpoint, and optionally sends the transcript to an OpenAI-compatible LLM API for post-processing.

## Main Features

- Local web interface built with Gradio.
- Romanian speech-to-text using Whisper through Hugging Face Transformers.
- Uses the project fine-tuned checkpoint by default when it exists:

```text
models/whisper-tiny-ro-smoke-test
```

- Audio input from microphone or uploaded audio file.
- Transcript normalization for Romanian text.
- Optional LLM post-processing:
  - Correct Romanian transcript
  - Summarize in Romanian
  - Extract action items
  - Translate to English
  - Answer questions about the transcript
- Runtime logs for model loading, transcription, LLM calls, and the running port.
- Clear warning when the LLM API key is missing or not working.

## Requirements

Run the app from the repository root.

Create and activate the virtual environment if it is not already active:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The app uses these main packages:

- `gradio`
- `torch`
- `transformers`
- `librosa`

## Model Used by the App

By default, `src/app.py` first checks for:

```text
models/whisper-tiny-ro-smoke-test
```

If that folder exists, it is used as the ASR model. This is the local fine-tuned checkpoint included in the project.

If the folder does not exist, the app falls back to:

```text
openai/whisper-small
```

To make sure the fine-tuned project model is used, run the app explicitly with:

```bash
python src/app.py --model-path models/whisper-tiny-ro-smoke-test
```

You can also set the model path through an environment variable:

```bash
export ASR_MODEL_PATH=models/whisper-tiny-ro-smoke-test
python src/app.py
```

## Running the App

Start the app from the repository root:

```bash
source .venv/bin/activate
python src/app.py --model-path models/whisper-tiny-ro-smoke-test --server-port 7862
```

Open the printed local URL in the browser:

```text
http://127.0.0.1:7862
```

If port `7862` is already busy, choose another port:

```bash
python src/app.py --model-path models/whisper-tiny-ro-smoke-test --server-port 7870
```

The app logs the running port at startup. Example:

```text
Launching Gradio app on http://127.0.0.1:7862
```

## How to Test ASR

1. Open the Gradio page in the browser.
2. In the `Audio input` component, either:
   - record Romanian speech with the microphone, or
   - upload a Romanian audio file.
3. Click `Transcribe`.
4. Read the result in `ASR transcript`.
5. Check `Runtime status` to confirm the model path and device.

Expected runtime status example:

```text
Model: models/whisper-tiny-ro-smoke-test | Device: cpu
```

The device may be `cpu`, `mps`, or `cuda`, depending on the machine.

## How to Test the LLM Integration

The ASR transcription works without an LLM API key.

The LLM post-processing step requires a valid OpenAI-compatible API key. You can provide it in one of two ways.

Option 1: set an environment variable before launching the app:

```bash
export OPENAI_API_KEY="your_api_key_here"
python src/app.py --model-path models/whisper-tiny-ro-smoke-test --server-port 7862
```

Option 2: paste the key in the web interface:

1. Open `LLM API settings`.
2. Paste the key into `API key`.
3. Keep the default base URL for OpenAI:

```text
https://api.openai.com/v1
```

4. Choose an LLM task.
5. Click `Run LLM task`.

Default LLM model:

```text
gpt-4o-mini
```

You can change it in the `LLM model` field or by setting:

```bash
export LLM_MODEL=gpt-4o-mini
```

## Important Note About the LLM API Key

The LLM API key used in this project may not actually work unless it is a real, active key from the selected provider.

The app explicitly reports this case. For example, if no key is configured, it shows:

```text
LLM API key is not working because no key is configured.
```

If the provider rejects the key with `401` or `403`, it shows:

```text
LLM API key is not working. The provider rejected the request with HTTP 401.
Check that the key is valid, active, and allowed to use the selected model.
```

This means the ASR part can still be demonstrated locally, but the LLM part needs a valid external API key and network access.

## Command Line Arguments

`src/app.py` supports the following arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--model-path` | `models/whisper-tiny-ro-smoke-test` if available | ASR model checkpoint or Hugging Face model name. |
| `--processor-path` | `ASR_PROCESSOR_PATH` env var | Optional processor/tokenizer path. |
| `--language` | `romanian` | Whisper language setting. |
| `--task` | `transcribe` | Whisper task setting. |
| `--generation-max-length` | `225` | Maximum generated token length. |
| `--num-beams` | `1` | Beam search size for generation. |
| `--server-name` | `127.0.0.1` | Host used by Gradio. |
| `--server-port` | `7860` | Port used by Gradio. |
| `--share` | disabled | Creates a public Gradio link when enabled. |
| `--llm-timeout-seconds` | `60` | Timeout for LLM API calls. |

Example with custom host and port:

```bash
python src/app.py \
  --model-path models/whisper-tiny-ro-smoke-test \
  --server-name 127.0.0.1 \
  --server-port 7862
```

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `ASR_MODEL_PATH` | ASR model path used when `--model-path` is not passed. |
| `ASR_PROCESSOR_PATH` | Optional processor/tokenizer path. |
| `ASR_LANGUAGE` | Whisper language setting. |
| `ASR_TASK` | Whisper task setting. |
| `OPENAI_API_KEY` | API key for the LLM step. |
| `LLM_BASE_URL` | OpenAI-compatible API base URL. |
| `LLM_MODEL` | LLM model name. |

## Logs

The app logs important runtime events to the terminal:

- app startup
- selected ASR model path
- device and dtype
- processor loading
- model loading
- running Gradio URL and port
- transcription start and finish
- LLM task requests
- LLM API failures

Example:

```text
2026-05-14 21:45:00,000 | INFO | Starting Romanian ASR + LLM app
2026-05-14 21:45:00,001 | INFO | Configured ASR model path: models/whisper-tiny-ro-smoke-test
2026-05-14 21:45:02,100 | INFO | Whisper model loaded successfully
2026-05-14 21:45:02,300 | INFO | Launching Gradio app on http://127.0.0.1:7862
```

## Troubleshooting

### Port Already in Use

If Gradio reports that the selected port is busy, use another port:

```bash
python src/app.py --model-path models/whisper-tiny-ro-smoke-test --server-port 7870
```

### ASR Model Does Not Load

Check that the checkpoint exists:

```bash
ls models/whisper-tiny-ro-smoke-test
```

If using another model, pass it explicitly:

```bash
python src/app.py --model-path path/to/your/checkpoint
```

### LLM API Key Is Not Working

The LLM step needs a valid provider key. If the app says the key is not working, check:

- the key is not empty
- the key has not expired
- the key has access to the selected model
- the selected `LLM_BASE_URL` is correct
- the machine has internet access

The ASR transcription can still be tested without the LLM key.

### Browser Does Not Open Automatically

Open the logged URL manually:

```text
http://127.0.0.1:7862
```

## Minimal Verification Commands

Check Python syntax:

```bash
python -m py_compile src/app.py
```

Start the app with the fine-tuned model:

```bash
python src/app.py --model-path models/whisper-tiny-ro-smoke-test --server-port 7862
```

Then open:

```text
http://127.0.0.1:7862
```

