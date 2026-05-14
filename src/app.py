import argparse
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import gradio as gr
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


SAMPLE_RATE = 16_000
DEFAULT_LOCAL_MODEL = Path("models/whisper-tiny-ro-smoke-test")
DEFAULT_BASE_PROCESSOR = "openai/whisper-small"
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()

    replacements = {
        "ş": "ș",
        "ţ": "ț",
        "ã": "ă",
        "–": "-",
        "—": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[.,!?;:\"'„”()\[\]]", "", text)
    text = " ".join(text.split())

    return text


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def default_model_path() -> str:
    env_model = os.getenv("ASR_MODEL_PATH")
    if env_model:
        return env_model

    if DEFAULT_LOCAL_MODEL.exists():
        return str(DEFAULT_LOCAL_MODEL)

    return DEFAULT_BASE_PROCESSOR


def load_processor(
    model_path: str,
    processor_path: str | None,
    language: str,
    task: str,
) -> WhisperProcessor:
    candidates = []

    if processor_path:
        candidates.append(processor_path)

    candidates.extend([model_path, DEFAULT_BASE_PROCESSOR])

    last_error: Exception | None = None

    for candidate in candidates:
        try:
            LOGGER.info("Loading Whisper processor from %s", candidate)
            return WhisperProcessor.from_pretrained(
                candidate,
                language=language,
                task=task,
            )
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Could not load Whisper processor from %s: %s", candidate, exc)

    raise RuntimeError(f"Could not load a Whisper processor. Last error: {last_error}")


def configure_generation(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    language: str,
    task: str,
) -> None:
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
    )

    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []

    # Newer Transformers versions expect generation settings on generation_config.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None


class RomanianASRApp:
    def __init__(
        self,
        model_path: str,
        processor_path: str | None,
        language: str,
        task: str,
        generation_max_length: int,
        num_beams: int,
    ) -> None:
        self.device = get_device()
        self.model_path = model_path
        self.generation_max_length = generation_max_length
        self.num_beams = num_beams

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        LOGGER.info("Using device=%s dtype=%s", self.device, dtype)

        self.processor = load_processor(
            model_path=model_path,
            processor_path=processor_path,
            language=language,
            task=task,
        )

        LOGGER.info("Loading Whisper model from %s", model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        LOGGER.info("Whisper model loaded successfully")

        configure_generation(
            model=self.model,
            processor=self.processor,
            language=language,
            task=task,
        )

    def transcribe(self, audio_path: str | None) -> tuple[str, str]:
        if not audio_path:
            LOGGER.info("Transcription requested without an audio file")
            return "", "Upload an audio file or record from the microphone first."

        LOGGER.info("Starting transcription for %s", audio_path)
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.model.dtype,
        )

        generation_kwargs: dict[str, Any] = {
            "input_features": input_features,
            "max_length": self.generation_max_length,
            "num_beams": self.num_beams,
        }

        if hasattr(inputs, "attention_mask"):
            generation_kwargs["attention_mask"] = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**generation_kwargs)

        raw_transcript = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        transcript = normalize_text(raw_transcript)

        status = f"Model: {self.model_path} | Device: {self.device}"
        LOGGER.info("Finished transcription; transcript_length=%s", len(transcript))
        return transcript, status


def build_llm_prompt(mode: str, transcript: str, question: str) -> str:
    if mode == "Correct Romanian transcript":
        return (
            "Corecteaza transcriptul romanesc de mai jos. Pastreaza sensul, "
            "nu inventa informatii si returneaza doar varianta corectata.\n\n"
            f"Transcript:\n{transcript}"
        )

    if mode == "Summarize in Romanian":
        return (
            "Rezuma in romana transcriptul de mai jos in 3-5 idei clare. "
            "Nu adauga informatii care nu apar in transcript.\n\n"
            f"Transcript:\n{transcript}"
        )

    if mode == "Extract action items":
        return (
            "Extrage actiunile concrete, deciziile si termenele din transcriptul "
            "de mai jos. Daca nu exista, spune clar ca nu ai identificat actiuni.\n\n"
            f"Transcript:\n{transcript}"
        )

    if mode == "Translate to English":
        return (
            "Translate the following Romanian transcript into natural English. "
            "Do not add details that are not present in the transcript.\n\n"
            f"Transcript:\n{transcript}"
        )

    return (
        "Raspunde la intrebare folosind strict transcriptul de mai jos. "
        "Daca transcriptul nu contine raspunsul, spune ca informatia nu apare.\n\n"
        f"Intrebare:\n{question}\n\nTranscript:\n{transcript}"
    )


def call_openai_compatible_chat(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful assistant for Romanian ASR transcripts. "
                    "Preserve meaning and avoid unsupported assumptions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        LOGGER.info("Sending LLM request to %s using model=%s", url, model)
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        LOGGER.warning("LLM request failed with HTTP %s: %s", exc.code, details)

        if exc.code in {401, 403}:
            raise RuntimeError(
                "LLM API key is not working. The provider rejected the request "
                f"with HTTP {exc.code}. Check that the key is valid, active, and "
                "allowed to use the selected model."
            ) from exc

        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        LOGGER.warning("LLM request failed before receiving a response: %s", exc.reason)
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        LOGGER.warning("Unexpected LLM response body: %s", body)
        raise RuntimeError(f"Unexpected LLM response: {body}") from exc


def run_llm_task(
    transcript: str,
    mode: str,
    question: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
) -> str:
    transcript = transcript.strip()
    question = question.strip()
    LOGGER.info("LLM task requested: %s", mode)

    if not transcript:
        LOGGER.info("LLM task skipped because transcript is empty")
        return "Transcribe audio first, then run the LLM step."

    if mode == "Answer question about transcript" and not question:
        return "Add a question for the transcript Q&A mode."

    resolved_api_key = api_key.strip() or os.getenv("OPENAI_API_KEY", "")
    resolved_base_url = base_url.strip() or os.getenv(
        "LLM_BASE_URL",
        "https://api.openai.com/v1",
    )
    resolved_model = model.strip() or os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not resolved_api_key:
        LOGGER.info("LLM task skipped because no API key is configured")
        return (
            "LLM API key is not working because no key is configured. Set "
            "OPENAI_API_KEY or paste a valid key in the app, then run this step again."
        )

    prompt = build_llm_prompt(mode=mode, transcript=transcript, question=question)

    try:
        return call_openai_compatible_chat(
            prompt=prompt,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError as exc:
        LOGGER.warning("LLM task failed: %s", exc)
        return str(exc)


def create_interface(app: RomanianASRApp, llm_timeout_seconds: int) -> gr.Blocks:
    with gr.Blocks(title="Romanian ASR + LLM Demo") as demo:
        gr.Markdown("# Romanian ASR + LLM Demo")

        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    label="Audio input",
                    sources=["microphone", "upload"],
                    type="filepath",
                )
                transcribe_button = gr.Button("Transcribe", variant="primary")
                status = gr.Textbox(label="Runtime status", interactive=False)

            with gr.Column():
                transcript = gr.Textbox(
                    label="ASR transcript",
                    lines=8,
                )

        gr.Markdown("## LLM post-processing")

        with gr.Row():
            mode = gr.Dropdown(
                label="LLM task",
                choices=[
                    "Correct Romanian transcript",
                    "Summarize in Romanian",
                    "Extract action items",
                    "Translate to English",
                    "Answer question about transcript",
                ],
                value="Correct Romanian transcript",
            )
            llm_model = gr.Textbox(
                label="LLM model",
                value=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            )

        question = gr.Textbox(
            label="Question",
            placeholder="Used only for transcript Q&A.",
        )

        with gr.Accordion("LLM API settings", open=False):
            gr.Markdown(
                "The LLM step only works with a valid API key. If the key is "
                "missing, expired, invalid, or not allowed for the selected model, "
                "the app will show that the LLM API key is not working."
            )
            api_key = gr.Textbox(
                label="API key",
                type="password",
                placeholder="Defaults to OPENAI_API_KEY from the environment.",
            )
            base_url = gr.Textbox(
                label="OpenAI-compatible base URL",
                value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            )

        run_llm_button = gr.Button("Run LLM task", variant="secondary")
        llm_output = gr.Textbox(
            label="LLM output",
            lines=8,
        )

        transcribe_button.click(
            fn=app.transcribe,
            inputs=audio,
            outputs=[transcript, status],
        )

        run_llm_button.click(
            fn=lambda transcript_text, task_mode, task_question, key, url, model_name: run_llm_task(
                transcript=transcript_text,
                mode=task_mode,
                question=task_question,
                api_key=key,
                base_url=url,
                model=model_name,
                timeout_seconds=llm_timeout_seconds,
            ),
            inputs=[transcript, mode, question, api_key, base_url, llm_model],
            outputs=llm_output,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Gradio demo for Romanian Whisper ASR with LLM integration.",
    )
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--processor-path", default=os.getenv("ASR_PROCESSOR_PATH"))
    parser.add_argument("--language", default=os.getenv("ASR_LANGUAGE", "romanian"))
    parser.add_argument("--task", default=os.getenv("ASR_TASK", "transcribe"))
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    LOGGER.info("Starting Romanian ASR + LLM app")
    LOGGER.info("Configured ASR model path: %s", args.model_path)
    app = RomanianASRApp(
        model_path=args.model_path,
        processor_path=args.processor_path,
        language=args.language,
        task=args.task,
        generation_max_length=args.generation_max_length,
        num_beams=args.num_beams,
    )
    demo = create_interface(app=app, llm_timeout_seconds=args.llm_timeout_seconds)
    LOGGER.info(
        "Launching Gradio app on http://%s:%s",
        args.server_name,
        args.server_port,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
