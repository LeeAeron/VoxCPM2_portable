"""VoxCPM2 Portable — Multilingual TTS (RU/EN).

Портативная русскоязычная сборка VoxCPM2 от Nerual Dreming + Нейро-Софт.
Поддерживает 30 языков, включая русский. Режимы: TTS / Voice Design / Voice Cloning / Ultimate Cloning.
"""

# === КРИТИЧЕСКИЙ ПАТЧ: отключение torch._dynamo ДО импорта voxcpm ===
# (Фикс threading-бага torch.compile при многопоточной загрузке)
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# === Стандартные импорты ===
import random
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf

from voxcpm import VoxCPM

# === Конфигурация ===
SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "output"
VOICES_DIR = SCRIPT_DIR / "voices"
OUTPUT_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)

MODEL_REF = "openbmb/VoxCPM2"

# === Voice pack (русские голоса с HuggingFace) ===
CLOUD_VOICES_REPO = "Slait/russia_voices"
CLOUD_VOICES_BASE_URL = "https://huggingface.co/datasets/Slait/russia_voices/resolve/main"
_CLOUD_VOICES_CACHE: list[str] = []


def scan_local_voices() -> list[str]:
    """Список локальных голосов в voices/ (.mp3/.wav/.flac)."""
    exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    names = set()
    for p in VOICES_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            names.add(p.stem)
    return sorted(names)


def voice_audio_path(name: str) -> Optional[str]:
    """Путь к аудиофайлу голоса в voices/ (любое расширение из поддерживаемых)."""
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            return str(p)
    return None


def voice_transcript(name: str) -> str:
    """Транскрипт голоса из voices/{name}.txt, если есть."""
    p = VOICES_DIR / f"{name}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


def fetch_cloud_voices_list() -> list[str]:
    """Получить список mp3-голосов из HF dataset."""
    global _CLOUD_VOICES_CACHE
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset"))
        voices = sorted(f[:-4] for f in files if f.endswith(".mp3"))
        _CLOUD_VOICES_CACHE = voices
        return voices
    except Exception as exc:
        print(f"[voices] fetch list error: {exc}")
        return []


def download_cloud_voice(name: str) -> bool:
    """Скачать один голос (mp3+txt) в voices/."""
    import requests
    try:
        mp3_url = f"{CLOUD_VOICES_BASE_URL}/{name}.mp3?download=true"
        r = requests.get(mp3_url, timeout=60, stream=True)
        r.raise_for_status()
        with open(VOICES_DIR / f"{name}.mp3", "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        # txt необязателен
        try:
            txt_url = f"{CLOUD_VOICES_BASE_URL}/{name}.txt?download=true"
            r2 = requests.get(txt_url, timeout=30)
            if r2.status_code == 200:
                (VOICES_DIR / f"{name}.txt").write_text(r2.text, encoding="utf-8")
        except Exception:
            pass
        return True
    except Exception as exc:
        print(f"[voices] download '{name}' error: {exc}")
        return False


def download_all_cloud_voices(progress=gr.Progress()):
    """Скачать все доступные голоса из HF dataset, с прогрессом."""
    voices = fetch_cloud_voices_list()
    if not voices:
        return "Не удалось получить список голосов с HuggingFace."
    local = set(scan_local_voices())
    to_download = [v for v in voices if v not in local]
    if not to_download:
        return f"Все {len(voices)} голосов уже скачаны."
    ok, fail = 0, 0
    for i, v in enumerate(progress.tqdm(to_download, desc="Скачивание")):
        if download_cloud_voice(v):
            ok += 1
        else:
            fail += 1
    return f"Скачано: {ok}, ошибок: {fail}. Всего в voices/: {len(scan_local_voices())}."

# === Определение устройства ===
def _detect_device() -> tuple[str, str]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda", f"{name} | VRAM: {vram:.1f} GB"
    return "cpu", "CPU (экспериментально / experimental — very slow)"

DEVICE, DEVICE_INFO = _detect_device()
print(f"[VoxCPM2] Device: {DEVICE_INFO}")

# === Ленивая загрузка модели ===
_model = None

def get_model() -> VoxCPM:
    global _model
    if _model is not None:
        return _model
    print(f"[VoxCPM2] Loading {MODEL_REF} (first time — downloading ~4-5 GB)...")
    _model = VoxCPM.from_pretrained(
        MODEL_REF,
        load_denoiser=True,
        optimize=False,  # фикс threading-багов torch.compile (из Colab-ноутбука AIQuest)
    )
    print(f"[VoxCPM2] Model loaded. Sample rate: {_model.tts_model.sample_rate} Hz")
    return _model


# === Вспомогательные утилиты ===
def _resolve_seed(seed, locked: bool) -> int:
    if locked and seed is not None and int(seed) >= 0:
        s = int(seed)
    else:
        s = random.randint(0, 2**31 - 1)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


def _collect_audio(result) -> np.ndarray:
    """Обрабатывает как generator (потоковая генерация), так и np.ndarray."""
    if isinstance(result, np.ndarray):
        return result
    chunks = []
    for chunk in result:
        arr = np.asarray(chunk)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        chunks.append(arr)
    if not chunks:
        raise gr.Error("Модель не вернула аудио / Model returned no audio.")
    return np.concatenate(chunks)


def _save_wav(wav: np.ndarray, sr: int, prefix: str = "tts") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{prefix}_{ts}.wav"
    sf.write(str(out_path), wav, sr)
    return str(out_path)


def _build_kwargs(
    *,
    text: str,
    cfg: float,
    steps: int,
    normalize: bool,
    retry: bool,
    reference_wav_path: Optional[str] = None,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    denoise: Optional[bool] = None,
) -> dict:
    kwargs = {
        "text": text,
        "cfg_value": float(cfg),
        "inference_timesteps": int(steps),
        "normalize": bool(normalize),
        "retry_badcase": bool(retry),
    }
    if reference_wav_path:
        kwargs["reference_wav_path"] = reference_wav_path
    if prompt_wav_path:
        kwargs["prompt_wav_path"] = prompt_wav_path
    if prompt_text:
        kwargs["prompt_text"] = prompt_text
    if denoise is not None and reference_wav_path is not None:
        kwargs["denoise"] = bool(denoise)
    return kwargs


# === Функции генерации ===
def tts_generate(text, cfg, steps, seed, locked, normalize, retry):
    if not (text or "").strip():
        raise gr.Error("Введите текст / Please enter text.")
    try:
        model = get_model()
        used_seed = _resolve_seed(seed, locked)
        kwargs = _build_kwargs(
            text=text.strip(), cfg=cfg, steps=steps,
            normalize=normalize, retry=retry,
        )
        wav = _collect_audio(model.generate(**kwargs))
        return _save_wav(wav, model.tts_model.sample_rate, "tts"), used_seed
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Ошибка генерации / Generation error: {e}") from e


def voice_design(description, text, cfg, steps, seed, locked, normalize, retry):
    if not (text or "").strip():
        raise gr.Error("Введите текст / Please enter text.")
    if not (description or "").strip():
        raise gr.Error("Введите описание голоса / Please enter voice description.")
    try:
        model = get_model()
        combined = f"({description.strip()}){text.strip()}"
        used_seed = _resolve_seed(seed, locked)
        kwargs = _build_kwargs(
            text=combined, cfg=cfg, steps=steps,
            normalize=normalize, retry=retry,
        )
        wav = _collect_audio(model.generate(**kwargs))
        return _save_wav(wav, model.tts_model.sample_rate, "design"), used_seed
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Ошибка генерации / Generation error: {e}") from e


def voice_clone(text, ref_audio, style, transcript, cfg, steps, seed, locked, normalize, denoise, retry):
    """Voice Cloning. Если transcript заполнен — автоматически используется Ultimate-режим
    (prompt_wav_path + prompt_text), иначе обычный reference-only."""
    if not (text or "").strip():
        raise gr.Error("Введите текст / Please enter text.")
    if not ref_audio:
        raise gr.Error("Загрузите референс-аудио / Please upload reference audio.")
    try:
        model = get_model()
        final_text = text.strip()
        if style and style.strip():
            final_text = f"({style.strip()}){final_text}"
        used_seed = _resolve_seed(seed, locked)

        transcript_clean = (transcript or "").strip()
        if transcript_clean:
            # Ultimate-режим: prompt_wav + prompt_text + reference
            kwargs = _build_kwargs(
                text=final_text, cfg=cfg, steps=steps,
                normalize=normalize, retry=retry,
                prompt_wav_path=ref_audio, prompt_text=transcript_clean,
                reference_wav_path=ref_audio, denoise=denoise,
            )
        else:
            # Обычный reference-режим
            kwargs = _build_kwargs(
                text=final_text, cfg=cfg, steps=steps,
                normalize=normalize, retry=retry,
                reference_wav_path=ref_audio, denoise=denoise,
            )
        wav = _collect_audio(model.generate(**kwargs))
        return _save_wav(wav, model.tts_model.sample_rate, "clone"), used_seed
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Ошибка генерации / Generation error: {e}") from e


def ultimate_clone(text, ref_audio, transcript, cfg, steps, seed, locked, normalize, denoise, retry):
    if not (text or "").strip():
        raise gr.Error("Введите текст / Please enter text.")
    if not ref_audio:
        raise gr.Error("Загрузите референс-аудио / Please upload reference audio.")
    if not (transcript or "").strip():
        raise gr.Error("Введите транскрипт референса / Please enter reference transcript.")
    try:
        model = get_model()
        used_seed = _resolve_seed(seed, locked)
        kwargs = _build_kwargs(
            text=text.strip(), cfg=cfg, steps=steps,
            normalize=normalize, retry=retry,
            prompt_wav_path=ref_audio,
            prompt_text=transcript.strip(),
            reference_wav_path=ref_audio,
            denoise=denoise,
        )
        wav = _collect_audio(model.generate(**kwargs))
        return _save_wav(wav, model.tts_model.sample_rate, "ultimate"), used_seed
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Ошибка генерации / Generation error: {e}") from e


# === i18n (RU / EN) ===
_I18N_TRANSLATIONS = {
    "en": {
        "app_title": "VoxCPM2 — Multilingual TTS",
        "app_subtitle": "2B parameters · 30 languages · 48 kHz output · Voice Design & Cloning",
        "tab_tts": "Text-to-Speech",
        "tab_design": "Voice Design",
        "tab_clone": "Voice Cloning",
        "tab_ultimate": "Ultimate Cloning",
        "tts_instructions": "Enter any text and the model will synthesize speech with natural prosody. Supports 30 languages including Russian, English, Chinese, etc.",
        "design_instructions": "Create a brand-new voice from a natural-language description — no reference audio needed. Describe gender, age, tone, emotion, pace, accent.",
        "clone_instructions": "Upload a short reference audio clip (5-30 s, up to 50 s) to clone the voice. Optionally steer emotion/pace with a style description.",
        "ultimate_instructions": "Provide reference audio AND its exact transcript for maximum fidelity cloning. The model continues the reference audio preserving every vocal detail.",
        "text_label": "Text",
        "text_placeholder": "Enter text in any of the 30 supported languages...",
        "description_label": "Voice description",
        "description_placeholder": "e.g. A young woman, gentle and sweet voice",
        "content_label": "Text content",
        "content_placeholder": "Hello, welcome to VoxCPM2!",
        "reference_label": "Reference audio",
        "style_label": "Style description (optional)",
        "style_placeholder": "e.g. slightly faster, cheerful tone",
        "transcript_label": "Reference audio transcript",
        "transcript_placeholder": "The exact transcript of the reference audio",
        "cfg_label": "CFG Scale",
        "cfg_info": "Higher = closer to prompt, lower = more creative",
        "steps_label": "Inference Steps",
        "steps_info": "Higher = better quality but slower (5-10 for speed)",
        "seed_label": "Seed",
        "lock_label": "Lock Seed",
        "advanced_label": "Advanced settings",
        "normalize_label": "Text normalization (wetext)",
        "normalize_info": "Normalize numbers, dates, abbreviations",
        "denoise_label": "Denoise reference audio",
        "denoise_info": "Apply ZipEnhancer denoising to reference audio",
        "retry_label": "Retry on bad case",
        "retry_info": "Regenerate if quality is poor",
        "generate_tts": "Generate Speech",
        "generate_design": "Design Voice",
        "generate_clone": "Clone Voice",
        "generate_ultimate": "Ultimate Clone",
        "output_label": "Generated audio",
        "device_label": "Device",
    },
    "ru": {
        "app_title": "VoxCPM2 — Мультиязычный TTS",
        "app_subtitle": "2B параметров · 30 языков · 48 kHz · Дизайн и клонирование голоса",
        "tab_tts": "Текст в речь",
        "tab_design": "Дизайн голоса",
        "tab_clone": "Клонирование",
        "tab_ultimate": "Ultimate-клонирование",
        "tts_instructions": "Введите любой текст — модель синтезирует речь с естественной просодией. 30 языков: русский, английский, китайский и др.",
        "design_instructions": "Создайте уникальный голос из текстового описания — без референс-аудио. Опишите пол, возраст, тон, эмоцию, темп, акцент.",
        "clone_instructions": "Загрузите короткий референс (5-30 сек, до 50 сек) для клонирования голоса. Опционально задайте стиль: эмоция, темп.",
        "ultimate_instructions": "Референс-аудио + его точный транскрипт для максимальной верности клонирования. Модель продолжает референс, сохраняя все детали голоса.",
        "text_label": "Текст",
        "text_placeholder": "Введите текст на любом из 30 поддерживаемых языков...",
        "description_label": "Описание голоса",
        "description_placeholder": "например: Молодая женщина, нежный и мягкий голос",
        "content_label": "Текст для озвучки",
        "content_placeholder": "Привет, добро пожаловать в VoxCPM2!",
        "reference_label": "Референс-аудио",
        "style_label": "Описание стиля (опционально)",
        "style_placeholder": "например: чуть быстрее, бодрым тоном",
        "transcript_label": "Транскрипт референс-аудио",
        "transcript_placeholder": "Точный текст того, что говорится в референс-аудио",
        "cfg_label": "CFG Scale",
        "cfg_info": "Выше = ближе к промпту, ниже = больше креатива",
        "steps_label": "Шаги диффузии",
        "steps_info": "Больше = качественнее, но медленнее (5-10 для скорости)",
        "seed_label": "Seed",
        "lock_label": "Зафиксировать Seed",
        "advanced_label": "Расширенные настройки",
        "normalize_label": "Нормализация текста (wetext)",
        "normalize_info": "Обработка чисел, дат, сокращений",
        "denoise_label": "Шумоподавление референса",
        "denoise_info": "ZipEnhancer денойзинг референс-аудио",
        "retry_label": "Повтор при плохой генерации",
        "retry_info": "Перегенерировать если качество плохое",
        "generate_tts": "Синтезировать",
        "generate_design": "Создать голос",
        "generate_clone": "Клонировать",
        "generate_ultimate": "Ultimate-клонировать",
        "output_label": "Результат",
        "device_label": "Устройство",
    },
}
# Aliases для русского
_I18N_TRANSLATIONS["ru-RU"] = _I18N_TRANSLATIONS["ru"]

I18N = gr.I18n(**_I18N_TRANSLATIONS)


# === Примеры (gr.Examples) ===
TTS_EXAMPLES = [
    ["Привет! Это портативная версия VoxCPM2 от Nerual Dreming и Нейро-Софт."],
    ["Hello! This is the portable VoxCPM2 build."],
    ["Сегодня прекрасная погода, солнце светит ярко, и птицы поют в парке."],
    ["The quick brown fox jumps over the lazy dog."],
]

DESIGN_EXAMPLES = [
    ["Молодая женщина, нежный и мягкий голос", "Привет, добро пожаловать в VoxCPM2!"],
    ["Пожилой мужчина с глубоким баритоном, говорит медленно и внушительно", "Давным-давно, в далёкой галактике..."],
    ["Весёлый ребёнок, энергично и быстро", "Ура! Сегодня выходной!"],
    ["A young woman with a soft gentle voice", "Hello, welcome to VoxCPM2!"],
    ["An elderly British man, deep and authoritative", "Once upon a time, in a galaxy far far away..."],
    ["A cheerful child, energetic and playful", "Yay! It's weekend today!"],
]

CLONE_STYLE_EXAMPLES = [
    "чуть быстрее, бодрым тоном",
    "медленно и драматично",
    "шёпотом, интимно",
    "slightly faster, cheerful tone",
    "slow and dramatic",
    "whispering, intimate",
]


# === CSS (тёмная тема + gradient header) ===
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.gradio-container { max-width: 1200px !important; margin: auto !important; }

.brand-header {
  text-align: center;
  background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #7e22ce 100%);
  padding: 28px 20px;
  border-radius: 16px;
  margin: 8px 0 16px 0;
  box-shadow: 0 10px 30px rgba(109, 40, 217, 0.35);
  color: white;
}
.brand-title { font-size: 1.9em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { font-size: 1em; opacity: 0.9; margin-bottom: 14px; }
.brand-credits { font-size: 0.9em; opacity: 0.95; }
.brand-credits a { color: #fbbf24; text-decoration: none; font-weight: 600; }
.brand-credits a:hover { text-decoration: underline; }
.device-badge {
  display: inline-block;
  background: rgba(255,255,255,0.15);
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.85em;
  margin-top: 10px;
}

button.primary {
  background: linear-gradient(135deg, #6d28d9 0%, #7e22ce 100%) !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}
"""

# === JS: принудительно тёмная тема (одноразовый check на load) ===
_HEAD_JS = """
function() {
  const url = new URL(window.location);
  if (!url.searchParams.has('__theme')) {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.toString());
  }
}
"""

_BRAND_HTML = f"""
<div class="brand-header">
  <div class="brand-title">🎙️ VoxCPM2 — Multilingual TTS (RU/EN)</div>
  <div class="brand-subtitle">2B параметров · 30 языков · 48 kHz · Voice Design &amp; Cloning</div>
  <div class="brand-credits">
    Портативная сборка:
    <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> ·
    <a href="https://neuro-cartel.com" target="_blank">neuro-cartel.com</a> ·
    <a href="https://t.me/neuroport" target="_blank">Нейро-Софт</a>
  </div>
  <div class="device-badge">💻 {DEVICE_INFO}</div>
</div>
"""


# === UI Builder ===
def _seed_row():
    with gr.Row():
        seed = gr.Number(value=-1, label=I18N("seed_label"), precision=0, scale=3)
        locked = gr.Checkbox(value=False, label=I18N("lock_label"), scale=1)
    return seed, locked


def _advanced_block(show_denoise: bool = False):
    """Accordion с расширенными параметрами."""
    with gr.Accordion(label=I18N("advanced_label"), open=False):
        with gr.Row():
            cfg = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label=I18N("cfg_label"), info=I18N("cfg_info"))
            steps = gr.Slider(5, 30, value=10, step=1, label=I18N("steps_label"), info=I18N("steps_info"))
        normalize = gr.Checkbox(value=True, label=I18N("normalize_label"), info=I18N("normalize_info"))
        retry = gr.Checkbox(value=False, label=I18N("retry_label"), info=I18N("retry_info"))
        denoise = None
        if show_denoise:
            denoise = gr.Checkbox(value=False, label=I18N("denoise_label"), info=I18N("denoise_info"))
    return cfg, steps, normalize, retry, denoise


def build_ui():
    with gr.Blocks() as demo:
        gr.HTML(_BRAND_HTML)

        # === Таб 1: TTS ===
        with gr.Tab(label=I18N("tab_tts")):
            gr.Markdown(I18N("tts_instructions"))
            with gr.Row():
                with gr.Column(scale=3):
                    tts_text = gr.Textbox(label=I18N("text_label"), placeholder=I18N("text_placeholder"), lines=4)
                    tts_cfg, tts_steps, tts_norm, tts_retry, _ = _advanced_block(show_denoise=False)
                    tts_seed, tts_locked = _seed_row()
                    tts_btn = gr.Button(I18N("generate_tts"), variant="primary", size="lg")
                with gr.Column(scale=2):
                    tts_out = gr.Audio(label=I18N("output_label"), type="filepath")
            gr.Examples(examples=TTS_EXAMPLES, inputs=[tts_text], label="Примеры / Examples", examples_per_page=10)
            tts_btn.click(
                tts_generate,
                inputs=[tts_text, tts_cfg, tts_steps, tts_seed, tts_locked, tts_norm, tts_retry],
                outputs=[tts_out, tts_seed],
            )

        # === Таб 2: Voice Design ===
        with gr.Tab(label=I18N("tab_design")):
            gr.Markdown(I18N("design_instructions"))
            with gr.Row():
                with gr.Column(scale=3):
                    vd_desc = gr.Textbox(label=I18N("description_label"), placeholder=I18N("description_placeholder"), lines=2)
                    vd_text = gr.Textbox(label=I18N("content_label"), placeholder=I18N("content_placeholder"), lines=3)
                    vd_cfg, vd_steps, vd_norm, vd_retry, _ = _advanced_block(show_denoise=False)
                    vd_seed, vd_locked = _seed_row()
                    vd_btn = gr.Button(I18N("generate_design"), variant="primary", size="lg")
                with gr.Column(scale=2):
                    vd_out = gr.Audio(label=I18N("output_label"), type="filepath")
            # Voice Design examples — кнопки, заполняющие оба поля (gr.Examples не резолвит gr.I18n в заголовках)
            gr.Markdown("**Примеры описаний голоса / Voice design examples** (кликни, чтобы подставить):")
            with gr.Row():
                for i, (desc, txt) in enumerate(DESIGN_EXAMPLES):
                    ex_btn = gr.Button(desc[:40] + ("…" if len(desc) > 40 else ""), size="sm")
                    ex_btn.click(
                        fn=lambda d=desc, t=txt: (d, t),
                        inputs=[],
                        outputs=[vd_desc, vd_text],
                    )
            vd_btn.click(
                voice_design,
                inputs=[vd_desc, vd_text, vd_cfg, vd_steps, vd_seed, vd_locked, vd_norm, vd_retry],
                outputs=[vd_out, vd_seed],
            )

        # === Таб 3: Voice Cloning ===
        with gr.Tab(label=I18N("tab_clone")):
            gr.Markdown(I18N("clone_instructions"))
            with gr.Row():
                with gr.Column(scale=3):
                    # Пак голосов — Slait/russia_voices (743 русских голоса)
                    with gr.Accordion("📦 Пак русских голосов (Slait/russia_voices)", open=False):
                        with gr.Row():
                            vc_voice_pick = gr.Dropdown(
                                label="Выбрать голос из локального пака",
                                choices=scan_local_voices(),
                                value=None,
                                interactive=True,
                                filterable=True,
                                scale=3,
                            )
                            vc_refresh_btn = gr.Button("🔄 Обновить", size="sm", scale=1)
                        with gr.Row():
                            vc_download_btn = gr.Button("📥 Скачать все 743 голоса (~1.5 GB)", size="sm", scale=3)
                            vc_pack_status = gr.Textbox(label="Статус", interactive=False, scale=2)
                    vc_ref = gr.Audio(label=I18N("reference_label"), type="filepath", sources=["upload", "microphone"])
                    vc_transcript = gr.Textbox(
                        label="Транскрипт референса (опционально — для макс. качества, автозаполняется из пака)",
                        placeholder="Точный текст того что говорится в референс-аудио",
                        lines=2,
                    )
                    vc_text = gr.Textbox(label=I18N("content_label"), placeholder=I18N("content_placeholder"), lines=3)
                    vc_style = gr.Dropdown(
                        label=I18N("style_label"),
                        choices=CLONE_STYLE_EXAMPLES,
                        value="",
                        allow_custom_value=True,
                        filterable=True,
                    )
                    vc_cfg, vc_steps, vc_norm, vc_retry, vc_denoise = _advanced_block(show_denoise=True)
                    vc_seed, vc_locked = _seed_row()
                    vc_btn = gr.Button(I18N("generate_clone"), variant="primary", size="lg")
                with gr.Column(scale=2):
                    vc_out = gr.Audio(label=I18N("output_label"), type="filepath")
            vc_btn.click(
                voice_clone,
                inputs=[vc_text, vc_ref, vc_style, vc_transcript, vc_cfg, vc_steps, vc_seed, vc_locked, vc_norm, vc_denoise, vc_retry],
                outputs=[vc_out, vc_seed],
            )
            # Voice pack handlers (Voice Cloning) — заполняет И аудио, И транскрипт из пака
            vc_voice_pick.change(
                fn=lambda n: (voice_audio_path(n) if n else None, voice_transcript(n) if n else ""),
                inputs=[vc_voice_pick],
                outputs=[vc_ref, vc_transcript],
            )
            vc_refresh_btn.click(
                fn=lambda: gr.update(choices=scan_local_voices()),
                inputs=[],
                outputs=[vc_voice_pick],
            )
            vc_download_btn.click(
                fn=download_all_cloud_voices,
                inputs=[],
                outputs=[vc_pack_status],
            ).then(
                fn=lambda: gr.update(choices=scan_local_voices()),
                inputs=[],
                outputs=[vc_voice_pick],
            )

        # === Таб 4: Ultimate Cloning ===
        with gr.Tab(label=I18N("tab_ultimate")):
            gr.Markdown(I18N("ultimate_instructions"))
            with gr.Row():
                with gr.Column(scale=3):
                    # Пак голосов с автозаполнением транскрипта
                    with gr.Accordion("📦 Пак русских голосов (Slait/russia_voices)", open=False):
                        with gr.Row():
                            uc_voice_pick = gr.Dropdown(
                                label="Выбрать голос из локального пака (автозаполнение транскрипта)",
                                choices=scan_local_voices(),
                                value=None,
                                interactive=True,
                                filterable=True,
                                scale=3,
                            )
                            uc_refresh_btn = gr.Button("🔄 Обновить", size="sm", scale=1)
                        with gr.Row():
                            uc_download_btn = gr.Button("📥 Скачать все 743 голоса (~1.5 GB)", size="sm", scale=3)
                            uc_pack_status = gr.Textbox(label="Статус", interactive=False, scale=2)
                    uc_ref = gr.Audio(label=I18N("reference_label"), type="filepath", sources=["upload", "microphone"])
                    uc_transcript = gr.Textbox(label=I18N("transcript_label"), placeholder=I18N("transcript_placeholder"), lines=2)
                    uc_text = gr.Textbox(label=I18N("content_label"), placeholder=I18N("content_placeholder"), lines=3)
                    uc_cfg, uc_steps, uc_norm, uc_retry, uc_denoise = _advanced_block(show_denoise=True)
                    uc_seed, uc_locked = _seed_row()
                    uc_btn = gr.Button(I18N("generate_ultimate"), variant="primary", size="lg")
                with gr.Column(scale=2):
                    uc_out = gr.Audio(label=I18N("output_label"), type="filepath")
            uc_btn.click(
                ultimate_clone,
                inputs=[uc_text, uc_ref, uc_transcript, uc_cfg, uc_steps, uc_seed, uc_locked, uc_norm, uc_denoise, uc_retry],
                outputs=[uc_out, uc_seed],
            )
            # Voice pack handlers (Ultimate Cloning — заполняет И аудио, И транскрипт)
            uc_voice_pick.change(
                fn=lambda n: (voice_audio_path(n) if n else None, voice_transcript(n) if n else ""),
                inputs=[uc_voice_pick],
                outputs=[uc_ref, uc_transcript],
            )
            uc_refresh_btn.click(
                fn=lambda: gr.update(choices=scan_local_voices()),
                inputs=[],
                outputs=[uc_voice_pick],
            )
            uc_download_btn.click(
                fn=download_all_cloud_voices,
                inputs=[],
                outputs=[uc_pack_status],
            ).then(
                fn=lambda: gr.update(choices=scan_local_voices()),
                inputs=[],
                outputs=[uc_voice_pick],
            )

    return demo


# === Точка входа ===
if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_port=None,
        inbrowser=True,
        i18n=I18N,
        theme=gr.themes.Soft(primary_hue="purple"),
        css=_CSS,
        js=_HEAD_JS,
        show_error=True,
    )
