<div align="center">

# VoxCPM2 Portable

**Портативная Windows-сборка VoxCPM2 — мультиязычный TTS с Voice Design, клонированием голоса и тренировкой LoRA.**

[![Stars](https://img.shields.io/github/stars/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/stargazers)
[![License](https://img.shields.io/github/license/timoncool/VoxCPM2_portable?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/commits/main)
[![Downloads](https://img.shields.io/github/downloads/timoncool/VoxCPM2_portable/total?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/releases)

**[English version](README.md)**

</div>

Синтезируй натуральную речь на 30 языках, создавай уникальные голоса из текстового описания, клонируй любой голос по короткому референсу и **тренируй свою LoRA на 10 аудиоклипах** — **100% локально**, без облака, без API-ключей. Установка в один клик на Windows, работает на любой NVIDIA GPU с 8+ GB VRAM.

Построен на [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) от OpenBMB — токенайзер-свободной 2B модели TTS (diffusion autoregressive), обученной на 2M+ часов речи.

## Почему VoxCPM2 Portable?

- **Бесплатно навсегда** — без API-ключей, кредитов и лимитов
- **Приватно** — твои голосовые данные никуда не уходят
- **Портативно** — всё в одной папке, скопируй на флешку, удалил = деинсталлировал
- **Один клик** — `install.bat` → `run.bat` → синтезируй
- **30 языков** — русский, английский, китайский, французский, немецкий, японский, корейский и другие
- **Тренировка LoRA прямо в UI** — fine-tune на 10-20 клипах за 5-15 мин на RTX 4090

## Возможности

### Синтез речи (Text-to-Speech)
- **30 языков** — RU/EN/ZH (+9 китайских диалектов)/AR/FR/DE/HI/IT/JA/KO/PT/ES и другие — автоопределение языка
- **48 kHz студийное качество** через AudioVAE V2 super-resolution (16→48 kHz)
- **Естественная просодия** — tokenizer-free diffusion autoregressive архитектура
- Форматы вывода: **MP3** (по умолчанию), WAV, FLAC, OGG

### Дизайн голоса (Voice Design)
- Создавай голоса из **текстового описания** — пол, возраст, тон, эмоция, темп, акцент
- Zero-shot — без референс-аудио
- 6 готовых примеров (EN+RU) — клик и заполняется

### Клонирование голоса (с опциональным Ultimate-режимом)
- Клонируй любой голос по **5-50 секундам** референс-аудио
- **Пак голосов** из коробки (~100 голосов: RU/EN/FR/DE/JP/KR/AR)
- **743 дополнительных русских голоса** по запросу из `Slait/russia_voices`
- Контроль стиля: `чуть быстрее, бодрым тоном` / `шёпотом, интимно` / `медленно и драматично`
- **Ultimate-режим** — заполни транскрипт → модель использует `prompt_wav_path + prompt_text + reference_wav_path` для максимальной верности
- Опциональное **ZipEnhancer шумоподавление** для шумных референсов

### LoRA Fine-Tuning прямо в UI!
- Тренировка на **10-20 клипах** (~5-10 мин аудио) → адаптация к голосу за 5-15 мин на RTX 4090
- Встроенный training script от OpenBMB (`training/scripts/train_voxcpm_finetune.py`)
- Live-прогресс в UI
- Сохраняется в `lora/<имя>/step_XXXX/` — сразу появляется в dropdown'е
- **Hot-swap**: переключение между LoRA без перезапуска через `unload_lora + load_lora`
- Работает во всех режимах (TTS / Voice Design / Voice Cloning)

### Все параметры модели выведены в UI
CFG Scale · Inference Steps · Min/Max длина · Retry при плохой генерации · Макс. повторов · Порог плохой генерации · Нормализация текста (wetext) · Денойзинг референса · Streaming (live-прогресс) · Seed + Lock

### Интерфейс
- **i18n RU/EN** через `gr.I18n` — переключение по языку браузера
- **Тёмная тема** с gradient header
- **FFmpeg в комплекте** (для кодирования MP3/OGG)
- **Автозагрузка** — модель (~4-5 GB) + пак голосов при первом запуске
- **Auto-port, auto-browser** — открывается на `localhost` автоматически

### GPU-ускорители (из коробки)
| GPU | Flash Attention 2 | SDPA flash | bfloat16 | AMP (тренировка) |
|-----|:---:|:---:|:---:|:---:|
| RTX 30xx / 40xx / 50xx | ✅ | ✅ | ✅ | ✅ |
| GTX 10xx / RTX 20xx | ❌ | ✅ | ✅ | ✅ |

## Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| GPU VRAM | 8 GB | 12+ GB |
| RAM | 16 GB | 32 GB |
| Диск | 15 GB | 30 GB (с пакетами голосов и LoRA) |
| ОС | Windows 10/11 | Windows 11 |
| GPU | RTX 2080 / RTX 3060 | RTX 4070+ |

CPU-режим поддерживается, но **очень медленный** (минуты на фразу).

## Быстрый старт

### 1. Клонируй
```bash
git clone https://github.com/timoncool/VoxCPM2_portable.git
cd VoxCPM2_portable
```

### 2. Установи
```
install.bat
```
Выбери тип GPU (6 вариантов). Устанавливает портативный Python 3.12 + PyTorch 2.7.1 + voxcpm + Flash Attention 2 + FFmpeg + дефолтный пак голосов. Ничего в системе.

### 3. Запусти
```
run.bat
```
Браузер откроется сам. Модель скачается при первом запуске (~4-5 GB в `models/`).

## Батники

| Скрипт | Описание |
|--------|----------|
| `install.bat` | Установщик в один клик — Python + PyTorch + voxcpm + ускорители + FFmpeg + пак голосов |
| `run.bat` | Запуск Gradio UI с полной изоляцией окружения |
| `update.bat` | Обновление портативки + voxcpm package |

## Workflow тренировки LoRA

1. **Подготовь датасет**: 10-50 WAV/MP3 клипов (3-15 сек каждый) **одного** голоса + транскрипты
2. Открой таб **LoRA** в UI
3. Перетащи аудио-файлы
4. Вставь транскрипты (формат: `имя_файла.wav|точный текст`, по строке)
5. Задай **Имя** для своей LoRA
6. Дефолты под **10-20 клипов**: `r=16`, `alpha=16`, `steps=300`, `lr=0.0001`
7. Жми **🎓 Начать обучение** — ждёшь 5-15 мин на RTX 4090
8. Чекпоинт появляется в `lora/<имя>/step_XXXX/` и в dropdown во всех табах
9. Выбери свою LoRA в **Расширенные настройки → 🧬 LoRA** для активации (первая активация ~30-60 сек — модель переинициализируется с LoRA-структурой; следующие переключения — мгновенный hot-swap)

### Рекомендуемые настройки по размеру датасета

| Клипов | Steps | r / α | LR |
|---|---|---|---|
| **10-20** (дефолт) | **300** | 16/16 | 0.0001 |
| 50-100 | 500-1000 | 32/32 | 0.0001 |
| 200+ | 1000-2000 | 32/32 или 64/64 | 0.00005-0.0001 |
| 1000+ | 2000-5000 | 64/64 | 0.00005 |

## Структура

```
VoxCPM2_portable/
├── app.py              # Gradio UI (4 таба: TTS / Voice Design / Клонирование / LoRA)
├── install.bat         # Выбор GPU + установщик
├── run.bat             # Запуск с изоляцией окружения
├── update.bat          # Обновление
├── requirements.txt    # Python-зависимости
├── training/
│   ├── scripts/        # Официальные OpenBMB train & inference скрипты (встроены)
│   └── conf/           # Шаблоны YAML-конфигов
├── python/             # Портативный Python 3.12 (создаётся install.bat)
├── models/             # HuggingFace кэш (VoxCPM2 ~4-5 GB, ZipEnhancer, SenseVoice)
├── voices/             # Пак голосов (дефолтные ~100 + пользовательские)
├── lora/               # Обученные LoRA чекпоинты (lora/<имя>/step_XXXX/)
├── train_data/         # Пользовательские датасеты для LoRA (аудио + транскрипты)
├── ffmpeg/             # Портативный FFmpeg (для кодирования MP3/OGG)
├── output/             # Сгенерированные аудио с таймстампами
├── cache/ / temp/      # Общий кэш / tempdir
```

## Обновление

```
update.bat
```

## Ссылки

- [OpenBMB / VoxCPM](https://github.com/OpenBMB/VoxCPM) — оригинальный проект
- [VoxCPM2 карточка модели](https://huggingface.co/openbmb/VoxCPM2) — веса
- [Демо-страница с примерами](https://openbmb.github.io/voxcpm2-demopage/)
- [Гайд по Fine-tuning](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)

## Другие портативные нейросети

| Проект | Описание |
|--------|----------|
| [ACE-Step Studio](https://github.com/timoncool/ACE-Step-Studio) | Локальная AI-студия для генерации музыки |
| [Foundation Music Lab](https://github.com/timoncool/Foundation-Music-Lab) | Генерация музыки + редактор таймлайна |
| [VibeVoice ASR](https://github.com/timoncool/VibeVoice_ASR_portable_ru) | Распознавание речи (ASR) |
| [LavaSR](https://github.com/timoncool/LavaSR_portable_ru) | Улучшение качества звука |
| [Qwen3-TTS](https://github.com/timoncool/Qwen3-TTS_portable_rus) | Синтез речи от Qwen |
| [SuperCaption Qwen3-VL](https://github.com/timoncool/SuperCaption_Qwen3-VL) | Описание изображений |
| [VideoSOS](https://github.com/timoncool/videosos) | AI-видеопродакшн |
| [RC Stable Audio Tools](https://github.com/timoncool/RC-stable-audio-tools-portable) | Генерация музыки и звуков |

## Авторы

- **Nerual Dreming** — [Telegram](https://t.me/nerual_dreming) | [neuro-cartel.com](https://neuro-cartel.com) | [ArtGeneration.me](https://artgeneration.me)
- **Нейро-Софт** — [Telegram](https://t.me/neuroport) | портативные нейросети

## Благодарности

- **[Команда OpenBMB / VoxCPM](https://github.com/OpenBMB/VoxCPM)** — открытая модель VoxCPM2
- **[Slait/russia_voices](https://huggingface.co/datasets/Slait/russia_voices)** — 743 русских голоса
- **[AIQuest Academy](https://github.com/TeamAIQ/Colab-notebooks)** — референс Colab UI
- **[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel)** — Windows wheels для Flash Attention 2
- **[Gradio](https://gradio.app/)** — UI фреймворк
- **[FFmpeg](https://ffmpeg.org/)** — кодирование аудио

## Поддержать проект

Я создаю софт и занимаюсь исследованиями в области ИИ. Большая часть всего, что я делаю, находится в открытом доступе. Ваши пожертвования позволяют мне создавать и исследовать больше, не отвлекаясь на поиск еды для продолжения существования =)

**[Все способы поддержки](https://dalink.to/nerual_dreming)** | **[dalink.to/nerual_dreming](https://dalink.to/nerual_dreming)** | **[boosty.to/neuro_art](https://boosty.to/neuro_art)**

- **BTC:** `1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC`
- **ETH (ERC20):** `0xb5db65adf478983186d4897ba92fe2c25c594a0c`
- **USDT (TRC20):** `TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C`

---

## Star History

<a href="https://www.star-history.com/?repos=timoncool%2FVoxCPM2_portable&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&legend=top-left" />
 </picture>
</a>
