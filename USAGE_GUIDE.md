# 🍼 BabyVis Usage Guide (v2.1)

This document walks you through installing BabyVis, choosing the generation backend highlighted in the Qwen Image Edit GGUF video, and running the app in web, CLI, or test mode.

## 1. Quick Start

```bash
# 1. (optional) create/activate your Python env
conda activate babyvis  # or source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the web UI (diffusers backend by default)
python main.py --mode web
```

Open `http://localhost:8000` in a browser, upload an ultrasound image, tune settings, and download the generated baby portrait.

## 2. Choosing a Model Backend

BabyVis now supports two backends, mirroring the workflow shown in the video:

| Backend       | When to use | How it works |
|---------------|-------------|--------------|
| `diffusers` *(default)* | You want the official Qwen/Qwen-Image-Edit pipeline with automatic VRAM optimisations. | Downloads the Hugging Face diffusers pipeline, enables attention slicing, CPU offload, and VAE tiling for GPUs around 4 GB just like the tutorial setup with ComfyUI. |
| `gguf`        | You have downloaded a local `QuantStack/Qwen-Image-Edit-GGUF` quantised file (Q4/Q5/Q8) and prefer to stay offline or use llama.cpp compatible runtimes. | Loads the GGUF file via `llama-cpp-python` and wraps it in a simple img2img interface. Falls back to the diffusers backend if the GGUF load fails. |

### 2.1 Diffusers backend (default)

Nothing to configure—just run `python main.py --mode web`. BabyVis will download the `Qwen/Qwen-Image-Edit` repo on the first launch and cache it in `~/.cache/huggingface`.

### 2.2 GGUF backend (video workflow)

1. **Download the GGUF file** (the video uses Q5/Q8). Example:
   ```bash
   huggingface-cli download \
       QuantStack/Qwen-Image-Edit-GGUF \
       Qwen-Image-Edit-Q5_1.gguf \
       --local-dir models/gguf/qwen_image_edit
   ```
2. **Install llama.cpp bindings**:
   ```bash
   pip install llama-cpp-python gguf
   ```
3. **Launch BabyVis with GGUF**:
   ```bash
   BABYVIS_MODEL_PROVIDER=gguf \
   BABYVIS_GGUF_PATH="models/gguf/qwen_image_edit/Qwen-Image-Edit-Q5_1.gguf" \
   BABYVIS_GGUF_QUANT=Q5_1 \
   python main.py --mode web
   ```
   BabyVis will attempt to use the GGUF loader; if it fails (e.g., missing dependencies) it automatically falls back to the diffusers pipeline and logs the reason.

Tip: you can mix flags and environment variables, e.g. `python main.py --mode web --provider gguf --gguf-path ...`.

## 3. Command-line Modes

### 3.1 Web mode
```bash
python main.py --mode web [--host 0.0.0.0] [--port 8000] [--reload]
```

### 3.2 CLI workflow
```bash
python main.py --mode cli
```
Follow the prompts to pick a source ultrasound image and output filename.

### 3.3 Diagnostics / smoke test
```bash
python main.py --mode test
```
Creates a placeholder ultrasound, runs validation, loads the configured backend, performs a short generation, and saves `outputs/test_baby_face.png`.

## 4. Dependency Management

Run a dependency health check any time:
```bash
python main.py --check-deps [--provider gguf]
```

If `diffusers` or other packages are missing, reinstall with `pip install -r requirements.txt`. For GGUF workflows also ensure `llama-cpp-python` and `gguf` are present.

## 5. Environment Cheatsheet

| Variable | Purpose | Default |
|----------|---------|---------|
| `BABYVIS_MODEL_PROVIDER` | `diffusers` or `gguf`. | `diffusers` |
| `QWEN_MODEL_ID` | Hugging Face repo ID (diffusers). | `Qwen/Qwen-Image-Edit` |
| `BABYVIS_DEVICE` | `cuda`, `cpu`, or `auto`. | `auto` |
| `BABYVIS_GGUF_PATH` | Local GGUF file path. | `None` |
| `BABYVIS_GGUF_QUANT` | Quant tag (e.g. `Q5_1`). | `auto` |
| `BABYVIS_DISABLE_CPU_OFFLOAD` | Set to `1` to keep the diffusers model fully on GPU/CPU without offload. | `0` |

## 6. start.sh Helper

A simplified launcher is provided:
```bash
./start.sh --install --provider gguf --gguf-path models/gguf/qwen_image_edit/Qwen-Image-Edit-Q5_1.gguf
```
Flags:
- `--install` – upgrade pip and reinstall dependencies.
- `--provider` – overrides the backend (`diffusers`/`gguf`).
- `--python` / `--venv` – choose interpreter or activate an existing virtualenv.

## 7. Troubleshooting

- **Diffusers import error (`split_torch_state_dict_into_shards`)** – upgrade `huggingface-hub` `>=0.24 <0.37` using `pip install -r requirements.txt`.
- **Missing `diffusers`** – install requirements or run `./start.sh --install`.
- **GGUF loader fails** – check that the GGUF file exists and `llama-cpp-python` is built with CUDA (if desired). Review logs for the fallback message and confirm the diffusers backend is working.
- **CUDA OOM** – lower steps/strength, keep CPU offload enabled, ensure input is 512×512.

Happy baby-face generation! 🎨👶
