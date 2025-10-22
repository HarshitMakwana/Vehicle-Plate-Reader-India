# Vehicle-Plate-Reader-India
Indian Vehicle Plate Reader — YOLO-based plate detection, PaddleOCR recognition, Groq fallback, simple batch processing.
Python-based Automatic Number Plate Recognition (ANPR) tool that:

- Detects license plates in images using `open_image_models` (YOLO-based detector).
- Extracts text from detected plates using `PaddleOCR` locally.
- Falls back to a Groq vision-enabled chat completion when local OCR is insufficient.
- Saves annotated images with bounding boxes and overlayed plate text.

This README explains how to configure, run, and extend the project.

## Features

- Batch-process images from a folder and write annotated outputs to another folder.
- Environment-driven configuration (supports `.env`).
- Console logging with per-image timing and a final processing summary.
- Robust handling for OCR and remote API errors.

## Files

- `main.py` — main script, orchestrates detection, OCR, Groq fallback, and saving annotated images.
- `requirements.txt` — inferred dependencies (remember to install a suitable PaddlePaddle wheel for PaddleOCR).
- `.env.sample` — sample environment variables to copy into `.env`.

## Requirements

- Python 3.8+ recommended.
- The `requirements.txt` contains the main packages. PaddleOCR requires PaddlePaddle; install the appropriate wheel for your CPU or GPU.

## Setup

1. Create & activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Configure environment variables by creating a `.env` in the project root (copy `.env.sample`):

```
copy .env.sample .env
# then edit .env and add your GROQ_API_KEY value
```

Or set variables in PowerShell directly:

```powershell
$env:GROQ_API_KEY = 'gsk_your_key_here'
$env:LPR_INPUT_FOLDER = './collection'
$env:LPR_OUTPUT_FOLDER = './output_images'
$env:LOG_LEVEL = 'INFO'
```

## Usage

Ensure your input images are placed in the input folder (default `./collection`). Then run:

```powershell
python .\main.py
```

You will see console logs for each image processed and a final summary containing processed/skipped/errors and total time.

## Configuration (env vars)

- `GROQ_API_KEY` (required) — API key for Groq (vision-enabled model). Without it, the script exits.
- `LPR_INPUT_FOLDER` (optional) — input folder path (default `./collection`).
- `LPR_OUTPUT_FOLDER` (optional) — output folder path (default `./output_images`).
- `LOG_LEVEL` (optional) — logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) default `INFO`.

## Notes & Troubleshooting

- PaddleOCR requires a matching PaddlePaddle installation. If you see import or runtime errors from Paddle, install a compatible wheel from https://www.paddlepaddle.org.cn/install/quick.
- Groq calls may produce latency or incur costs — the code falls back gracefully when Groq is unavailable.
- If processing large images, consider resizing them before detection to improve performance.

## Next steps (suggested improvements)

- Add CLI flags with `argparse` to override env variables at runtime.
- Add multiprocessing or a worker pool to parallelize image processing.
- Save recognition results (file name, plate text, confidence) to CSV/JSON for downstream use.
- Add unit tests and sample images for CI.

## License

This project does not include a license file. Add one that fits your intended use (MIT/Apache/BSD/etc.).

---

If you want, I can add the README to the repository (done), pin dependency versions in `requirements.txt`, or add CLI flags next. Which should I do next?
