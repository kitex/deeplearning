import base64
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import requests


def save_image_from_response(response_data, output_dir: Path):
    image_b64 = None

    if isinstance(response_data, dict):
        if isinstance(response_data.get("images"), list) and response_data["images"]:
            image_b64 = response_data["images"][0]
        elif isinstance(response_data.get("image"), str):
            image_b64 = response_data["image"]
        elif isinstance(response_data.get("response"), str):
            image_b64 = response_data["response"]
    elif isinstance(response_data, str):
        image_b64 = response_data

    if isinstance(image_b64, list) and image_b64:
        image_b64 = image_b64[0]

    if isinstance(image_b64, str):
        if image_b64.startswith("data:image/"):
            header, image_b64 = image_b64.split(",", 1)
            mime_type = header.split(";")[0].split(":")[1]
            ext = mime_type.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
        else:
            ext = "png"

        output_path = output_dir / f"generated_image.{ext}"
        with output_path.open("wb") as f:
            f.write(base64.b64decode(image_b64))
        print(f"Saved image to {output_path}")
        return output_path

    raise ValueError("No image data found in the response")


def encode_image_file(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_from_image(
    image_path: str,
    image_strength: float = 0.25,
    seed: int | None = None,
    guidance_scale: float | None = None,
    prompt: str | None = None,
):
    """Send an image to the model endpoint and return the parsed JSON response.

    Parameters:
    - image_path: Path to input image file.
    - image_strength: How strongly to apply the transformation (0.0 = preserve input, 1.0 = ignore input).
    - seed: Optional deterministic seed.
    - guidance_scale: Optional classifier-free guidance / guidance scale.
    - prompt: Optional prompt override.
    """

    img_base64 = encode_image_file(image_path)

    url = "http://localhost:11434/api/generate"
    final_prompt = (
        prompt
        or "generate outline sketch of image"
    )

    payload = {
        "model": "x/flux2-klein:9b",
        "prompt": final_prompt,
        "images": [img_base64],
        "stream": False,
        "image_strength": image_strength,
    }

    if seed is not None:
        payload["seed"] = int(seed)
    if guidance_scale is not None:
        payload["guidance_scale"] = float(guidance_scale)

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()
    return response.json()


root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[
        ("PNG", "*.png"),
        ("JPEG", "*.jpg"),
        ("JPEG", "*.jpeg"),
        ("GIF", "*.gif"),
        ("BMP", "*.bmp"),
        ("WebP", "*.webp"),
        ("All files", "*"),
    ],
)

root.destroy()

if not file_path:
    raise SystemExit("No file selected")

response_data = generate_from_image(file_path)
print(response_data)
save_image_from_response(response_data, Path.cwd())
