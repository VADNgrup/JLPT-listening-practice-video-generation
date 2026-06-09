import os
import json
import base64
from io import BytesIO
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables from .env file
# --- CONFIGURATION ---
MODEL_NAME = os.getenv("IGEN_MODEL_NAME", "runwayml/stable-diffusion-v1-5")
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
SCRIPTS_ROOT = Path(os.getenv("SCRIPTS_OUTPUT_DIR", "outputs/jpc_scripts"))
IMAGES_OUTPUT_DIR = Path(os.getenv("IMAGES_OUTPUT_DIR", "outputs/jpc_scripts/images"))
os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)
DEFAULT_ASPECT_RATIO = "16:9"
ASPECT_RATIOS = {
    "1:1": (1328, 1328), # W:H
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": ( 1080, 810),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
    "2:2": (512, 512),
}

DEFAULT_NEGATIVE_PROMPT = (
    "low resolution, low quality, distorted body, malformed hands, oversaturated, "
)


def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_auth_token=HF_AUTH_TOKEN,
    ).to('cuda')

    # uncomment if you run out of memory
    # pipe.enable_model_cpu_offload()

    return pipe, 'cuda'


def build_visual_prompt(script_data):
    level = script_data.get("level", "")
    grammar_name = script_data.get("grammar_name", "Japanese grammar")
    context = script_data.get("context", "")
    characters = script_data.get("characters", [])
    scripts = script_data.get("scripts", [])

    character_lines = []
    for character in characters[:5]:
        parts = [str(character.get("name", "Unknown"))]
        if character.get("gender"):
            parts.append(str(character["gender"]))
        if character.get("age"):
            parts.append(str(character["age"]))
        if character.get("profession"):
            parts.append(str(character["profession"]))
        character_lines.append(", ".join(parts))

    script_preview = []
    for item in scripts[:6]:
        char_name = item.get("char", "")
        text = item.get("text", "")
        if char_name or text:
            script_preview.append(f"{char_name}: {text}".strip())

    prompt = (
        f"A colorful Anime Image where:  {context}. "
        # f"The image should look like a clean cinematic storyboard frame or learning material illustration, "
        # f"Background and visual element hints: {' | '.join(script_preview)}"
    )
    return prompt


def encode_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_image(pipe, prompt, aspect_ratio=DEFAULT_ASPECT_RATIO):
    width, height = ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS[DEFAULT_ASPECT_RATIO])
    generator = torch.Generator(device='cuda').manual_seed(42)

    result = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=generator,
    )
    return result.images[0]


def process_script_file(pipe, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lecture_id = json_path.parent.name
    script_number = json_path.stem
    image_path = IMAGES_OUTPUT_DIR / f"{lecture_id}_{script_number}.png"

    prompt = build_visual_prompt(data)
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = generate_image(pipe, prompt)
        image.save(image_path)

    image_b64 = encode_image_to_base64(image)

    data["visualization_base64"] = image_b64
    data["visualization_prompt"] = prompt
    data["visualization_image_path"] = str(image_path.as_posix())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return json_path, image_path


def iter_script_files(root_dir):
    if not root_dir.exists():
        return

    for level_dir in sorted(root_dir.iterdir()):
        if not level_dir.is_dir():
            continue
        for grammar_dir in sorted(level_dir.iterdir()):
            if grammar_dir.is_dir():
                for script_file in sorted(grammar_dir.glob("*.json")):
                    yield script_file
            elif grammar_dir.is_file() and grammar_dir.suffix.lower() == ".json":
                yield grammar_dir


def main():

    if not SCRIPTS_ROOT.exists():
        print(f"Scripts folder not found: {SCRIPTS_ROOT}")
        return

    pipe, _ = load_pipeline()
    processed = 0

    for script_file in iter_script_files(SCRIPTS_ROOT):
        try:
            _, image_path = process_script_file(pipe, script_file)
            processed += 1
            print(f"Processed: {script_file} -> {image_path}")
        except Exception as exc:
            print(f"Failed: {script_file} | {exc}")

    print(f"Finished. Updated {processed} script file(s).")


if __name__ == "__main__":
    main()
