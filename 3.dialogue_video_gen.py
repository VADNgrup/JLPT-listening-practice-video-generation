import argparse
import base64
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

import soundfile as sf
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from omnivoice import OmniVoice

# --- CONFIGURATION ---
VOICE_MODEL_NAME = "k2-fsa/OmniVoice"
VOICE_DEVICE = "cuda:0"
VOICE_DTYPE = torch.float16
SCRIPTS_ROOT = Path(os.getenv("SCRIPTS_OUTPUT_DIR", "outputs/jpc_scripts"))
VIDEO_OUTPUT_ROOT = Path(os.getenv("VIDEO_OUTPUT_DIR", "outputs/jpc_scripts/videos"))
VIDEO_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
VOICE_SOURCE_ROOT = Path(os.getenv("VOICE_SOURCE_ROOT", "ingredients/voices"))
SUPPORTED_TTS_TAGS = [
    "[laughter]",
    "[confirmation-en]",
    "[surprise-oh]",
]

ALLOWED_TTS_TAGS = set(SUPPORTED_TTS_TAGS)
ALL_TAG_PATTERN = re.compile(r"\[[^\]]+\]")
DISPLAY_TAG_PATTERN = re.compile(r"\[(?:laughter|sigh|confirmation-en)\]")
WHITESPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([、。！？,.!?])")
BASE_ASPECT_SIZE = (1920, 1080)
PANEL_MARGIN = 70
PANEL_HEIGHT = 330
PANEL_BG = (10, 12, 20, 190)
PANEL_BORDER = (255, 255, 255, 32)
ACCENT = (255, 214, 102, 255)
TEXT_COLOR = (245, 246, 250, 255)
SECONDARY_TEXT_COLOR = (210, 214, 224, 255)
VIDEO_FPS = 30
VIDEO_CRF = 18
VIDEO_PRESET = "slow"
AUDIO_RATE = 48000
AUDIO_BITRATE = "128k"
PAUSE_SECONDS_RANGE = (0.75, 1.5)
AUDIO_SAMPLE_RATE = 24000
AUDIO_TRIM_THRESHOLD = 0.001
AUDIO_TRIM_PAD_SEC = 0.05
VOICE_PROFILES = [
    {
        "ref_audio": VOICE_SOURCE_ROOT / "voice_preview_mitsuki.mp3",
        "name": "Mitsuki",
        "Gender": "female",
        "ref_text": "みんな、今日も元気にしてる。水木の声が君の毎日をちょっぴり特別にできたら嬉しいな。小さな夢もいつ叶えられるよ。さあ、一緒に声で魔法をかけちゃお。",
    },
    {
        "ref_audio": VOICE_SOURCE_ROOT / "voice_preview_hinata.mp3",
        "name": "Hinata",
        "Gender": "male",
        "ref_text": "スイスの観光地として人気の高い、ユングフラウ鉄道。ここにはヨーロッパでとても高い場所にある駅があります。アルプスの雄大な景色を眺めながら、列車は終点の駅を目指し走り出します。",
    },
    {
        "ref_audio": VOICE_SOURCE_ROOT / "voice_preview_koichi.mp3",
        "name": "Koichi",
        "Gender": "male",
        "ref_text": "こんばんは。中低音の落ち着きのあるこの声を使って、いろいろ声で遊んでみてください。ナレーションからキャラクターボイスまで対応しております。皆さんのご利用お待ちしております。",
    },
    # {
    #     "ref_audio": VOICE_SOURCE_ROOT / "voice_preview_fumi.mp3",
    #     "name": "Fumi",
    #     "Gender": "female",
    #     "ref_text": "こんにちは。優しく落ち着いた声で、聞く人に安心感や前向きな気持ちを届けます。ナレーションやガイドサポート音声など、聞きやすさと信頼感を大切にしています。ぜひお任せください。",
    # },
    {
        "ref_audio": VOICE_SOURCE_ROOT / "voice_preview_akira.mp3",
        "name": "Akira",
        "Gender": "male",
        "ref_text": "こんにちは。明です。この声は落ち着いたトーンでありながら、感情の強弱や自然な問を大切にした日本語ナレーションに適しています。優しい説明から力強い表現まで幅広く対応できます。",
    },
    {
        "ref_audio": VOICE_SOURCE_ROOT / "voice_preview_natsu.mp3",
        "name": "Natsu",
        "Gender": "female",
        "ref_text": "こんにちは。日本人女性の落ち着いた声です。物語やドキュメンタリーレクチャーなど、様々なコアに適していると思います。声には説得力があると思いますので、教育用の教材に利用してください。",
    },
]




def load_voice_model():
    return OmniVoice.from_pretrained(
        VOICE_MODEL_NAME,
        device_map=VOICE_DEVICE,
        dtype=VOICE_DTYPE,
    )


def decode_base64_image(image_b64):
    raw = base64.b64decode(image_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def find_font_path():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMonoCJK-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # Fallback: search for any CJK font
    import subprocess
    try:
        result = subprocess.run(
            ["find", "/usr/share/fonts", "-name", "*CJK*.ttf", "-o", "-name", "*CJK*.ttc"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout:
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass
    return None


def load_font(size, bold=False):
    font_path = find_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception as e:
            print(f"Warning: Failed to load font from {font_path}: {e}")
    print(f"Warning: No suitable font found for CJK text. Install fonts: apt-get install fonts-noto-cjk")
    return ImageFont.load_default()


def collapse_whitespace(text):
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def strip_disallowed_tags(text):
    def _replace(match):
        tag = match.group(0)
        if tag == "[sigh]":
            return ""
        return tag if tag in ALLOWED_TTS_TAGS else ""

    return ALL_TAG_PATTERN.sub(_replace, text)


def sanitize_tts_text(text):
    text = strip_disallowed_tags(text)
    return collapse_whitespace(text)


def strip_all_tags_for_display(text):
    text = ALL_TAG_PATTERN.sub("", text)
    return collapse_whitespace(text)


def wrap_text(text, font, max_width, draw):
    lines = []
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue

        current = ""
        for char in paragraph:
            trial = current + char
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def build_character_voice_map(characters):
    female_profiles = [profile for profile in VOICE_PROFILES if profile.get("Gender", "").lower() == "female"]
    male_profiles = [profile for profile in VOICE_PROFILES if profile.get("Gender", "").lower() == "male"]
    random.shuffle(female_profiles)
    random.shuffle(male_profiles)

    female_index = 0
    male_index = 0
    voice_map = {}

    for character in characters:
        character_name = character.get("name", "Unknown")
        gender = str(character.get("gender", "")).lower()
        if gender.startswith("f") and female_profiles:
            profile = female_profiles[female_index % len(female_profiles)]
            female_index += 1
        elif gender.startswith("m") and male_profiles:
            profile = male_profiles[male_index % len(male_profiles)]
            male_index += 1
        else:
            profile = random.choice(VOICE_PROFILES)

        voice_map[character_name] = profile

    return voice_map


def generate_voice_audio(voice_model, text, voice_profile, output_wav_path):
    audio = voice_model.generate(
        text=text,
        ref_audio=str(voice_profile["ref_audio"]),
        ref_text=voice_profile["ref_text"],
    )
    samples = audio[0]
    trimmed = trim_audio_samples(samples, AUDIO_SAMPLE_RATE)
    sf.write(str(output_wav_path), trimmed, AUDIO_SAMPLE_RATE)
    return float(len(trimmed)) / float(AUDIO_SAMPLE_RATE)


def trim_audio_samples(samples, sample_rate):
    tensor = torch.as_tensor(samples).float().flatten()
    if tensor.numel() == 0:
        return samples

    threshold = AUDIO_TRIM_THRESHOLD
    active = torch.nonzero(torch.abs(tensor) > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return samples

    pad = int(AUDIO_TRIM_PAD_SEC * sample_rate)
    start = max(int(active[0]) - pad, 0)
    end = min(int(active[-1]) + pad, tensor.numel() - 1)
    return tensor[start:end + 1].cpu().numpy()


def compute_layout(aspect_size):
    scale_x = aspect_size[0] / BASE_ASPECT_SIZE[0]
    scale_y = aspect_size[1] / BASE_ASPECT_SIZE[1]
    scale = min(scale_x, scale_y)

    panel_margin = int(PANEL_MARGIN * scale)
    panel_height = int(PANEL_HEIGHT * scale)

    return {
        "panel_margin": panel_margin,
        "panel_height": panel_height,
        "scale": scale,
    }


def compose_turn_frame(background_image, speaker_name, display_text, turn_index, total_turns, aspect_size):
    layout = compute_layout(aspect_size)
    panel_margin = layout["panel_margin"]
    panel_height = layout["panel_height"]

    background_canvas = ImageOps.fit(background_image, aspect_size, method=Image.Resampling.LANCZOS)
    canvas = background_canvas.convert("RGBA")
    overlay = Image.new("RGBA", aspect_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    panel_x1 = panel_margin
    panel_y1 = aspect_size[1] - panel_margin - panel_height
    panel_x2 = aspect_size[0] - panel_margin
    panel_y2 = aspect_size[1] - panel_margin

    draw.rounded_rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y2],
        radius=28,
        fill=PANEL_BG,
        outline=PANEL_BORDER,
        width=2,
    )

    title_font = load_font(int(60 * layout["scale"]))
    body_font = load_font(int(34 * layout["scale"]))
    small_font = load_font(int(22 * layout["scale"]))

    header_text = f"{speaker_name}"
    draw.text((panel_x1 + 30, panel_y1 + 24), header_text, font=title_font, fill=ACCENT)

    turn_text = f"{turn_index}/{total_turns}"
    turn_text_width = draw.textlength(turn_text, font=small_font)
    draw.text(
        (panel_x2 - 30 - turn_text_width, panel_y1 + 34),
        turn_text,
        font=small_font,
        fill=SECONDARY_TEXT_COLOR,
    )

    display_lines = wrap_text(display_text, body_font, panel_x2 - panel_x1 - 60, draw)
    max_lines = 4
    if len(display_lines) > max_lines:
        display_lines = display_lines[: max_lines - 1] + ["…"]

    text_y = panel_y1 + 92
    for line in display_lines:
        draw.text((panel_x1 + 30, text_y), line, font=body_font, fill=TEXT_COLOR)
        text_y += 44

    composed = Image.alpha_composite(canvas, overlay).convert("RGB")
    return composed


def compose_pause_frame(background_image, aspect_size):
    layout = compute_layout(aspect_size)
    panel_margin = layout["panel_margin"]
    panel_height = layout["panel_height"]

    background_canvas = ImageOps.fit(background_image, aspect_size, method=Image.Resampling.LANCZOS)
    canvas = background_canvas.convert("RGBA")
    overlay = Image.new("RGBA", aspect_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    panel_x1 = panel_margin
    panel_y1 = aspect_size[1] - panel_margin - panel_height
    panel_x2 = aspect_size[0] - panel_margin
    panel_y2 = aspect_size[1] - panel_margin
    draw.rounded_rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y2],
        radius=28,
        fill=PANEL_BG,
        outline=PANEL_BORDER,
        width=2,
    )
    body_font = load_font(int(34 * layout["scale"]))
    draw.text((panel_x1 + 30, panel_y1 + 40), "", font=body_font, fill=SECONDARY_TEXT_COLOR)
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def ffmpeg_escape(path):
    return path.as_posix().replace("'", "'\\''")


def create_segment_video(frame_path, audio_path, segment_path, duration_seconds=None):
    audio_filter = (
        f"aresample=async=1:first_pts=0,asetpts=N/SR/TB,"
        f"aformat=channel_layouts=stereo:sample_rates={AUDIO_RATE}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(frame_path),
        "-i",
        str(audio_path),
        "-vf",
        f"fps={VIDEO_FPS},setpts=N/({VIDEO_FPS}*TB)",
        "-c:v",
        "libx264",
        "-crf",
        str(VIDEO_CRF),
        "-preset",
        VIDEO_PRESET,
        "-tune",
        "stillimage",
        "-af",
        audio_filter,
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
    ]
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.3f}"])
    command.append(str(segment_path))
    subprocess.run(command, check=True)


def create_silent_audio(audio_path, duration_seconds):
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
        "-t",
        f"{duration_seconds:.2f}",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    subprocess.run(command, check=True)


def concat_segments(segment_paths, output_path, temp_dir):
    concat_file = temp_dir / "concat_list.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for segment_path in segment_paths:
            f.write(f"file '{ffmpeg_escape(segment_path)}'\n")

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-vf",
        f"fps={VIDEO_FPS},setpts=N/({VIDEO_FPS}*TB)",
        "-c:v",
        "libx264",
        "-crf",
        str(VIDEO_CRF),
        "-preset",
        VIDEO_PRESET,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-af",
        f"aresample=async=1:first_pts=0,asetpts=N/SR/TB,aformat=channel_layouts=stereo:sample_rates={AUDIO_RATE}",
        "-shortest",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def process_script_file(voice_model, json_path, regen=False):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not regen and data.get("video_path") and Path(data["video_path"]).exists() and VIDEO_OUTPUT_ROOT in Path(data["video_path"]).parents:
        return data["video_path"]

    image_b64 = data.get("visualization_base64")
    if not image_b64:
        raise ValueError("Missing visualization_base64")

    characters = data.get("characters", [])
    scripts = data.get("scripts", [])
    if not characters or not scripts:
        raise ValueError("Missing characters or scripts")

    level = data.get("level", "Unknown")
    grammar_id = data.get("grammar_id", "unknown")

    voice_map = build_character_voice_map(characters)
    background_image = decode_base64_image(image_b64)
    aspect_size = background_image.size

    output_dir = VIDEO_OUTPUT_ROOT / level / grammar_id
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex
    output_path = output_dir / f"{video_id}.mp4"

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        segment_paths = []
        pause_seconds_list = []
        total_turns = len(scripts)
        timeline_seconds = 0.0

        for index, turn in enumerate(scripts, start=1):
            speaker_name = turn.get("char", "Unknown")
            raw_text = turn.get("text", "")
            tts_text = sanitize_tts_text(raw_text)
            display_text = strip_all_tags_for_display(raw_text)

            voice_profile = voice_map.get(speaker_name)
            if not voice_profile:
                voice_profile = random.choice(VOICE_PROFILES)
                voice_map[speaker_name] = voice_profile

            frame_image = compose_turn_frame(
                background_image,
                speaker_name,
                display_text,
                index,
                total_turns,
                aspect_size,
            )
            frame_path = temp_dir / f"frame_{index:03d}.png"
            audio_path = temp_dir / f"audio_{index:03d}.wav"
            segment_path = temp_dir / f"segment_{index:03d}.mp4"

            frame_image.save(frame_path)
            audio_duration = generate_voice_audio(voice_model, tts_text, voice_profile, audio_path)
            turn["start_time"] = round(timeline_seconds, 3)
            turn["audio_seconds"] = round(audio_duration, 3)
            turn["end_time"] = round(timeline_seconds + audio_duration, 3)
            timeline_seconds += audio_duration
            create_segment_video(frame_path, audio_path, segment_path, duration_seconds=audio_duration)
            segment_paths.append(segment_path)

            if index < total_turns:
                pause_seconds = random.uniform(*PAUSE_SECONDS_RANGE)
                if pause_seconds > 0:
                    pause_seconds_list.append(pause_seconds)
                    turn["pause_seconds"] = round(pause_seconds, 3)
                    timeline_seconds += pause_seconds
                    pause_frame = compose_pause_frame(background_image, aspect_size)
                    pause_frame_path = temp_dir / f"pause_frame_{index:03d}.png"
                    pause_audio_path = temp_dir / f"pause_audio_{index:03d}.wav"
                    pause_segment_path = temp_dir / f"pause_segment_{index:03d}.mp4"

                    pause_frame.save(pause_frame_path)
                    create_silent_audio(pause_audio_path, pause_seconds)
                    create_segment_video(
                        pause_frame_path,
                        pause_audio_path,
                        pause_segment_path,
                        duration_seconds=pause_seconds,
                    )
                    segment_paths.append(pause_segment_path)
                else:
                    turn["pause_seconds"] = 0.0

        if not segment_paths:
            raise ValueError("No segments were generated")

        concat_segments(segment_paths, output_path, temp_dir)

    data["video_id"] = video_id
    data["video_path"] = str(output_path.as_posix())
    data["voice_map"] = {
        name: profile["name"]
        for name, profile in voice_map.items()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return str(output_path.as_posix())


def iter_script_files(root_dir):
    if not root_dir.exists():
        return
    for json_path in root_dir.rglob("*.json"):
        if json_path.is_file():
            yield json_path


def main():
    parser = argparse.ArgumentParser(description="Generate dialogue videos from script JSON files.")
    parser.add_argument("--regen", action="store_true", help="Regenerate videos even when video_path already exists in the JSON")
    args = parser.parse_args()

    if not SCRIPTS_ROOT.exists():
        print(f"Scripts folder not found: {SCRIPTS_ROOT}")
        return

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    voice_model = load_voice_model()
    processed = 0
    failed = 0

    for script_file in iter_script_files(SCRIPTS_ROOT):
        try:
            video_path = process_script_file(voice_model, script_file, regen=args.regen)
            processed += 1
            print(f"Processed: {script_file} -> {video_path}")
        except Exception as exc:
            failed += 1
            print(f"Failed: {script_file} | {exc}")

    print(f"Finished. Updated {processed} script file(s). Failed: {failed}.")


if __name__ == "__main__":
    main()
