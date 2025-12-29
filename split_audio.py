import subprocess
from pathlib import Path


def split_audio(audio_path, out_dir, chunk_sec):
    Path(out_dir).mkdir(exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(chunk_sec),
        "-c", "copy",
        f"{out_dir}/chunk_%03d.wav"
    ]

    subprocess.run(cmd, check=True)
    return sorted(Path(out_dir).glob("chunk_*.wav"))
