from pathlib import Path

from PIL import Image


def load_png_with_transparent_white(path: Path, threshold: int = 245) -> Image.Image:
    with Image.open(path).convert("RGBA") as img:
        data = img.getdata()
        cleaned = []
        for r, g, b, a in data:
            if r > threshold and g > threshold and b > threshold:
                cleaned.append((r, g, b, 0))
            else:
                cleaned.append((r, g, b, a))
        img.putdata(cleaned)
        return img.copy()
