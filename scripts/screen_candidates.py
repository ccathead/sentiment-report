"""Create a local visual screening packet from the pinned MVSA source."""
import argparse
import hashlib
import html
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
GROUPS = {
    "positive_candidates": [224,498,565,570,572,595,822,878,915,968,1096,1239,1325,1491,1554,1559,1790,1898,2795,2919,3161,3720,3737,4770],
    "negative_candidates": [10,118,442,819,901,920,928,929,1309,1512,1627,1676,1958,2691,2746,2747,2931,3280,3547,3850,4385,4387,4440,4532],
    "neutral_candidates": [117,218,314,374,581,621,797,849,1068,1254,1326,1435,1857,2015,2533,2787,2804,2924,2965,3202,3644,3672,3807,5052],
    "challenge_candidates": [1,80,87,659,744,1007,1213,1263,1545,1604,1772,1896],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    out = PROJECT / "work" / "screening_v0_1"
    out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for line in (args.source / "labelResultAll.txt").read_text(encoding="utf-8-sig").splitlines()[1:]:
        key, pair = line.split("\t")
        labels[int(key)] = pair.split(",")
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 20)
    records = []
    for group, ids in GROUPS.items():
        for key in ids:
            raw = (args.source / "data" / f"{key}.txt").read_bytes()
            text = " ".join(html.unescape(raw.decode("utf-8-sig")).split())
            image_path = args.source / "data" / f"{key}.jpg"
            records.append({"source_id": str(key), "screening_group": group,
                            "text": text, "source_text_label": labels[key][0],
                            "source_image_label": labels[key][1],
                            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                            "raw_text_sha256": hashlib.sha256(raw).hexdigest()})
        for offset in range(0, len(ids), 12):
            batch = ids[offset:offset+12]
            sheet = Image.new("RGB", (1440, 1320), "white")
            draw = ImageDraw.Draw(sheet)
            for n, key in enumerate(batch):
                left, top = (n % 4)*360, (n//4)*440
                record = next(x for x in records if x["source_id"] == str(key))
                draw.rectangle((left,top,left+359,top+439),outline="#bbbbbb")
                draw.text((left+10,top+7),f"ID {key}",font=title_font,fill="black")
                with Image.open(args.source / "data" / f"{key}.jpg") as image:
                    thumbnail = ImageOps.contain(image.convert("RGB"),(340,260))
                    sheet.paste(thumbnail,(left+10+(340-thumbnail.width)//2,top+40+(260-thumbnail.height)//2))
                caption = textwrap.fill(record["text"],width=44)
                for line_n, line in enumerate(caption.splitlines()[:7]):
                    draw.text((left+10,top+307+line_n*18),line,font=font,fill="black")
            target = out / f"{group}_{offset//12+1}.jpg"
            sheet.save(target,quality=93)
            print(target)
    (out / "candidates.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Screening candidates: {len(records)}")


if __name__ == "__main__":
    main()
