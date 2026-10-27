import json
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 research-agent/1.0"}
CANDS = [
    "organika/sdxl-detector",
    "Ateeqq/ai-vs-human-image-detector",
    "mmanikanta/ConvNeXT_AI_image_detector",
    "mmanikanta/ResNet_AI_image_detector",
    "haywoodsloan/ai-image-detector-deploy",
    "Cyberdon360/ai-vs-real-image-detector",
    "umm-maybe/AI-image-detector",
]
for c in CANDS:
    try:
        info = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(f"https://hf-mirror.com/api/models/{c}", headers=UA), timeout=40
            ).read()
        )
        tags = info.get("tags", [])
        lic = [t for t in tags if t.startswith("license:")]
        print(f"{c:55s} dl={info.get('downloads'):8d} lic={lic} sha={info.get('sha')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{c:55s} FAIL {exc}")
