"""Run one OCR engine over a set's page images and write what it read.

Step 8 of METHOD.md. Reads <images>/<sha256 first 16>-p<page>.png for each page
of the set (tools/pages.py, tools/draft_c.py) and writes one JSON line per page,
{sha256, page, text, seconds}, the run format tools/score.py reads. Set C pages
the correction dropped or kept spare are not run.

A run stops and resumes: each page is written as soon as it is read, and pages
already in --out are skipped. A page the engine fails on (out of memory) is
left out, so the scorer counts it missing, and is tried again on the next start;
three such pages in a row stop the run, since the settings, not the page, are
then at fault.

Next to the run, <out>.meta.json records what produced it: the engine and model
versions, every setting that changes the output, the machine's GPU and library
versions, and one entry per session with its pages and seconds. Time per page is
only compared between runs of the same machine.

Engines:
  tesseract   CPU; the vie model, page segmentation 3 (automatic).
  qwen        Qwen2.5-VL 7B Instruct through transformers, greedy decoding.
              --weights 8bit: bitsandbytes LLM.int8 for the language model, its
              output layer and the vision encoder left in bfloat16; --weights awq: Qwen's own AWQ
              4-bit release (float16), for a card that cannot hold 8-bit.
              FlashAttention-2 when flash_attn is installed, else PyTorch SDPA,
              whose image encoder holds a full attention matrix (about 3 GB at
              the default cap) and does not fit beside 8-bit weights on 12 GB.
              bitsandbytes runs its 8-bit products in float16 (recorded).
              --max-pixels caps the image the model sees (it is scaled down to
              fit, keeping its shape): the cap sets how small tone marks get.
  paddleocr   PP-OCR through PaddleOCR, lang vi, no page unwarping or orientation
              steps: one line per detected box. --boxes DIR also keeps each page's
              boxes and scores, no text, for vietocr and the abstain signal.
  paddlevl    The PaddleOCR-VL 1.6 pipeline (layout, then the model on each block),
              with weights from a pinned Hugging Face commit, so --model
              VietAlphaLabs/SenOCR-Vi runs that fine-tune the same way. Bfloat16 where
              the card has it; float32 elsewhere does not fit 8 GB.
  vietocr     VietOCR vgg_transformer, which reads single lines, on the boxes
              paddleocr kept (--boxes DIR): both engines see the same detector.

Text is NFC-normalised (syllables.normal); a model's ``` fence around the whole
answer is removed, and a paddlevl table becomes plain rows, as the qwen prompt
asks for. Nothing else is changed.

Usage:
  python -I tools/run_engine.py sets/disclosures-c.jsonl --images cache/set-c \\
      --engine qwen --out data/runs/qwen7b-c.jsonl --machine lab
"""
import argparse
import hashlib
import html
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from records import page_key, page_of, read_jsonl, write_jsonl  # noqa: E402
from syllables import normal  # noqa: E402

PROMPT = ("Transcribe all the text on this page of a Vietnamese document exactly as printed, "
          "keeping every tone mark, digit and punctuation mark. Write plain text, one printed "
          "line per line, in reading order. Where two columns stand side by side, write the left "
          "column first, then the right. Write a table row by row, its cells separated by spaces. "
          "Do not translate, correct, summarise or add anything, and leave out text you cannot read.")
# Kept bfloat16 under --weights 8bit: the vision encoder, and the output layer,
# which bitsandbytes would turn back into a float32 copy (2 GiB) on every page.
SKIP_8BIT = ["visual", "lm_head"]
FENCE_RE = re.compile(r"\A\s*```[a-z]*\n(.*?)\n?```\s*\Z", re.S)
PADDLEVL_REVISIONS = {"PaddlePaddle/PaddleOCR-VL-1.6": "c5630abae1d940eafe0697512a0325494b02ab42",
                      "VietAlphaLabs/SenOCR-Vi": "9c52e6f53ef85434be57f51a8993d7facf12c2f6"}


def version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


class Tesseract:
    name = "tesseract"

    def __init__(self, a):
        if not shutil.which("tesseract"):
            raise SystemExit("tesseract is not installed")
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout
        langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
        if "vie" not in langs.split():
            raise SystemExit("tesseract has no vie model")
        tessdata = re.search(r'"(.+?)"', langs).group(1)
        model = os.path.join(tessdata, "vie.traineddata")
        self.settings = {"version": out.splitlines()[0], "lang": "vie", "psm": 3,
                         "model_bytes": os.path.getsize(model), "model_sha256": sha256(model)}

    def free(self):
        pass

    def read(self, png: str) -> str:
        return subprocess.run(["tesseract", png, "stdout", "-l", "vie", "--psm", "3"],
                              check=True, capture_output=True, text=True).stdout


class Qwen:
    name = "qwen"
    MODELS = {"8bit": "Qwen/Qwen2.5-VL-7B-Instruct", "awq": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"}

    def __init__(self, a):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch, self.max_new = torch, a.max_new_tokens
        model_id = a.model or self.MODELS[a.weights]
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        kw = {"attn_implementation": attn, "device_map": "cuda:0", "revision": a.revision}
        if a.weights == "8bit":
            from transformers import BitsAndBytesConfig
            kw["torch_dtype"] = torch.bfloat16
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP_8BIT)
        else:
            kw["torch_dtype"] = torch.float16       # the AWQ kernels compute in float16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, revision=a.revision,
                                                  min_pixels=a.min_pixels, max_pixels=a.max_pixels)
        self.settings = {"model": model_id,
                         "revision": getattr(self.model.config, "_commit_hash", None) or a.revision,
                         "weights": a.weights,
                         "kept_16bit": SKIP_8BIT if a.weights == "8bit" else None,
                         "int8_matmul": "float16" if a.weights == "8bit" else None,
                         "compute": str(kw["torch_dtype"]).replace("torch.", ""),
                         "attention": attn, "min_pixels": a.min_pixels, "max_pixels": a.max_pixels,
                         "max_new_tokens": a.max_new_tokens, "decoding": "greedy", "prompt": PROMPT}

    def free(self):
        self.torch.cuda.empty_cache()

    def read(self, png: str) -> str:
        from PIL import Image
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(png) as im:
            inputs = self.proc(text=[prompt], images=[im.convert("RGB")], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        new = out[0, inputs["input_ids"].shape[1]:]
        self.cut = len(new) >= self.max_new
        text = self.proc.decode(new, skip_special_tokens=True)
        m = FENCE_RE.match(text)
        return m.group(1) if m else text


class PaddleOcr:
    name = "paddleocr"

    def __init__(self, a):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR
        self.boxes = a.boxes
        if self.boxes:
            os.makedirs(self.boxes, exist_ok=True)
        tuned = {"text_recognition_model_name": "PP-OCRv6_medium_rec",
                 "text_recognition_model_dir": a.rec_model} if a.rec_model else {}
        self.ocr = PaddleOCR(lang="vi", device=a.device, use_doc_orientation_classify=False,
                             use_doc_unwarping=False, use_textline_orientation=False, **tuned)
        pipe = self.ocr.paddlex_pipeline._pipeline
        self.settings = {"lang": "vi", "detector": pipe.text_det_model.model_name,
                         "recognizer": pipe.text_rec_model.model_name,
                         "orientation_and_unwarping": False}
        if a.rec_model:
            with open(os.path.join(a.rec_model, "inference.pdiparams"), "rb") as fh:
                self.settings["recognizer_sha256"] = hashlib.sha256(fh.read()).hexdigest()

    def free(self):
        import paddle
        paddle.device.cuda.empty_cache()

    def read(self, png: str) -> str:
        r = self.ocr.predict(png)[0].json["res"]
        if self.boxes:
            stem = os.path.splitext(os.path.basename(png))[0]
            with open(os.path.join(self.boxes, stem + ".json"), "w", encoding="utf-8") as fh:
                json.dump({"boxes": r["rec_boxes"], "scores": r["rec_scores"]}, fh)
        return "\n".join(r["rec_texts"])


def block_text(content: str) -> str:
    """A pipeline block as plain text: a table's rows on lines, its cells spaced."""
    content = re.sub(r"</t[dh]>", " ", content.replace("</tr>", "\n"))
    return html.unescape(re.sub(r"<[^>]+>", "", content)).strip()


def single_weights_file(path: str) -> str:
    """The pipeline only finds weights named model.safetensors, so a release with one
    shard gets a folder of links beside its snapshot, with that one renamed."""
    if os.path.exists(os.path.join(path, "model.safetensors")):
        return path
    links = path.rstrip("/") + "-single"
    os.makedirs(links, exist_ok=True)
    for name in os.listdir(path):
        if name == "model.safetensors.index.json":
            continue
        new = "model.safetensors" if re.fullmatch(r"model-00001-of-00001\.safetensors", name) else name
        if not os.path.lexists(os.path.join(links, new)):
            os.symlink(os.path.realpath(os.path.join(path, name)), os.path.join(links, new))
    return links


class PaddleVl(PaddleOcr):
    name = "paddlevl"

    def __init__(self, a):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from huggingface_hub import snapshot_download
        from paddleocr import PaddleOCRVL
        model_id = a.model or "PaddlePaddle/PaddleOCR-VL-1.6"
        revision = a.revision or PADDLEVL_REVISIONS.get(model_id)
        if not revision:
            raise SystemExit(f"{model_id}: pin a commit with --revision")
        path = single_weights_file(snapshot_download(model_id, revision=revision))
        self.ocr = PaddleOCRVL(pipeline_version="v1.6", vl_rec_model_dir=path, device=a.device,
                               use_doc_orientation_classify=False, use_doc_unwarping=False)
        self.settings = {"pipeline": "v1.6", "model": model_id, "revision": revision,
                         "orientation_and_unwarping": False}

    def read(self, png: str) -> str:
        r = list(self.ocr.predict(png, use_queues=False))[0].json["res"]
        return "\n".join(t for t in (block_text(b["block_content"]) for b in r["parsing_res_list"]) if t)


class VietOcr:
    name = "vietocr"

    def __init__(self, a):
        import torch
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        if not a.boxes:
            raise SystemExit("vietocr needs --boxes, from a paddleocr run")
        self.torch, self.boxes = torch, a.boxes
        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["device"] = a.device or "cuda:0"
        cfg["cnn"]["pretrained"] = False
        cfg["predictor"]["beamsearch"] = False
        self.model = Predictor(cfg)
        self.settings = {"config": "vgg_transformer", "weights": cfg["weights"], "beamsearch": False,
                         "boxes": "PaddleOCR detector (--boxes)"}

    def free(self):
        self.torch.cuda.empty_cache()

    def read(self, png: str) -> str:
        from PIL import Image
        stem = os.path.splitext(os.path.basename(png))[0]
        with open(os.path.join(self.boxes, stem + ".json"), encoding="utf-8") as fh:
            boxes = json.load(fh)["boxes"]
        with Image.open(png) as im:
            page = im.convert("RGB")
            return "\n".join(self.model.predict(page.crop(tuple(b))) for b in boxes)


ENGINES = {"tesseract": Tesseract, "qwen": Qwen, "paddleocr": PaddleOcr, "paddlevl": PaddleVl, "vietocr": VietOcr}


def sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def machine(label: str) -> dict:
    """The GPU and the libraries that shape a run's output and speed."""
    m = {"label": label, "python": sys.version.split()[0]}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, check=True)
        m["gpu"] = q.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        m["gpu"] = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            m["cpu"] = next((ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")), None)
    except OSError:
        m["cpu"] = None
    libs = ["torch", "transformers", "accelerate", "bitsandbytes", "flash_attn", "autoawq", "pillow",
            "paddlepaddle-gpu", "paddleocr", "paddlex", "vietocr"]
    m["libs"] = {d: v for d in libs if (v := version(d))}
    return m


def done_rows(path: str) -> list[dict]:
    """The rows of an earlier session; a last line cut short by a stop is dropped."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    write_jsonl(path, rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--images", required=True)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine", required=True, help="a label for the machine, e.g. dev or lab")
    ap.add_argument("--limit", type=int, help="stop after this many pages (a trial)")
    ap.add_argument("--device", help="paddleocr, paddlevl, vietocr: gpu:0 (default), or cpu for a trial")
    ap.add_argument("--boxes", help="paddleocr, vietocr: where each page's detected boxes are kept")
    ap.add_argument("--rec-model", help="paddleocr: a fine-tuned recogniser's folder (tools/train_rec.sh)")
    g = ap.add_argument_group("qwen, and --model and --revision for paddlevl")
    g.add_argument("--weights", choices=["8bit", "awq"], default="8bit")
    g.add_argument("--model", help="a model id other than the one --weights picks")
    g.add_argument("--revision", help="a model commit to pin; the one loaded is recorded anyway")
    g.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    g.add_argument("--max-pixels", type=int, default=2560 * 28 * 28)
    g.add_argument("--max-new-tokens", type=int, default=6144)
    a = ap.parse_args()

    entries = [page_of(r) for r in read_jsonl(a.set)]
    entries = [e for e in entries if e.get("fate", "kept") == "kept"]
    missing = [page_key(e) for e in entries if not os.path.exists(os.path.join(a.images, page_key(e) + ".png"))]
    if missing:
        sys.exit(f"{len(missing)} page images missing from {a.images}, e.g. {missing[0]}.png")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    seen = {(r["sha256"], r["page"]) for r in done_rows(a.out)}
    todo = [e for e in entries if (e["sha256"], e["page"]) not in seen]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.engine}: {len(seen)} pages done before, {len(todo)} to read", flush=True)
    if not todo:
        return

    eng = ENGINES[a.engine](a)
    meta_path = a.out + ".meta.json"
    meta = {"sessions": []}
    if seen and os.path.exists(meta_path):          # a run with no page yet may change its settings
        meta = json.load(open(meta_path, encoding="utf-8"))
    if meta.get("engine", a.engine) != a.engine or meta.get("settings", eng.settings) != eng.settings:
        sys.exit(f"{a.out} was made with other settings ({meta_path}); write to a new --out")
    session = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "machine": machine(a.machine), "pages": 0, "seconds": 0.0, "failed": [], "cut": []}
    meta.update(engine=a.engine, settings=eng.settings)
    meta["sessions"].append(session)

    def save_meta():
        with open(meta_path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        os.replace(meta_path + ".tmp", meta_path)

    save_meta()
    in_a_row = 0
    with open(a.out, "a", encoding="utf-8") as out:
        for n, e in enumerate(todo, 1):
            k = page_key(e)
            t0 = time.perf_counter()
            try:
                text = normal(eng.read(os.path.join(a.images, k + ".png")))
            except Exception as ex:          # out of memory on one page: leave it out, go on
                if "out of memory" not in str(ex).lower():
                    raise
                eng.free()
                session["failed"].append(k)
                in_a_row += 1
                print(f"{n}/{len(todo)} {k}: out of memory, left out", flush=True)
                save_meta()
                if in_a_row == 3:
                    sys.exit(f"three pages in a row out of memory ({ex}); attention is "
                             f"{eng.settings.get('attention')}: lower --max-pixels or try --weights awq")
                continue
            in_a_row = 0
            secs = round(time.perf_counter() - t0, 2)
            out.write(json.dumps({"sha256": e["sha256"], "page": e["page"], "text": text,
                                  "seconds": secs}, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            if getattr(eng, "cut", False):
                session["cut"].append(k)
            session["pages"] += 1
            session["seconds"] = round(session["seconds"] + secs, 2)
            save_meta()
            print(f"{n}/{len(todo)} {k}: {secs:.1f} s{' (cut at max tokens)' if getattr(eng, 'cut', False) else ''}",
                  flush=True)
    print(f"{a.engine}: {session['pages']} pages in {session['seconds']:.0f} s, "
          f"{len(session['failed'])} failed, {len(session['cut'])} cut; {a.out}")


if __name__ == "__main__":
    main()
