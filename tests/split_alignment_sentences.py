#!/usr/bin/env python3
"""Split an FA alignment into sentence chunks.

Reads tests/outputs/alignment.json and writes the same schema to
tests/outputs/alignment_sentences.json. Chunk sample spans stay half-open,
but each chunk is one sentence instead of an ASR window.

Items carry no spaces or punctuation. Those characters stay on the original
chunk text and are attached to the preceding item. Overlapping ASR windows are
removed before the sentence split, so a period inserted at a window edge does
not become a sentence boundary.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PUNCT = set(".,?!。！？…~、，,:：;；\"'“”‘’()[]{}%&")
CLOSERS = set("\"'”’)]}")
SENT_END = set(".?!。！？…")


def core(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace() and ch not in PUNCT)


def align_chunk(chunk: dict) -> list[dict]:
    text = chunk["text"]
    items = chunk["items"]
    expected = "".join(core(item["text"]) for item in items)
    if core(text) != expected:
        raise ValueError(f"chunk {chunk['index']}: text and items diverged")

    pos = 0
    core_ends: list[int] = []
    for item in items:
        want = core(item["text"])
        if want == "":
            raise ValueError(f"chunk {chunk['index']}: empty aligned item")
        got = ""
        while pos < len(text) and got != want:
            ch = text[pos]
            pos += 1
            if not ch.isspace() and ch not in PUNCT:
                got += ch
        if got != want:
            raise ValueError(f"chunk {chunk['index']}: could not align {item['text']!r}")
        core_ends.append(pos)

    tokens: list[dict] = []
    prev = 0
    for item, end in zip(items, core_ends):
        trail = end
        while trail < len(text) and (text[trail].isspace() or text[trail] in PUNCT):
            trail += 1
        tokens.append(
            {
                "item": {
                    "text": item["text"],
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                },
                "slice": text[prev:trail],
                "language": chunk["language"],
            }
        )
        prev = trail
    return tokens


def _bounds(parts: list[str]) -> list[int]:
    bounds = [0]
    for part in parts:
        bounds.append(bounds[-1] + len(part))
    return bounds


def drop_overlapped_tail(
    prev: list[dict],
    nxt: list[dict],
    max_chars: int,
    boundary_sec: float,
) -> list[dict]:
    """Drop the previous copy of speech repeated by an overlapping next chunk."""
    if not prev or not nxt or max_chars <= 0:
        return prev

    prev_cores = [core(token["item"]["text"]) for token in prev]
    next_cores = [core(token["item"]["text"]) for token in nxt]
    prev_bounds = _bounds(prev_cores)
    next_bounds = _bounds(next_cores)
    prev_text = "".join(prev_cores)
    next_text = "".join(next_cores)
    limit = min(len(prev_text), len(next_text), max_chars)
    best = 0
    for size in range(1, limit + 1):
        if prev_text[-size:] == next_text[:size]:
            best = size
    if best == 0:
        return _drop_by_time(prev, boundary_sec)

    drop_from = next(
        (index for index in range(len(prev_cores)) if prev_bounds[-1] - prev_bounds[index] == best),
        None,
    )
    next_take = next(
        (index for index in range(len(next_cores) + 1) if next_bounds[index] == best),
        None,
    )
    if drop_from is None or next_take is None:
        return _drop_by_time(prev, boundary_sec)

    dropped = prev[drop_from:]
    if any(token["item"]["start_time"] < boundary_sec - 0.5 for token in dropped):
        return _drop_by_time(prev, boundary_sec)
    return prev[:drop_from]


def _drop_by_time(prev: list[dict], boundary_sec: float) -> list[dict]:
    keep = 0
    for token in prev:
        if token["item"]["start_time"] >= boundary_sec - 1e-3:
            break
        keep += 1
    return prev[:keep]


def sentence_end(token: dict, nxt: dict | None) -> bool:
    text = token["slice"].rstrip()
    if not text:
        return False
    index = len(text) - 1
    while index >= 0 and text[index] in CLOSERS:
        index -= 1
    if index < 0 or text[index] not in SENT_END:
        return False
    # ASR writes "..." where the speaker hesitates and then continues.
    if text[index] == "…" or (index >= 2 and text[index - 2 : index + 1] == "..."):
        return False
    if text[index] != ".":
        return True

    after = token["slice"][index + 1 :] + ("" if nxt is None else nxt["slice"])
    cursor = 0
    saw_space = False
    while cursor < len(after) and (after[cursor].isspace() or after[cursor] in CLOSERS):
        if after[cursor].isspace():
            saw_space = True
        cursor += 1
    if cursor < len(after) and after[cursor].isdigit() and not saw_space:
        return False
    return True


def sentence_chunks(doc: dict) -> dict:
    audio = doc["audio"]
    sample_rate = audio["sample_rate"]
    num_samples = audio["num_samples"]
    source = [
        chunk
        for chunk in doc["chunks"]
        if chunk.get("status") == "aligned" and chunk.get("items")
    ]
    source.sort(key=lambda chunk: (chunk["start_sample"], chunk["index"]))

    stream: list[dict] = []
    for position, chunk in enumerate(source):
        tokens = align_chunk(chunk)
        if position > 0 and chunk["start_sample"] < source[position - 1]["end_sample"]:
            overlap_sec = (source[position - 1]["end_sample"] - chunk["start_sample"]) / sample_rate
            stream = drop_overlapped_tail(
                stream,
                tokens,
                max(16, int(overlap_sec * 25)),
                chunk["start_sample"] / sample_rate,
            )
        stream.extend(tokens)

    sentences: list[dict] = []
    buffer: list[dict] = []

    def flush() -> None:
        if not buffer:
            return
        items = [token["item"] for token in buffer]
        start = max(0, math.floor(items[0]["start_time"] * sample_rate + 1e-6))
        end = min(num_samples, math.ceil(items[-1]["end_time"] * sample_rate - 1e-6))
        sentences.append(
            {
                "index": len(sentences),
                "start_sample": start,
                "end_sample": max(start + 1, end),
                "text": "".join(token["slice"] for token in buffer).strip(),
                "language": buffer[0]["language"],
                "status": "aligned",
                "items": items,
            }
        )
        buffer.clear()

    for index, token in enumerate(stream):
        buffer.append(token)
        nxt = stream[index + 1] if index + 1 < len(stream) else None
        if sentence_end(token, nxt):
            flush()
    flush()

    return {
        "audio": audio,
        "time_reference": doc.get("time_reference", "audio_start"),
        "overlap_deduplicated": True,
        "chunks": sentences,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    source = root / "outputs" / "alignment.json"
    target = root / "outputs" / "alignment_sentences.json"
    doc = json.loads(source.read_text(encoding="utf-8"))
    converted = sentence_chunks(doc)
    target.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chunks = converted["chunks"]
    print(f"[INFO] source_chunks={len(doc['chunks'])} sentence_chunks={len(chunks)}")
    print(f"[INFO] saved={target}")
    if chunks:
        print(f"[INFO] first={chunks[0]['text']}")
        print(f"[INFO] last={chunks[-1]['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
