from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from paddleocr import PaddleOCR, LayoutDetection

from transformers import LayoutLMv3ForTokenClassification

from layoutreader.v3.helpers import (
    prepare_inputs,
    boxes2inputs,
    parse_logits,
)

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path("report_original.png")

OLLAMA_MODEL = "qwen3:4b"

# LayoutReader model
LAYOUT_READER_MODEL = "hantian/layoutreader"


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Document image not found: {IMAGE_PATH}\n"
        f"Put report_original.png in the same directory as this script."
    )


# ============================================================
# 1. PADDLEOCR
# ============================================================

print("\n" + "=" * 80)
print("STEP 1 - PADDLEOCR")
print("=" * 80)

print("Loading PaddleOCR...")

ocr = PaddleOCR(lang="en")

print("Running OCR...")

ocr_result = ocr.predict(str(IMAGE_PATH))

if not ocr_result:
    raise RuntimeError("PaddleOCR returned no result.")

page = ocr_result[0]

print("\nOCR result keys:")
print(page.keys())


# ============================================================
# GET OCR TEXT
# ============================================================

texts = page.get("rec_texts", [])

scores = page.get(
    "rec_scores",
    [0.0] * len(texts)
)


# PaddleOCR versions can expose polygons differently
boxes = page.get("rec_polys")

if boxes is None:
    boxes = page.get("dt_polys")

if boxes is None:
    raise RuntimeError(
        "Could not find OCR bounding boxes. "
        "Available keys:\n"
        f"{page.keys()}"
    )


# ============================================================
# CONVERT POLYGON -> XYXY
# ============================================================

ocr_regions: List[Dict[str, Any]] = []

for text, box, score in zip(texts, boxes, scores):

    box = np.asarray(box)

    x_coords = box[:, 0]
    y_coords = box[:, 1]

    x1 = float(np.min(x_coords))
    y1 = float(np.min(y_coords))
    x2 = float(np.max(x_coords))
    y2 = float(np.max(y_coords))

    ocr_regions.append(
        {
            "text": str(text),
            "bbox": [x1, y1, x2, y2],
            "confidence": float(score),
        }
    )


print(f"\nOCR regions detected: {len(ocr_regions)}")


# ============================================================
# PRINT OCR
# ============================================================

print("\n" + "-" * 80)
print("RAW OCR")
print("-" * 80)

for i, item in enumerate(ocr_regions):

    print(
        f"{i:03d} | "
        f"{item['text']} | "
        f"bbox={item['bbox']} | "
        f"score={item['confidence']:.3f}"
    )


# ============================================================
# VISUALIZE OCR
# ============================================================

def visualize_ocr(
    image_path: Path,
    regions: List[Dict[str, Any]]
):

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    fig, ax = plt.subplots(
        figsize=(16, 12)
    )

    ax.imshow(image)

    for i, region in enumerate(regions):

        x1, y1, x2, y2 = region["bbox"]

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            fill=False,
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            str(i),
            fontsize=8,
        )

    ax.set_title("PaddleOCR Bounding Boxes")

    plt.tight_layout()

    plt.savefig(
        "ocr_boxes.png",
        dpi=150
    )

    plt.show()


visualize_ocr(
    IMAGE_PATH,
    ocr_regions
)


# ============================================================
# 2. LAYOUT READER
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 - LAYOUT READER")
print("=" * 80)

print(
    f"Loading LayoutReader model: "
    f"{LAYOUT_READER_MODEL}"
)

layout_model = LayoutLMv3ForTokenClassification.from_pretrained(
    LAYOUT_READER_MODEL
)

layout_model.eval()


# ============================================================
# GET IMAGE SIZE
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError(
        f"Could not read image: {IMAGE_PATH}"
    )

image_height, image_width = image.shape[:2]

print(
    f"Image size: "
    f"{image_width} x {image_height}"
)


# ============================================================
# NORMALIZE BOXES
# ============================================================

def normalize_bbox(
    bbox: List[float],
    width: int,
    height: int,
) -> List[int]:

    x1, y1, x2, y2 = bbox

    return [
        int(1000 * x1 / width),
        int(1000 * y1 / height),
        int(1000 * x2 / width),
        int(1000 * y2 / height),
    ]


normalized_boxes = [
    normalize_bbox(
        region["bbox"],
        image_width,
        image_height,
    )
    for region in ocr_regions
]


# ============================================================
# RUN LAYOUT READER
# ============================================================

def get_reading_order(
    boxes: List[List[int]]
) -> List[int]:

    if not boxes:
        return []

    # LayoutReader expects boxes
    # in normalized 0-1000 coordinates.

    inputs = boxes2inputs(
        boxes=boxes
    )

    inputs = prepare_inputs(
        inputs
    )

    import torch

    with torch.no_grad():

        outputs = layout_model(
            **inputs
        )

    logits = outputs.logits

    order = parse_logits(
        logits
    )

    return order


print("Predicting reading order...")

reading_order = get_reading_order(
    normalized_boxes
)

print("\nReading order:")
print(reading_order)


# ============================================================
# REORDER OCR TEXT
# ============================================================

def get_ordered_text(
    regions: List[Dict[str, Any]],
    order: List[int],
) -> str:

    lines = []

    for position, index in enumerate(order):

        if index < 0 or index >= len(regions):
            continue

        text = regions[index]["text"]

        lines.append(
            f"{position + 1}. {text}"
        )

    return "\n".join(lines)


ordered_text = get_ordered_text(
    ocr_regions,
    reading_order,
)


print("\n" + "-" * 80)
print("ORDERED DOCUMENT TEXT")
print("-" * 80)

print(ordered_text)


# ============================================================
# 3. PADDLEOCR LAYOUT DETECTION
# ============================================================

print("\n" + "=" * 80)
print("STEP 3 - LAYOUT DETECTION")
print("=" * 80)

print("Loading LayoutDetection...")

layout_engine = LayoutDetection()

print("Detecting document layout...")

layout_result = layout_engine.predict(
    str(IMAGE_PATH)
)

if not layout_result:
    raise RuntimeError(
        "LayoutDetection returned no result."
    )

layout_page = layout_result[0]

print("\nLayout result keys:")
print(layout_page.keys())


# ============================================================
# EXTRACT LAYOUT BOXES
# ============================================================

layout_boxes = layout_page.get(
    "boxes",
    []
)

print(
    f"\nLayout regions detected: "
    f"{len(layout_boxes)}"
)


layout_regions: List[Dict[str, Any]] = []


for index, box in enumerate(layout_boxes):

    label = box.get(
        "label",
        "unknown"
    )

    score = box.get(
        "score",
        0.0
    )

    coordinate = box.get(
        "coordinate",
        []
    )

    if len(coordinate) != 4:
        continue

    x1, y1, x2, y2 = coordinate

    region = {
        "id": index,
        "label": label,
        "score": float(score),
        "bbox": [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        ],
    }

    layout_regions.append(
        region
    )


# ============================================================
# PRINT LAYOUT
# ============================================================

print("\n" + "-" * 80)
print("LAYOUT REGIONS")
print("-" * 80)

for region in layout_regions:

    print(
        f"ID={region['id']} | "
        f"type={region['label']} | "
        f"score={region['score']:.3f} | "
        f"bbox={region['bbox']}"
    )


# ============================================================
# VISUALIZE LAYOUT
# ============================================================

def visualize_layout(
    image_path: Path,
    regions: List[Dict[str, Any]]
):

    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    fig, ax = plt.subplots(
        figsize=(16, 12)
    )

    ax.imshow(image)

    for region in regions:

        x1, y1, x2, y2 = region["bbox"]

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            fill=False,
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            (
                f"{region['id']}: "
                f"{region['label']}"
            ),
            fontsize=9,
        )

    ax.set_title(
        "PaddleOCR Layout Detection"
    )

    plt.tight_layout()

    plt.savefig(
        "layout_detection.png",
        dpi=150
    )

    plt.show()


visualize_layout(
    IMAGE_PATH,
    layout_regions
)


# ============================================================
# 4. IMAGE CROPPING
# ============================================================

def crop_region(
    image_path: Path,
    bbox: List[float],
) -> np.ndarray:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    x1, y1, x2, y2 = [
        int(v)
        for v in bbox
    ]

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(
        image.shape[1],
        x2
    )

    y2 = min(
        image.shape[0],
        y2
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    return crop


# ============================================================
# SAVE REGION CROPS
# ============================================================

REGION_DIR = Path(
    "document_regions"
)

REGION_DIR.mkdir(
    exist_ok=True
)


for region in layout_regions:

    crop = crop_region(
        IMAGE_PATH,
        region["bbox"]
    )

    if crop.size == 0:
        continue

    filename = (
        f"region_{region['id']}_"
        f"{region['label']}.png"
    )

    output_path = (
        REGION_DIR / filename
    )

    cv2.imwrite(
        str(output_path),
        crop
    )


print(
    f"\nRegion images saved to: "
    f"{REGION_DIR}"
)


# ============================================================
# 5. OCR TEXT ASSOCIATED WITH LAYOUT REGIONS
# ============================================================

def bbox_center(
    bbox: List[float]
):

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def point_inside_bbox(
    point,
    bbox,
):

    x, y = point

    x1, y1, x2, y2 = bbox

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def get_region_text(
    layout_region: Dict[str, Any],
) -> str:

    result = []

    for ocr_item in ocr_regions:

        center = bbox_center(
            ocr_item["bbox"]
        )

        if point_inside_bbox(
            center,
            layout_region["bbox"]
        ):

            result.append(
                ocr_item["text"]
            )

    return "\n".join(result)


# ============================================================
# PRINT REGION TEXT
# ============================================================

print("\n" + "=" * 80)
print("TEXT ASSOCIATED WITH LAYOUT REGIONS")
print("=" * 80)

for region in layout_regions:

    text = get_region_text(
        region
    )

    print("\n" + "-" * 60)

    print(
        f"REGION {region['id']} "
        f"({region['label']})"
    )

    print("-" * 60)

    print(text[:2000])


# ============================================================
# 6. LOCAL QWEN3 WITH OLLAMA
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 - OLLAMA QWEN3")
print("=" * 80)

print(
    f"Using local model: "
    f"{OLLAMA_MODEL}"
)


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


# ============================================================
# 7. CHART ANALYSIS TOOL
# ============================================================

@tool
def AnalyzeChart(
    region_id: int,
) -> str:

    """
    Analyze a chart/figure region using OCR text
    and local Qwen3 reasoning.

    region_id:
        ID of the detected document region.
    """

    region = next(
        (
            r
            for r in layout_regions
            if r["id"] == region_id
        ),
        None,
    )

    if region is None:

        return (
            f"Region {region_id} "
            f"was not found."
        )

    region_text = get_region_text(
        region
    )

    prompt = f"""
You are analyzing a chart or figure
inside a document.

Document region type:
{region['label']}

OCR text found inside this region:

{region_text}

Analyze the chart based on the available
OCR information.

Return:

1. Chart purpose
2. Titles
3. Axis information
4. Important values
5. Trends
6. Main conclusion

Do not invent values that are not present.
If information is unavailable, say so.
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# ============================================================
# 8. TABLE ANALYSIS TOOL
# ============================================================

@tool
def AnalyzeTable(
    region_id: int,
) -> str:

    """
    Analyze a table region using OCR text
    and local Qwen3 reasoning.

    region_id:
        ID of the detected document region.
    """

    region = next(
        (
            r
            for r in layout_regions
            if r["id"] == region_id
        ),
        None,
    )

    if region is None:

        return (
            f"Region {region_id} "
            f"was not found."
        )

    region_text = get_region_text(
        region
    )

    prompt = f"""
You are analyzing a table inside a document.

OCR text extracted from the table:

{region_text}

Analyze the table.

Return:

1. Table purpose
2. Column names
3. Important rows
4. Important values
5. Relationships or trends
6. Short summary

Do not invent missing cells or values.
If the OCR is unclear, explicitly say so.
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# ============================================================
# 9. DOCUMENT CONTEXT FOR THE AGENT
# ============================================================

def build_document_context() -> str:

    parts = []

    parts.append(
        "DOCUMENT OCR TEXT:\n"
    )

    parts.append(
        ordered_text
    )

    parts.append(
        "\n\nLAYOUT REGIONS:\n"
    )

    for region in layout_regions:

        region_text = get_region_text(
            region
        )

        parts.append(
            f"""
Region ID: {region['id']}
Type: {region['label']}
Confidence: {region['score']:.3f}

Text:
{region_text}

"""
        )

    return "\n".join(parts)


document_context = build_document_context()


# ============================================================
# 10. LANGCHAIN AGENT
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 - LANGCHAIN AGENT")
print("=" * 80)


SYSTEM_PROMPT = """
You are a document understanding assistant.

You have access to OCR text and document layout
information.

Your job is to answer questions about the document.

Available tools:

AnalyzeChart
AnalyzeTable

Use AnalyzeTable when the user asks about a table.

Use AnalyzeChart when the user asks about a chart.

For ordinary text questions, use the OCR text
provided in the conversation.

Do not invent information.

If the document does not contain the requested
information, clearly say that it was not found.
"""


tools = [
    AnalyzeChart,
    AnalyzeTable,
]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 11. AGENT FUNCTION
# ============================================================

def run_agent(
    question: str,
) -> str:

    full_question = f"""
Here is the document context:

{document_context}

User question:

{question}
"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": full_question,
                }
            ]
        }
    )

    messages = response.get(
        "messages",
        []
    )

    if not messages:
        return "No response generated."

    return messages[-1].content


# ============================================================
# 12. INTERACTIVE MODE
# ============================================================

def interactive_mode():

    print("\n" + "=" * 80)

    print(
        "DOCUMENT AGENT READY"
    )

    print(
        "Using Ollama:",
        OLLAMA_MODEL
    )

    print(
        "Type 'exit' to stop."
    )

    print("=" * 80)

    while True:

        question = input(
            "\nAsk a question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            break

        if not question:

            continue

        print(
            "\nThinking..."
        )

        try:

            answer = run_agent(
                question
            )

            print(
                "\n" + "-" * 80
            )

            print(
                "ANSWER"
            )

            print(
                "-" * 80
            )

            print(answer)

        except Exception as e:

            print(
                "\nAgent error:"
            )

            print(e)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    interactive_mode()
