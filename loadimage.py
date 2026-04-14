"""
Newspaper OCR using Azure Computer Vision — Column-Aware Version
----------------------------------------------------------------
Fixes the "row mixing" problem by detecting columns and sorting
text blocks left-to-right, top-to-bottom within each column.

Requirements:
    pip install azure-ai-vision-imageanalysis

Setup:
    Replace AZURE_KEY and AZURE_ENDPOINT below with your values.
"""

from pathlib import Path
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# ─────────────────────────────────────────────────────
# CONFIGURATION — paste your values here
# ─────────────────────────────────────────────────────

IMAGE_PATH  = r"C:\Users\Swamini\Downloads\samplenewsarticle.png"
OUTPUT_FILE = r"C:\Users\Swamini\Downloads\azure_extracted_columns.txt"

AZURE_KEY      = "5vrEU7tRuSH0xBxa7ZfW3jEm1DVRtgh4uUvuJ74g8HzD8ILHyyPzJQQJ99CDACYeBjFXJ3w3AAAFACOGKnj3"
AZURE_ENDPOINT = "https://my-ocr-vision56.cognitiveservices.azure.com/"

# How many columns does your newspaper have?
# Set to None to auto-detect, or set manually e.g. 3
NUM_COLUMNS = None

# ─────────────────────────────────────────────────────


def get_line_bbox(line) -> dict:
    """Return bounding box info for a line: x_start, y_start, x_end, y_end."""
    pts = line.bounding_polygon  # list of {x, y} points
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return {
        "x":      min(xs),
        "y":      min(ys),
        "x_end":  max(xs),
        "y_end":  max(ys),
        "text":   line.text
    }


def detect_columns(lines: list, image_width: int, num_columns=None) -> int:
    """
    Auto-detect number of columns by looking at x-start distribution.
    Returns the number of columns found.
    """
    if num_columns:
        return num_columns

    if not lines:
        return 1

    # Collect all x-start positions
    x_starts = sorted([l["x"] for l in lines])

    # Look for large horizontal gaps — those are column boundaries
    gaps = []
    for i in range(1, len(x_starts)):
        gap = x_starts[i] - x_starts[i - 1]
        if gap > image_width * 0.05:  # gap > 5% of image width = column boundary
            gaps.append((gap, x_starts[i]))

    # Unique significant gap x-positions
    unique_gaps = sorted(set(round(g[1] / 50) * 50 for g in gaps))
    n_cols = len(unique_gaps) + 1
    print(f"   Auto-detected {n_cols} column(s) (found {len(unique_gaps)} gap(s))")
    return max(1, min(n_cols, 6))  # cap between 1 and 6


def assign_columns(lines: list, image_width: int, num_columns: int) -> list:
    """
    Assign each line to a column bucket based on its x-start position.
    Returns lines with a 'column' field added.
    """
    col_width = image_width / num_columns

    for line in lines:
        col_idx = int(line["x"] // col_width)
        col_idx = min(col_idx, num_columns - 1)  # clamp
        line["column"] = col_idx

    return lines


def sort_and_group_columns(lines: list, num_columns: int) -> str:
    """
    Sort lines: first by column (left → right), then by y position (top → bottom).
    Returns formatted text with column headers.
    """
    # Sort: column first, then y position within column
    sorted_lines = sorted(lines, key=lambda l: (l["column"], l["y"]))

    output = []
    current_col = -1

    for line in sorted_lines:
        if line["column"] != current_col:
            current_col = line["column"]
            output.append(f"\n{'─' * 50}")
            output.append(f"  COLUMN {current_col + 1}")
            output.append(f"{'─' * 50}")

        output.append(line["text"])

    return "\n".join(output)


def extract_text_column_aware(image_path: str, key: str, endpoint: str) -> str:
    """Main function: call Azure, detect columns, sort and return clean text."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n📂 Loading: {path.name}  ({path.stat().st_size / 1024:.1f} KB)")

    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    with open(image_path, "rb") as f:
        image_data = f.read()

    print("🔍 Sending to Azure Computer Vision...")

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
        language="en"
    )

    if not result.read or not result.read.blocks:
        return "❌ No text detected in the image."

    # Collect all lines with bounding box info
    all_lines = []
    for block in result.read.blocks:
        for line in block.lines:
            all_lines.append(get_line_bbox(line))

    print(f"   Found {len(all_lines)} lines of text")

    # Estimate image width from rightmost x coordinate
    image_width = max(l["x_end"] for l in all_lines)
    print(f"   Estimated image width: {image_width}px")

    # Detect or use manual column count
    n_cols = detect_columns(all_lines, image_width, NUM_COLUMNS)

    # Assign columns
    all_lines = assign_columns(all_lines, image_width, n_cols)

    # Sort and format
    output_text = sort_and_group_columns(all_lines, n_cols)

    return output_text


def main():
    print("=" * 55)
    print("  Azure OCR — Column-Aware Newspaper Reader")
    print("=" * 55)

    if AZURE_KEY == "your-azure-key-here":
        print("\n❌ ERROR: Please paste your Azure Key and Endpoint in the script.")
        print("   Azure Portal → Your Resource → Keys and Endpoint")
        return

    try:
        text = extract_text_column_aware(IMAGE_PATH, AZURE_KEY, AZURE_ENDPOINT)

        print("\n" + "=" * 55)
        print("  EXTRACTED TEXT (Column-Sorted)")
        print("=" * 55)
        print(text)
        print("=" * 55)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"\n💾 Saved to: {OUTPUT_FILE}")
        print("✅ Done!")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check your Azure Key and Endpoint.")


if __name__ == "__main__":
    main()
