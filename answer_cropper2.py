from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR


CIRCLED_TO_DIGIT = {
    "①": "1",
    "②": "2",
    "③": "3",
    "④": "4",
    "⑤": "5",
}

NOISY_ANSWER_MAP = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "|": "1",
    "B": "8",
    "S": "5",
    "x": "",
    "X": "",
}

QUESTION_PREFIX_RE = re.compile(r"^(?P<number>\d{4})(?P<tail>.*)$")
ENTRY_RE = re.compile(r"(?P<number>\d{4})\s*(?P<answer>[①②③④⑤12345])$")
_OCR_ENGINE: RapidOCR | None = None


@dataclass
class OcrBox:
    text: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float


@dataclass
class QuestionAnchor:
    question_number: str
    box: OcrBox


@dataclass
class ColumnBand:
    left: int
    right: int


@dataclass
class RowBand:
    top: int
    bottom: int
    anchors: list[QuestionAnchor]
    top_margin: int
    bottom_margin: int


@dataclass
class ExpandedRowBand:
    x1: int
    y1: int
    x2: int
    y2: int
    row: RowBand


@dataclass
class ItemBand:
    x1: int
    y1: int
    x2: int
    y2: int
    anchor: QuestionAnchor


def normalize_text(text: str) -> str:
    text = text.strip().replace(" ", "")
    for old, new in CIRCLED_TO_DIGIT.items():
        text = text.replace(old, new)
    for old, new in NOISY_ANSWER_MAP.items():
        text = text.replace(old, new)
    return text


def extract_entry(text: str) -> tuple[str, str] | None:
    normalized = normalize_text(text)
    match = ENTRY_RE.match(normalized)
    if not match:
        return None
    return match.group("number"), match.group("answer")


def extract_question_prefix(text: str) -> tuple[str, str] | None:
    normalized = normalize_text(text)
    match = QUESTION_PREFIX_RE.match(normalized)
    if not match:
        return None
    return match.group("number"), match.group("tail")


def render_page(page: fitz.Page, scale: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def build_question_color_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 40, 40), (140, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    masked = np.full_like(image, 255)
    masked[mask > 0] = image[mask > 0]
    return masked


def build_question_color_binary_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (5, 40, 40), (140, 255, 255))


def to_ocr_boxes(result: list[list[object]] | None) -> list[OcrBox]:
    boxes: list[OcrBox] = []
    if not result:
        return boxes
    for points, text, score in result:
        pts = np.array(points, dtype=np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        boxes.append(
            OcrBox(
                text=text,
                score=float(score),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                cx=float((x1 + x2) / 2),
                cy=float((y1 + y2) / 2),
            )
        )
    return boxes


def collect_question_anchors(boxes: list[OcrBox], min_score: float = 0.75) -> list[QuestionAnchor]:
    anchors: list[QuestionAnchor] = []
    for box in boxes:
        parsed = extract_question_prefix(box.text)
        if not parsed or box.score < min_score:
            continue
        question_number, _ = parsed
        anchors.append(QuestionAnchor(question_number=question_number, box=box))
    return anchors


def merge_question_anchors(*anchor_groups: list[QuestionAnchor]) -> list[QuestionAnchor]:
    merged: dict[str, QuestionAnchor] = {}
    for anchors in anchor_groups:
        for anchor in anchors:
            prev = merged.get(anchor.question_number)
            if prev is None or anchor.box.score > prev.box.score:
                merged[anchor.question_number] = anchor
    return list(merged.values())


def detect_columns(page_image: np.ndarray, anchors: list[QuestionAnchor]) -> list[ColumnBand]:
    page_width = page_image.shape[1]
    midpoint = page_width / 2.0
    anchor_widths = [anchor.box.x2 - anchor.box.x1 for anchor in anchors]
    mean_anchor_width = (
        float(sum(anchor_widths) / len(anchor_widths))
        if anchor_widths
        else max(16.0, page_width * 0.03)
    )

    left_anchors = [anchor for anchor in anchors if anchor.box.cx < midpoint]
    right_anchors = [anchor for anchor in anchors if anchor.box.cx >= midpoint]

    if (not left_anchors or not right_anchors) and len(anchors) >= 2:
        sorted_anchors = sorted(anchors, key=lambda item: item.box.cx)
        split_index = len(sorted_anchors) // 2
        left_anchors = sorted_anchors[:split_index]
        right_anchors = sorted_anchors[split_index:]

    left_default = ColumnBand(0, max(0, int(midpoint) - 1))
    right_default = ColumnBand(min(page_width, int(midpoint)), page_width)
    left_band = build_column_band_from_anchors(page_width, left_anchors, mean_anchor_width, left_default)
    right_band = build_column_band_from_anchors(page_width, right_anchors, mean_anchor_width, right_default)
    return [left_band, right_band]


def build_column_band_from_anchors(
    page_width: int,
    anchors: list[QuestionAnchor],
    mean_anchor_width: float,
    default_band: ColumnBand,
) -> ColumnBand:
    if not anchors:
        return default_band

    left = int(round(min(anchor.box.x1 for anchor in anchors) - mean_anchor_width * 0.2))
    right = int(round(max(anchor.box.x2 for anchor in anchors) + mean_anchor_width * 1.3))
    left = max(0, left)
    right = min(page_width, right)
    if right <= left:
        return default_band
    return ColumnBand(left, right)


def compute_page_anchor_margins(anchors: list[QuestionAnchor]) -> tuple[int, int]:
    if not anchors:
        return 8, 8
    mean_height = sum(anchor.box.y2 - anchor.box.y1 for anchor in anchors) / len(anchors)
    return max(1, int(round(mean_height * 0.4))), max(1, int(round(mean_height * 0.2)))


def choose_column(anchor: QuestionAnchor, columns: list[ColumnBand]) -> int:
    for idx, column in enumerate(columns):
        if column.left <= anchor.box.cx <= column.right:
            return idx
    distances = [
        min(abs(anchor.box.cx - column.left), abs(anchor.box.cx - column.right))
        for column in columns
    ]
    return int(np.argmin(distances))


def build_row_bands(
    column_anchors: list[QuestionAnchor],
    page_height: int,
    top_margin: int,
    bottom_margin: int,
) -> list[RowBand]:
    if not column_anchors:
        return []

    anchors = sorted(column_anchors, key=lambda item: (item.box.cy, item.box.x1))
    y_groups: list[list[QuestionAnchor]] = []
    for anchor in anchors:
        if not y_groups:
            y_groups.append([anchor])
            continue
        last = y_groups[-1]
        last_center = sum(item.box.cy for item in last) / len(last)
        threshold = max(max(item.box.y2 - item.box.y1 for item in last), anchor.box.y2 - anchor.box.y1, 40.0) * 0.9
        if abs(anchor.box.cy - last_center) <= threshold:
            last.append(anchor)
        else:
            y_groups.append([anchor])

    rows: list[RowBand] = []
    for group in y_groups:
        group.sort(key=lambda item: item.box.x1)
        group_top = min(item.box.y1 for item in group)
        group_bottom = max(item.box.y2 for item in group)
        top = max(0, int(group_top - top_margin))
        bottom = int(group_bottom)
        if bottom <= top:
            bottom = min(page_height, top + 40)
        rows.append(
            RowBand(
                top=top,
                bottom=bottom,
                anchors=group,
                top_margin=top_margin,
                bottom_margin=bottom_margin,
            )
        )

    return rows


def select_full_anchor_candidate(
    full_by_number: dict[str, list[QuestionAnchor]],
    expected_number: str,
    row: RowBand,
    x_min: float,
    x_max: float,
    target_x: float,
) -> QuestionAnchor | None:
    candidates = full_by_number.get(expected_number, [])
    if not candidates:
        return None

    x_padding = max(24.0, abs(x_max - x_min) * 0.15)
    y_padding = max(row.top_margin, row.bottom_margin, 12)
    valid_candidates = [
        candidate
        for candidate in candidates
        if (x_min - x_padding) <= candidate.box.x1 <= (x_max + x_padding)
        and (row.top - y_padding) <= candidate.box.cy <= (row.bottom + y_padding)
    ]
    if not valid_candidates:
        return None

    def candidate_score(candidate: QuestionAnchor) -> tuple[float, float]:
        x_distance = abs(candidate.box.x1 - target_x)
        return (candidate.box.score, -x_distance)

    return max(valid_candidates, key=candidate_score)


def append_recovered_anchor(
    recovered: list[QuestionAnchor],
    existing_numbers: set[str],
    candidate: QuestionAnchor | None,
) -> None:
    if candidate is None:
        return
    if candidate.question_number in existing_numbers:
        return
    recovered.append(candidate)
    existing_numbers.add(candidate.question_number)


def row_needs_full_ocr_recovery(row_anchors: list[QuestionAnchor]) -> bool:
    if len(row_anchors) < 2:
        return False

    nums = [int(anchor.question_number) for anchor in row_anchors]
    for prev_num, next_num in zip(nums, nums[1:]):
        if next_num - prev_num > 1:
            return True

    if len(nums) >= 3:
        if nums[2] == nums[1] + 1 and nums[0] != nums[1] - 1:
            return True
        if nums[-2] == nums[-3] + 1 and nums[-1] != nums[-2] + 1:
            return True

    return False


def collect_question_anchors_in_roi(
    page_image: np.ndarray,
    ocr_engine: RapidOCR,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> list[QuestionAnchor]:
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(page_image.shape[1], x2)
    y2 = min(page_image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return []

    roi = page_image[y1:y2, x1:x2]
    result, _ = ocr_engine(roi)
    boxes = to_ocr_boxes(result)
    anchors = collect_question_anchors(boxes)
    shifted: list[QuestionAnchor] = []
    for anchor in anchors:
        box = anchor.box
        shifted.append(
            QuestionAnchor(
                question_number=anchor.question_number,
                box=OcrBox(
                    text=box.text,
                    score=box.score,
                    x1=box.x1 + x1,
                    y1=box.y1 + y1,
                    x2=box.x2 + x1,
                    y2=box.y2 + y1,
                    cx=box.cx + x1,
                    cy=box.cy + y1,
                ),
            )
        )
    return shifted


def recover_row_anchors_from_full_ocr(
    page_image: np.ndarray,
    ocr_engine: RapidOCR,
    color_anchors: list[QuestionAnchor],
    columns: list[ColumnBand],
    page_height: int,
) -> list[QuestionAnchor]:
    if not color_anchors:
        return color_anchors

    page_top_margin, page_bottom_margin = compute_page_anchor_margins(color_anchors)
    anchors_by_column: dict[int, list[QuestionAnchor]] = {idx: [] for idx in range(len(columns))}
    for anchor in color_anchors:
        anchors_by_column[choose_column(anchor, columns)].append(anchor)

    corrected_by_anchor_id: dict[int, QuestionAnchor] = {}
    recovered: list[QuestionAnchor] = []
    existing_numbers = {anchor.question_number for anchor in color_anchors}

    for column_idx, column_anchors in anchors_by_column.items():
        column = columns[column_idx]
        rows = build_row_bands(column_anchors, page_height, page_top_margin, page_bottom_margin)
        for row in rows:
            row_anchors = sorted(row.anchors, key=lambda item: item.box.x1)
            if not row_needs_full_ocr_recovery(row_anchors):
                continue

            roi_x1 = max(column.left, int(min(anchor.box.x1 for anchor in row_anchors) - 80))
            roi_x2 = min(column.right, int(max(anchor.box.x2 for anchor in row_anchors) + 220))
            roi_y1 = max(0, int(row.top - max(row.top_margin, 12)))
            roi_y2 = min(page_image.shape[0], int(row.bottom + max(row.bottom_margin, 18)))
            roi_full_anchors = collect_question_anchors_in_roi(
                page_image=page_image,
                ocr_engine=ocr_engine,
                x1=roi_x1,
                y1=roi_y1,
                x2=roi_x2,
                y2=roi_y2,
            )
            if not roi_full_anchors:
                continue

            full_by_number: dict[str, list[QuestionAnchor]] = {}
            for anchor in roi_full_anchors:
                full_by_number.setdefault(anchor.question_number, []).append(anchor)

            # Repair a bad first anchor when the next two anchors form a clear sequence.
            if len(row_anchors) >= 3:
                first_anchor = row_anchors[0]
                second_anchor = row_anchors[1]
                third_anchor = row_anchors[2]
                second_num = int(second_anchor.question_number)
                third_num = int(third_anchor.question_number)
                expected_num = second_num - 1
                if (
                    third_num == second_num + 1
                    and int(first_anchor.question_number) != expected_num
                    and expected_num > 0
                ):
                    expected_key = f"{expected_num:04d}"
                    candidate = select_full_anchor_candidate(
                        full_by_number=full_by_number,
                        expected_number=expected_key,
                        row=row,
                        x_min=min(first_anchor.box.x1, second_anchor.box.x1),
                        x_max=max(first_anchor.box.x1, second_anchor.box.x1),
                        target_x=first_anchor.box.x1,
                    )
                    if candidate is not None:
                        current_candidate = select_full_anchor_candidate(
                            full_by_number=full_by_number,
                            expected_number=first_anchor.question_number,
                            row=row,
                            x_min=min(first_anchor.box.x1, second_anchor.box.x1),
                            x_max=max(first_anchor.box.x1, second_anchor.box.x1),
                            target_x=first_anchor.box.x1,
                        )
                        if current_candidate is not None:
                            append_recovered_anchor(recovered, existing_numbers, candidate)
                        else:
                            corrected_by_anchor_id[id(first_anchor)] = candidate
                            existing_numbers.add(expected_key)

            # Repair a single outlier only when full-page OCR supports the expected number nearby.
            for prev_anchor, current_anchor, next_anchor in zip(
                row_anchors,
                row_anchors[1:],
                row_anchors[2:],
            ):
                prev_num = int(prev_anchor.question_number)
                next_num = int(next_anchor.question_number)
                expected_num = prev_num + 1
                if next_num != prev_num + 2 or int(current_anchor.question_number) == expected_num:
                    continue

                expected_key = f"{expected_num:04d}"
                candidate = select_full_anchor_candidate(
                    full_by_number=full_by_number,
                    expected_number=expected_key,
                    row=row,
                    x_min=prev_anchor.box.x1,
                    x_max=next_anchor.box.x1,
                    target_x=current_anchor.box.x1,
                )
                if candidate is None:
                    continue

                corrected_by_anchor_id[id(current_anchor)] = candidate
                existing_numbers.add(expected_key)

            # Repair a bad last anchor when the previous two anchors form a clear sequence.
            if len(row_anchors) >= 3:
                third_last_anchor = row_anchors[-3]
                second_last_anchor = row_anchors[-2]
                last_anchor = row_anchors[-1]
                third_last_num = int(third_last_anchor.question_number)
                second_last_num = int(second_last_anchor.question_number)
                expected_num = second_last_num + 1
                if (
                    second_last_num == third_last_num + 1
                    and int(last_anchor.question_number) != expected_num
                ):
                    expected_key = f"{expected_num:04d}"
                    candidate = select_full_anchor_candidate(
                        full_by_number=full_by_number,
                        expected_number=expected_key,
                        row=row,
                        x_min=min(second_last_anchor.box.x1, last_anchor.box.x1),
                        x_max=max(second_last_anchor.box.x1, last_anchor.box.x1),
                        target_x=last_anchor.box.x1,
                    )
                    if candidate is not None:
                        current_candidate = select_full_anchor_candidate(
                            full_by_number=full_by_number,
                            expected_number=last_anchor.question_number,
                            row=row,
                            x_min=min(second_last_anchor.box.x1, last_anchor.box.x1),
                            x_max=max(second_last_anchor.box.x1, last_anchor.box.x1),
                            target_x=last_anchor.box.x1,
                        )
                        if current_candidate is not None:
                            append_recovered_anchor(recovered, existing_numbers, candidate)
                        else:
                            corrected_by_anchor_id[id(last_anchor)] = candidate
                            existing_numbers.add(expected_key)

            corrected_row_anchors = [
                corrected_by_anchor_id.get(id(anchor), anchor) for anchor in row_anchors
            ]
            corrected_row_anchors.sort(key=lambda item: item.box.x1)

            # Recover only missing numbers that also have full-page OCR evidence in the same row zone.
            for prev_anchor, next_anchor in zip(corrected_row_anchors, corrected_row_anchors[1:]):
                prev_num = int(prev_anchor.question_number)
                next_num = int(next_anchor.question_number)
                if next_num - prev_num <= 1:
                    continue
                for missing_num in range(prev_num + 1, next_num):
                    missing_key = f"{missing_num:04d}"
                    if missing_key in existing_numbers:
                        continue
                    target_x = (prev_anchor.box.x1 + next_anchor.box.x1) / 2.0
                    candidate = select_full_anchor_candidate(
                        full_by_number=full_by_number,
                        expected_number=missing_key,
                        row=row,
                        x_min=prev_anchor.box.x1,
                        x_max=next_anchor.box.x1,
                        target_x=target_x,
                    )
                    append_recovered_anchor(recovered, existing_numbers, candidate)

    corrected_anchors = [
        corrected_by_anchor_id.get(id(anchor), anchor) for anchor in color_anchors
    ]
    return merge_question_anchors(corrected_anchors, recovered)


def replace_anchor_number(anchor: QuestionAnchor, question_number: str) -> QuestionAnchor:
    return QuestionAnchor(question_number=question_number, box=anchor.box)


def correct_cross_row_sequence_anomalies(
    anchors: list[QuestionAnchor],
    columns: list[ColumnBand],
    page_height: int,
) -> list[QuestionAnchor]:
    if not anchors:
        return anchors

    top_margin, bottom_margin = compute_page_anchor_margins(anchors)
    anchors_by_column: dict[int, list[QuestionAnchor]] = {idx: [] for idx in range(len(columns))}
    for anchor in anchors:
        anchors_by_column[choose_column(anchor, columns)].append(anchor)

    replacements: dict[int, QuestionAnchor] = {}
    assigned_numbers = {anchor.question_number for anchor in anchors}

    for column_idx, column_anchors in anchors_by_column.items():
        rows = build_row_bands(column_anchors, page_height, top_margin, bottom_margin)
        sorted_rows = [sorted(row.anchors, key=lambda item: item.box.x1) for row in rows]

        for row_idx in range(len(sorted_rows) - 1):
            current = sorted_rows[row_idx]
            next_row = sorted_rows[row_idx + 1]

            # Fix a bad last item when the next row starts with the continued sequence.
            if len(current) >= 2 and len(next_row) >= 2:
                prev_num = int(current[-2].question_number)
                expected_num = prev_num + 1
                next_num = int(next_row[0].question_number)
                next_next_num = int(next_row[1].question_number)
                current_num = int(current[-1].question_number)
                expected_key = f"{expected_num:04d}"
                if (
                    next_num == expected_num + 1
                    and next_next_num == expected_num + 2
                    and current_num != expected_num
                    and (expected_key not in assigned_numbers or current[-1].question_number == expected_key)
                ):
                    replacements[id(current[-1])] = replace_anchor_number(current[-1], expected_key)
                    assigned_numbers.add(expected_key)

            # Fix a bad first item when the previous row ends with the continued sequence.
            if len(current) >= 2 and row_idx > 0:
                prev_row = sorted_rows[row_idx - 1]
                if len(prev_row) >= 2:
                    expected_num = int(prev_row[-1].question_number) + 1
                    current_second_num = int(current[1].question_number)
                    current_first_num = int(current[0].question_number)
                    expected_key = f"{expected_num:04d}"
                    if (
                        current_second_num == expected_num + 1
                        and current_first_num != expected_num
                        and (expected_key not in assigned_numbers or current[0].question_number == expected_key)
                    ):
                        replacements[id(current[0])] = replace_anchor_number(current[0], expected_key)
                        assigned_numbers.add(expected_key)

    if not replacements:
        return anchors

    return [replacements.get(id(anchor), anchor) for anchor in anchors]


def build_crop_foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color_spread = image.max(axis=2) - image.min(axis=2)
    mask = ((gray < 225) | (color_spread > 24)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    footer_h = min(220, image.shape[0] // 8)
    footer = image[image.shape[0] - footer_h :, :]
    footer_gray = gray[image.shape[0] - footer_h :, :]
    footer_spread = color_spread[image.shape[0] - footer_h :, :]
    pale_footer = (
        (footer_gray > 218)
        & (footer_gray < 250)
        & (footer_spread < 16)
        & (footer[:, :, 2] >= footer[:, :, 1])
    )
    mask[image.shape[0] - footer_h :, :][pale_footer] = 0
    return mask


def extend_row_bottom_until_blank(
    foreground_mask: np.ndarray,
    probe_x1: int,
    probe_x2: int,
    start_y: int,
    limit_y: int,
    other_color_mask: np.ndarray | None = None,
    dark_mask: np.ndarray | None = None,
) -> int:
    probe_x1 = max(0, probe_x1)
    probe_x2 = min(foreground_mask.shape[1], probe_x2)
    probe = foreground_mask[:, probe_x1:probe_x2]
    other_probe = None if other_color_mask is None else other_color_mask[:, probe_x1:probe_x2]
    dark_probe = None if dark_mask is None else dark_mask[:, probe_x1:probe_x2]
    current_bottom = start_y
    blank_run = 0
    threshold = max(8, int((probe_x2 - probe_x1) * 0.004))
    other_color_threshold = max(20, int((probe_x2 - probe_x1) * 0.015))
    max_gap = max(0, min(limit_y, foreground_mask.shape[0]) - start_y)
    required_blank_run = min(55, max(8, int(max_gap * 0.4)))

    for y in range(start_y, min(limit_y, foreground_mask.shape[0])):
        if other_probe is not None:
            other_count = int(np.count_nonzero(other_probe[y]))
            if other_count >= other_color_threshold:
                dark_count = 0 if dark_probe is None else int(np.count_nonzero(dark_probe[y]))
                if dark_count < threshold:
                    break
        row_pixels = int(np.count_nonzero(probe[y]))
        if row_pixels >= threshold:
            current_bottom = y + 1
            blank_run = 0
        else:
            blank_run += 1
            if blank_run >= required_blank_run:
                break

    return max(start_y, current_bottom)


def expand_row_band(
    row: RowBand,
    column: ColumnBand,
    next_row_top: int,
    next_anchor_y1: int | None,
    mean_anchor_height: float,
    foreground_mask: np.ndarray,
    probe_x1: int,
    probe_x2: int,
    other_color_mask: np.ndarray | None = None,
    dark_mask: np.ndarray | None = None,
) -> ExpandedRowBand:
    extended_bottom = row.bottom
    limit_y = next_row_top - 1
    if next_anchor_y1 is not None:
        anchor_cap = int(round(next_anchor_y1 - mean_anchor_height * 0.2))
        limit_y = max(limit_y, anchor_cap)
        limit_y = min(foreground_mask.shape[0] - 1, limit_y)
    if limit_y - row.bottom >= 10:
        extended_bottom = extend_row_bottom_until_blank(
            foreground_mask=foreground_mask,
            probe_x1=probe_x1,
            probe_x2=probe_x2,
            start_y=row.bottom,
            limit_y=limit_y,
            other_color_mask=other_color_mask,
            dark_mask=dark_mask,
        )

    final_bottom = min(foreground_mask.shape[0], extended_bottom + row.bottom_margin)
    if final_bottom <= row.top:
        final_bottom = min(foreground_mask.shape[0], row.top + 1)

    return ExpandedRowBand(
        x1=column.left,
        y1=row.top,
        x2=column.right,
        y2=final_bottom,
        row=row,
    )


def build_item_bands_for_row(
    expanded_row: ExpandedRowBand,
    row_anchors: list[QuestionAnchor],
) -> list[ItemBand]:
    if not row_anchors:
        return []

    item_bands: list[ItemBand] = []
    sorted_anchors = sorted(row_anchors, key=lambda item: item.box.x1)
    mean_anchor_width = sum(anchor.box.x2 - anchor.box.x1 for anchor in sorted_anchors) / len(sorted_anchors)
    min_item_width = max(8, int(round(mean_anchor_width * 0.35)))
    for idx, anchor in enumerate(sorted_anchors):
        x1 = max(expanded_row.x1, int(anchor.box.x1))
        if idx + 1 < len(sorted_anchors):
            x2 = min(expanded_row.x2, int(sorted_anchors[idx + 1].box.x1))
        else:
            x2 = expanded_row.x2
        if x2 <= x1:
            x2 = min(expanded_row.x2, x1 + 1)
        if x2 - x1 < min_item_width:
            continue
        item_bands.append(ItemBand(x1=x1, y1=expanded_row.y1, x2=x2, y2=expanded_row.y2, anchor=anchor))
    return item_bands


def refine_item_crop(image: np.ndarray, page_anchor_mean_height: float) -> np.ndarray:
    if image.size == 0:
        return image

    margin = max(1, int(round(page_anchor_mean_height * 0.2)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 170).astype(np.uint8)
    col_dark = np.count_nonzero(dark_mask > 0, axis=0)
    xs = np.where(col_dark > 0)[0]
    if len(xs) > 0:
        right = min(image.shape[1], int(xs.max()) + margin + 1)
        image = image[:, :right]

    if image.size == 0:
        return image

    foreground_mask = build_crop_foreground_mask(image)
    row_foreground = np.count_nonzero(foreground_mask > 0, axis=1)
    ys = np.where(row_foreground > 0)[0]
    if len(ys) > 0:
        top = max(0, int(ys.min()) - margin)
        bottom = min(image.shape[0], int(ys.max()) + margin + 1)
        image = image[top:bottom, :]

    return image


def upscale_image_2x(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image
    return cv2.resize(
        image,
        (image.shape[1] * 2, image.shape[0] * 2),
        interpolation=cv2.INTER_CUBIC,
    )


def write_crop(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Failed to encode image for {output_path}")
    output_path.write_bytes(encoded.tobytes())


def get_ocr_engine() -> RapidOCR:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def build_crop_layout(
    page_image: np.ndarray,
    ocr_engine: RapidOCR,
) -> tuple[list[QuestionAnchor], list[ColumnBand], dict[int, list[QuestionAnchor]], np.ndarray, np.ndarray]:
    anchor_image = build_question_color_mask(page_image)
    anchor_result, _ = ocr_engine(anchor_image)
    anchor_boxes = to_ocr_boxes(anchor_result)
    color_anchors = collect_question_anchors(anchor_boxes)
    columns = detect_columns(page_image, color_anchors)
    anchors = recover_row_anchors_from_full_ocr(
        page_image=page_image,
        ocr_engine=ocr_engine,
        color_anchors=color_anchors,
        columns=columns,
        page_height=page_image.shape[0],
    )
    anchors = correct_cross_row_sequence_anomalies(
        anchors=anchors,
        columns=columns,
        page_height=page_image.shape[0],
    )

    anchors_by_column: dict[int, list[QuestionAnchor]] = {idx: [] for idx in range(len(columns))}
    for anchor in anchors:
        anchors_by_column[choose_column(anchor, columns)].append(anchor)

    foreground_mask = build_crop_foreground_mask(page_image)
    return anchors, columns, anchors_by_column, foreground_mask, []


def parse_page_spec(page_spec: str | None) -> set[int] | None:
    if page_spec is None:
        return None

    selected_pages: set[int] = set()
    for part in page_spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start <= 0 or end <= 0:
                raise ValueError("Page numbers must be positive integers.")
            if end < start:
                raise ValueError(f"Invalid page range: {token}")
            selected_pages.update(range(start, end + 1))
            continue
        page_number = int(token)
        if page_number <= 0:
            raise ValueError("Page numbers must be positive integers.")
        selected_pages.add(page_number)

    if not selected_pages:
        raise ValueError("No valid pages were provided.")
    return selected_pages


def collect_entries(pdf_path: Path, output_dir: Path, scale: float, selected_pages: set[int] | None = None) -> int:
    ocr_engine = get_ocr_engine()
    document = fitz.open(pdf_path)
    saved_count = 0
    seen_numbers: set[str] = set()

    for page in document:
        page_number = page.number + 1
        if selected_pages is not None and page_number not in selected_pages:
            continue

        page_image = render_page(page, scale=scale)
        anchors, columns, anchors_by_column, foreground_mask, _ = build_crop_layout(page_image, ocr_engine)
        if not anchors:
            continue

        page_top_margin, page_bottom_margin = compute_page_anchor_margins(anchors)
        page_anchor_mean_height = sum(anchor.box.y2 - anchor.box.y1 for anchor in anchors) / len(anchors)
        question_color_mask = build_question_color_binary_mask(page_image)
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray < 170).astype(np.uint8) * 255
        nonwhite_mask = ((gray < 248) | ((page_image.max(axis=2) - page_image.min(axis=2)) > 12)).astype(np.uint8) * 255
        other_color_mask = cv2.bitwise_and(
            nonwhite_mask,
            cv2.bitwise_not(cv2.bitwise_or(question_color_mask, dark_mask)),
        )

        for column_idx, column in enumerate(columns):
            column_anchors = anchors_by_column[column_idx]
            if not column_anchors:
                continue

            rows = build_row_bands(column_anchors, page_image.shape[0], page_top_margin, page_bottom_margin)
            probe_x1 = int(min(anchor.box.x1 for anchor in column_anchors)) if column_anchors else column.left
            probe_x2 = int(max(anchor.box.x2 for anchor in column_anchors)) if column_anchors else column.right
            for row_idx, row in enumerate(rows):
                row_anchors = sorted(row.anchors, key=lambda item: item.box.x1)
                next_row_top = rows[row_idx + 1].top if row_idx + 1 < len(rows) else page_image.shape[0]
                next_anchor_y1 = (
                    int(min(anchor.box.y1 for anchor in rows[row_idx + 1].anchors))
                    if row_idx + 1 < len(rows)
                    else None
                )
                last_row_other_color_mask = other_color_mask if row_idx == len(rows) - 1 else None
                last_row_dark_mask = dark_mask if row_idx == len(rows) - 1 else None
                expanded_row = expand_row_band(
                    row=row,
                    column=column,
                    next_row_top=next_row_top,
                    next_anchor_y1=next_anchor_y1,
                    mean_anchor_height=page_anchor_mean_height,
                    foreground_mask=foreground_mask,
                    probe_x1=probe_x1,
                    probe_x2=probe_x2,
                    other_color_mask=last_row_other_color_mask,
                    dark_mask=last_row_dark_mask,
                )
                item_bands = build_item_bands_for_row(expanded_row=expanded_row, row_anchors=row_anchors)
                for item_band in item_bands:
                    anchor = item_band.anchor
                    if anchor.question_number in seen_numbers:
                        continue

                    x1 = item_band.x1
                    x2 = item_band.x2
                    y1 = item_band.y1
                    y2 = item_band.y2
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = page_image[y1:y2, x1:x2]
                    crop = refine_item_crop(crop, page_anchor_mean_height)
                    crop = upscale_image_2x(crop)

                    answer_entry = extract_entry(anchor.box.text)
                    suffix = f"_{answer_entry[1]}" if answer_entry else ""
                    output_path = output_dir / f"{anchor.question_number}{suffix}.png"
                    write_crop(crop, output_path)

                    seen_numbers.add(anchor.question_number)
                    saved_count += 1

    return saved_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop answer items from workbook PDFs.")
    parser.add_argument(
        "input_pdf",
        nargs="?",
        default=None,
        help="Input PDF path. If omitted, the first PDF in the current folder is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="cropped_answers",
        help="Directory where cropped PNG files are saved.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Render scale for OCR and cropping.",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Page selection like '2' or '2,4-5'. Uses 1-based page numbers.",
    )
    return parser


def resolve_input_pdf(user_value: str | None) -> Path:
    if user_value:
        pdf_path = Path(user_value)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {pdf_path}")
        return pdf_path
    pdfs = sorted(Path(".").glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDF file found in the current directory.")
    return pdfs[0]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pdf_path = resolve_input_pdf(args.input_pdf)
    output_dir = Path(args.output_dir)
    selected_pages = parse_page_spec(args.pages)
    saved_count = collect_entries(
        pdf_path=pdf_path,
        output_dir=output_dir,
        scale=args.scale,
        selected_pages=selected_pages,
    )

    print(f"Input PDF: {pdf_path}")
    if selected_pages is not None:
        print(f"Pages: {sorted(selected_pages)}")
    print(f"Saved crops: {saved_count}")
    print(f"Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
