#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    "#2f6f73",
    "#c44e36",
    "#5b7cba",
    "#7d5ba6",
    "#c78d2e",
    "#4f8f49",
    "#777777",
]


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def method_order(label):
    if label == "Sign-Iteration":
        return 10
    if label == "Box-PGA":
        return 20
    if label.startswith("LR-SDP r="):
        return 100 + int(label.rsplit("=", 1)[1])
    if label == "SDP+GW":
        return 1000
    return 10000


def nice_num(x):
    if abs(x) >= 1000 or (0 < abs(x) < 0.01):
        return f"{x:.2e}"
    return f"{x:.3g}"


def nice_step(raw_step):
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)
    for candidate in (1, 2, 2.5, 5, 10):
        if fraction <= candidate:
            return candidate * (10 ** exponent)
    return 10 ** (exponent + 1)


def regular_ticks(raw_min, raw_max, target_intervals=5, include_zero=False):
    if raw_min == raw_max:
        raw_min -= 1.0
        raw_max += 1.0
    if include_zero:
        raw_min = min(0.0, raw_min)
    step = nice_step((raw_max - raw_min) / target_intervals)
    start = math.floor(raw_min / step) * step
    stop = math.ceil(raw_max / step) * step
    ticks = []
    value = start
    while value <= stop + 0.5 * step:
        if not include_zero or value >= -1e-9:
            ticks.append(0.0 if abs(value) < 1e-9 else value)
        value += step
    return ticks


def read_groups(path):
    groups = {}
    valid_status = {"ok", "converged", "solved"}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] not in valid_status:
                continue
            try:
                obj = float(row["binary_objective"])
                sec = float(row["solve_seconds"])
            except ValueError:
                continue
            groups.setdefault(row["method_label"], []).append((obj, sec))
    return {label: groups[label] for label in sorted(groups, key=method_order)}


def draw_axes(draw, box, y_ticks, y_to_px, font):
    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    for tick in y_ticks:
        y = y_to_px(tick)
        draw.line((left - 6, y, left, y), fill="#333333", width=1)
        draw.line((left, y, right, y), fill="#e8e8e8", width=1)
        label = f"{tick:.0f}" if abs(tick) >= 10 else nice_num(tick)
        w = draw.textlength(label, font=font)
        draw.text((left - 12 - w, y - 8), label, fill="#333333", font=font)


def draw_x_labels(draw, labels, xs, y, font):
    for label, x in zip(labels, xs):
        parts = label.split(" ")
        if len(parts) >= 2 and label.startswith("LR-SDP"):
            text = f"LR-SDP\n{parts[-1]}"
        else:
            text = label.replace("-", "-\n") if label in {"Sign-Iteration"} else label
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            w = draw.textlength(line, font=font)
            draw.text((x - w / 2, y + idx * 17), line, fill="#333333", font=font)


def save_objective_plot(groups, path):
    labels = list(groups)
    means = [statistics.mean(v[0] for v in groups[label]) for label in labels]
    stds = [
        statistics.stdev(v[0] for v in groups[label]) if len(groups[label]) > 1 else 0.0
        for label in labels
    ]
    counts = [len(groups[label]) for label in labels]

    maxes = [max(v[0] for v in groups[label]) for label in labels]
    raw_min = min(m - s for m, s in zip(means, stds))
    raw_max = max(max(maxes), max(m + s for m, s in zip(means, stds)))
    ticks = regular_ticks(raw_min, raw_max, target_intervals=5)
    y_min = ticks[0]
    y_max = ticks[-1]

    width, height = 1400, 840
    left, top, right, bottom = 130, 90, 1340, 680
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(18)
    tick_font = load_font(16)
    label_font = load_font(17)

    draw.text((left, 28), "T=2000 objective by method", fill="#111111", font=title_font)
    draw.text(
        (left, 64),
        "Bars show mean x'Wx; error bars are +/- 1 standard deviation; black diamonds show the max value achieved.",
        fill="#555555",
        font=subtitle_font,
    )

    def y_to_px(y):
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    draw_axes(draw, (left, top, right, bottom), ticks, y_to_px, tick_font)

    step = (right - left) / len(labels)
    xs = [left + (i + 0.5) * step for i in range(len(labels))]
    bar_w = min(88, step * 0.52)
    for i, (label, mean, std, max_value) in enumerate(zip(labels, means, stds, maxes)):
        x = xs[i]
        y = y_to_px(mean)
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=color)
        if std > 0:
            y_hi = y_to_px(mean + std)
            y_lo = y_to_px(mean - std)
            draw.line((x, y_hi, x, y_lo), fill="#222222", width=3)
            draw.line((x - 14, y_hi, x + 14, y_hi), fill="#222222", width=3)
            draw.line((x - 14, y_lo, x + 14, y_lo), fill="#222222", width=3)
        max_y = y_to_px(max_value)
        marker = 8
        draw.polygon(
            (
                (x, max_y - marker),
                (x + marker, max_y),
                (x, max_y + marker),
                (x - marker, max_y),
            ),
            fill="#111111",
        )
        count_label = f"n={counts[i]}"
        w = draw.textlength(count_label, font=tick_font)
        draw.text((x - w / 2, bottom + 52), count_label, fill="#666666", font=tick_font)

    draw_x_labels(draw, labels, xs, bottom + 20, label_font)
    draw.text((20, top + 230), "binary objective x'Wx", fill="#333333", font=label_font)
    img.save(path)


def save_runtime_plot(groups, path):
    labels = list(groups)
    totals = [sum(v[1] for v in groups[label]) for label in labels]
    ticks = regular_ticks(0.0, max(totals) * 1.05, target_intervals=5, include_zero=True)
    y_min = ticks[0]
    y_max = ticks[-1]

    width, height = 1400, 840
    left, top, right, bottom = 130, 90, 1340, 680
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(18)
    tick_font = load_font(16)
    label_font = load_font(17)

    draw.text((left, 28), "T=2000 runtime by method", fill="#111111", font=title_font)
    draw.text(
        (left, 64),
        "Bars show total wall-clock solve time for the current restart budget; y-axis is linear seconds.",
        fill="#555555",
        font=subtitle_font,
    )

    def y_to_px(y):
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    draw_axes(draw, (left, top, right, bottom), ticks, y_to_px, tick_font)

    step = (right - left) / len(labels)
    xs = [left + (i + 0.5) * step for i in range(len(labels))]
    bar_w = min(88, step * 0.52)
    for i, (label, total) in enumerate(zip(labels, totals)):
        x = xs[i]
        y = y_to_px(total)
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=color)
        time_label = f"{nice_num(total)}s"
        w = draw.textlength(time_label, font=tick_font)
        draw.text((x - w / 2, bottom + 52), time_label, fill="#666666", font=tick_font)

    draw_x_labels(draw, labels, xs, bottom + 20, label_font)
    draw.text((20, top + 245), "seconds", fill="#333333", font=label_font)
    img.save(path)


def save_avg_runtime_plot(groups, path):
    labels = list(groups)
    averages = [statistics.mean(v[1] for v in groups[label]) for label in labels]
    ticks = regular_ticks(0.0, max(averages) * 1.08, target_intervals=5,
                          include_zero=True)
    y_min = ticks[0]
    y_max = ticks[-1]

    width, height = 1400, 840
    left, top, right, bottom = 130, 90, 1340, 680
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(18)
    tick_font = load_font(16)
    label_font = load_font(17)

    draw.text((left, 28), "T=2000 average runtime per start", fill="#111111",
              font=title_font)
    draw.text(
        (left, 64),
        "Bars show average wall-clock solve time per random start/seed. SDP+GW has one run.",
        fill="#555555",
        font=subtitle_font,
    )

    def y_to_px(y):
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    draw_axes(draw, (left, top, right, bottom), ticks, y_to_px, tick_font)

    step = (right - left) / len(labels)
    xs = [left + (i + 0.5) * step for i in range(len(labels))]
    bar_w = min(88, step * 0.52)
    for i, (label, average) in enumerate(zip(labels, averages)):
        x = xs[i]
        y = y_to_px(average)
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=color)
        time_label = f"{nice_num(average)}s"
        w = draw.textlength(time_label, font=tick_font)
        draw.text((x - w / 2, bottom + 52), time_label, fill="#666666",
                  font=tick_font)

    draw_x_labels(draw, labels, xs, bottom + 20, label_font)
    draw.text((20, top + 220), "seconds per start", fill="#333333",
              font=label_font)
    img.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/t2000_seeded_qubo.csv")
    parser.add_argument("--output-dir", default="results/plots/t2000_seeded")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = read_groups(args.input)
    save_objective_plot(groups, out_dir / "t2000_objective_errorbars.png")
    save_runtime_plot(groups, out_dir / "t2000_runtime.png")
    save_avg_runtime_plot(groups, out_dir / "t2000_runtime_per_start.png")


if __name__ == "__main__":
    main()
