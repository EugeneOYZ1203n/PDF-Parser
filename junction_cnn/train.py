from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from junction_cnn.constants import DIRECTION_THRESHOLD, DIRECTION_TOLERANCE, GRADIENT_CLIP, IMAGE_HEIGHT, IMAGE_WIDTH, JUNCTION_THRESHOLD, JUNCTION_TOLERANCE, NUM_JUNCTIONS, NUM_WORKERS, VALIDATION_SIZE, WEIGHT_DECAY
from junction_cnn.decode import decode_junctions
from junction_cnn.losses import total_loss
from junction_cnn.model import JunctionCNN
from junction_cnn.synthetic import SyntheticDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train junction/direction CNN.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save the best model."
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dataset-size", type=int, default=4)
    return parser.parse_args()


def distance_sq(a: tuple[int, int], b: tuple[int, int]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def match_junctions(
    predictions: list[tuple[int, int, float]],
    targets: list[tuple[int, int]],
) -> tuple[int, int, int]:
    """Greedy one-to-one junction matching."""
    tolerance_sq = JUNCTION_TOLERANCE**2
    candidates = []

    for pi, (px, py, _) in enumerate(predictions):
        for ti, target in enumerate(targets):
            d2 = distance_sq((px, py), target)
            if d2 <= tolerance_sq:
                candidates.append((d2, pi, ti))

    candidates.sort()

    matched_predictions = set()
    matched_targets = set()
    tp = 0

    for _, pi, ti in candidates:
        if pi in matched_predictions or ti in matched_targets:
            continue
        matched_predictions.add(pi)
        matched_targets.add(ti)
        tp += 1

    return tp, len(predictions) - tp, len(targets) - tp


def angular_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def direction_accuracy(
    direction_logits: torch.Tensor,
    direction_target: torch.Tensor,
    junction_target: torch.Tensor,
) -> tuple[int, int]:
    """Measure whether each target direction has a nearby prediction."""
    probabilities = torch.sigmoid(direction_logits)
    bins, height, width = probabilities.shape

    if junction_target.ndim == 3:
        junction_target = junction_target[0]

    junctions = decode_junctions(junction_target)
    correct = total = 0
    angle_step = 360.0 / bins

    for item in junctions:
        x, y = item[0], item[1]

        # Local region around junction
        x0, x1 = max(0, x - 3), min(width, x + 4)
        y0, y1 = max(0, y - 3), min(height, y + 4)

        scores = probabilities[:, y0:y1, x0:x1].amax(dim=(1, 2))
        predicted_bins = torch.where(scores >= DIRECTION_THRESHOLD)[0]
        target_bins = torch.where(direction_target[:, y, x] > 0.5)[0]

        predicted_angles = predicted_bins.float() * angle_step
        target_angles = target_bins.float() * angle_step

        for target_angle in target_angles:
            total += 1
            if len(predicted_angles) == 0:
                continue

            min_error = min(
                angular_difference(float(target_angle), float(pred_angle))
                for pred_angle in predicted_angles
            )
            correct += int(min_error <= DIRECTION_TOLERANCE)

    return correct, total


@torch.no_grad()
def evaluate(
    model: JunctionCNN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    losses = [0.0, 0.0, 0.0]
    tp = fp = fn = 0
    direction_correct = direction_total = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        junction_target = batch["junction"].to(device, non_blocking=True)
        direction_target = batch["direction"].to(device, non_blocking=True)

        outputs = model(image)

        loss, loss_dict = total_loss(outputs, junction_target, direction_target)

        losses[0] += loss_dict["total"]
        losses[1] += loss_dict["junction"]
        losses[2] += loss_dict["direction"]

        junction_predictions = torch.sigmoid(outputs["junction_logits"])

        for i in range(image.shape[0]):
            predictions = decode_junctions(
                junction_predictions[i, 0],
                threshold=JUNCTION_THRESHOLD,
            )
            targets = decode_junctions(junction_target[i, 0])

            batch_tp, batch_fp, batch_fn = match_junctions(predictions, targets)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn

            batch_correct, batch_total = direction_accuracy(
                outputs["direction_logits"][i],
                direction_target[i],
                junction_target[i],
            )
            direction_correct += batch_correct
            direction_total += batch_total

    n = len(loader)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": losses[0] / n,
        "junction_loss": losses[1] / n,
        "direction_loss": losses[2] / n,
        "junction_precision": precision,
        "junction_recall": recall,
        "junction_f1": f1,
        "direction_accuracy": direction_correct / max(direction_total, 1),
    }


def train_epoch(
    model: JunctionCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float, float]:
    """Train for one epoch. Returns average losses."""
    model.train()
    total_l = total_j = total_d = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs:03d}")

    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        junction_target = batch["junction"].to(device, non_blocking=True)
        direction_target = batch["direction"].to(device, non_blocking=True)

        outputs = model(image)

        loss, loss_dict = total_loss(outputs, junction_target, direction_target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_l += loss_dict["total"]
        total_j += loss_dict["junction"]
        total_d += loss_dict["direction"]

        progress.set_postfix(
            loss=f"{loss_dict['total']:.4f}",
            junction=f"{loss_dict['junction']:.4f}",
            direction=f"{loss_dict['direction']:.4f}",
        )

    n = len(loader)
    return total_l / n, total_j / n, total_d / n


def save_checkpoint(
    model: JunctionCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        **metrics,
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = JunctionCNN().to(device)

    train_dataset = SyntheticDataset(
        length=args.dataset_size,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        num_junctions=NUM_JUNCTIONS,
    )
    validation_dataset = SyntheticDataset(
        length=VALIDATION_SIZE,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        num_junctions=NUM_JUNCTIONS,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_junction, train_direction = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs
        )

        metrics = evaluate(model, validation_loader, device)

        print(f"\nEpoch {epoch:03d}")
        print(f"  train loss:      {train_loss:.6f}")
        print(f"  train junction:  {train_junction:.6f}")
        print(f"  train direction: {train_direction:.6f}")
        print(f"  val loss:        {metrics['loss']:.6f}")
        print(f"  val junction:    {metrics['junction_loss']:.6f}")
        print(f"  val direction:   {metrics['direction_loss']:.6f}")
        print(f"  junction P:      {metrics['junction_precision']:.4f}")
        print(f"  junction R:      {metrics['junction_recall']:.4f}")
        print(f"  junction F1:     {metrics['junction_f1']:.4f}")
        print(f"  direction acc:   {metrics['direction_accuracy']:.4f}")

        if metrics["junction_f1"] > best_f1:
            best_f1 = metrics["junction_f1"]
            save_checkpoint(model, optimizer, epoch, metrics, args.output)
            print(f"  ★ saved best model (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()