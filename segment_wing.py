"""
Wing segmentation pipeline.

Three-stage pipeline:
  1. crop_wing        – Detect and crop the wing from a raw image using SAM3.
  2. segment_wing     – Generate fine-grained segmentation masks on a cropped wing using SAM.
  3. save_masks_tiff  – Write masks to an OME-TIFF file (one channel per mask).

Every function accepts a `device` argument ("cpu", "cuda", or "cuda:0", etc.)
so callers can pin execution to a specific backend.
"""

import gc
import os
import random

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Preset 30 distinct colors (RGB, 0-1 range) for optional visualisation
# ---------------------------------------------------------------------------
MASK_COLORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 0.0],
    [0.5, 0.0, 1.0],
    [0.0, 1.0, 0.5],
    [1.0, 0.0, 0.5],
    [0.5, 1.0, 0.0],
    [0.0, 0.5, 1.0],
    [0.8, 0.4, 0.0],
    [0.4, 0.0, 0.8],
    [0.0, 0.8, 0.4],
    [0.8, 0.0, 0.4],
    [0.4, 0.8, 0.0],
    [0.0, 0.4, 0.8],
    [1.0, 0.7, 0.7],
    [0.7, 1.0, 0.7],
    [0.7, 0.7, 1.0],
    [1.0, 1.0, 0.7],
    [1.0, 0.7, 1.0],
    [0.7, 1.0, 1.0],
    [0.6, 0.3, 0.0],
    [0.3, 0.0, 0.6],
    [0.0, 0.6, 0.3],
    [0.6, 0.0, 0.3],
    [0.3, 0.6, 0.0],
    [0.0, 0.3, 0.6],
]


# ── helpers ────────────────────────────────────────────────────────────────
def _resolve_device(device: str | None = None) -> str:
    """Return the requested device, falling back to CUDA when available."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_deterministic(seed: int = 0) -> None:
    """Best-effort deterministic settings for reproducible mask counts."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        print(f"[determinism] Warning: {exc}")


def _unload(*objects):
    """Delete objects and reclaim GPU / CPU memory."""
    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ── 1. crop ────────────────────────────────────────────────────────────────
def crop_wing(
    image_path: str,
    text_prompt: str = "wing",
    threshold: float = 0.5,
    device: str | None = None,
) -> tuple[Image.Image, np.ndarray] | tuple[None, None]:
    """Detect and crop the largest wing region from *image_path* using SAM3.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    text_prompt : str
        Text prompt fed to the SAM3 processor (default ``"wing"``).
    threshold : float
        Mask / detection threshold (default ``0.5``).
    device : str | None
        PyTorch device string.  ``None`` → auto-detect.

    Returns
    -------
    tuple[Image.Image, np.ndarray] | tuple[None, None]
        ``(cropped_pil_image, cropped_mask_bool_array)`` on success,
        or ``(None, None)`` if no masks were found.
    """
    from transformers import Sam3Model, Sam3Processor

    model_path = "model_cache/sam3"

    device = _resolve_device(device)
    print(f"[crop_wing] Loading SAM3 model on {device} …")
    model = Sam3Model.from_pretrained(model_path, local_files_only=True).to(device)
    processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)

    try:
        image = Image.open(image_path)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"]
        print(
            f"[crop_wing] Found {len(masks)} object(s) in {os.path.basename(image_path)}"
        )

        if len(masks) == 0:
            print("[crop_wing] No masks found – skipping crop.")
            return None, None

        # Pick the largest mask by pixel area
        largest_idx = max(range(len(masks)), key=lambda i: masks[i].sum().item())
        largest_area = masks[largest_idx].sum().item()
        print(f"[crop_wing] Largest mask is #{largest_idx} (area {largest_area})")

        largest_mask = masks[largest_idx].cpu().numpy()
        y_coords, x_coords = np.where(largest_mask)

        if len(y_coords) == 0:
            return None, None

        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
        cropped_mask = largest_mask[y_min : y_max + 1, x_min : x_max + 1]

        print(f"[crop_wing] Cropped bbox: ({x_min},{y_min}) → ({x_max},{y_max})")
        return cropped_image, cropped_mask

    finally:
        _unload(model, processor)


# ── 2. segment ─────────────────────────────────────────────────────────────
def segment_wing(
    cropped_image: Image.Image,
    wing_mask: np.ndarray,
    overlap_threshold: float = 0.7,
    device: str | None = None,
    return_mask_tokens: bool = True,
    points_per_batch: int | None = None,
) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
    """Run SAM auto-mask generation on a cropped wing image.

    Masks that do not sufficiently overlap the *wing_mask* are discarded, and
    larger masks that mostly duplicate a smaller, already-kept mask are removed.

    Parameters
    ----------
    cropped_image : PIL.Image.Image
        The cropped wing image (output of :func:`crop_wing`).
    wing_mask : np.ndarray
        Boolean / binary mask of the wing region (same spatial size as
        *cropped_image*).
    overlap_threshold : float
        Minimum fraction of a candidate mask that must fall inside
        *wing_mask* for it to be kept (default ``0.7``).
    device : str | None
        PyTorch device string.  ``None`` → auto-detect.

    Returns
    -------
    list[np.ndarray] | list[tuple[np.ndarray, np.ndarray]]
        Deduplicated segmentation masks sorted by area (smallest first).
        If *return_mask_tokens* is True, returns a list of ``(mask, token)``
        pairs, where *token* is the output token per mask.
    """
    from transformers import pipeline

    device = _resolve_device(device)
    print(f"[segment_wing] Loading SAM (facebook/sam-vit-huge) on {device} …")
    generator = pipeline("mask-generation", model="model_cache/sam", device=device)

    def _filter_masks_with_tokens(
        masks: torch.Tensor,
        iou_scores: torch.Tensor,
        original_size: tuple[int, int],
        cropped_box_image: torch.Tensor,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        mask_threshold: float,
        stability_score_offset: float,
        mask_tokens: torch.Tensor,
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
        from transformers.models.sam.image_processing_sam import (
            _batched_mask_to_box,
            _compute_stability_score_pt,
            _is_box_near_crop_edge,
            _mask_to_rle_pytorch,
            _pad_masks,
        )

        original_height, original_width = original_size
        iou_scores = iou_scores.flatten(0, 1)
        masks = masks.flatten(0, 1)
        mask_tokens = mask_tokens.flatten(0, 1)

        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError("masks and iou_scores must have the same batch size.")

        if masks.device != iou_scores.device:
            iou_scores = iou_scores.to(masks.device)

        keep_mask = torch.ones(masks.shape[0], dtype=torch.bool, device=masks.device)

        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_pt(
                masks, mask_threshold, stability_score_offset
            )
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)

        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]
        mask_tokens = mask_tokens[keep_mask]

        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box(masks)

        keep_mask = ~_is_box_near_crop_edge(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )

        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]
        mask_tokens = mask_tokens[keep_mask]

        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        masks = _mask_to_rle_pytorch(masks)

        return masks, scores, converted_boxes, mask_tokens

    def _generate_masks_with_tokens(image: Image.Image):
        from torchvision.ops.boxes import batched_nms
        from transformers.models.sam.image_processing_sam import _rle_to_mask

        preprocess_kwargs, forward_params, postprocess_kwargs = generator._sanitize_parameters()
        if points_per_batch is not None:
            preprocess_kwargs["points_per_batch"] = points_per_batch
        model_inputs = generator.preprocess(image, **preprocess_kwargs)

        all_scores = []
        all_masks = []
        all_boxes = []
        all_tokens = []

        pred_iou_thresh = forward_params.get("pred_iou_thresh", 0.88)
        stability_score_thresh = forward_params.get("stability_score_thresh", 0.95)
        mask_threshold = forward_params.get("mask_threshold", 0)
        stability_score_offset = forward_params.get("stability_score_offset", 1)
        max_hole_area = forward_params.get("max_hole_area", None)
        max_sprinkle_area = forward_params.get("max_sprinkle_area", None)

        for model_input in model_inputs:
            model_input = dict(model_input)
            captured = {}

            def _hook(_module, _inputs, outputs):
                captured["point_embedding"] = outputs[0] if isinstance(outputs, tuple) else outputs

            input_boxes = model_input.pop("input_boxes")
            _ = model_input.pop("is_last")
            original_sizes = model_input.pop("original_sizes").tolist()
            reshaped_input_sizes = model_input.pop("reshaped_input_sizes", None)
            reshaped_input_sizes = (
                reshaped_input_sizes.tolist() if reshaped_input_sizes is not None else None
            )

            model_input = generator._ensure_tensor_on_device(
                model_input, device=generator.device
            )

            handle = generator.model.mask_decoder.transformer.register_forward_hook(_hook)
            model_outputs = generator.model(**model_input)
            handle.remove()

            point_embedding = captured.get("point_embedding")
            if point_embedding is None:
                raise RuntimeError("Failed to capture mask output tokens from SAM.")

            low_resolution_masks = model_outputs["pred_masks"]
            postprocess_kwargs_local = {}
            if max_hole_area is not None:
                postprocess_kwargs_local["max_hole_area"] = max_hole_area
            if max_sprinkle_area is not None and max_sprinkle_area > 0:
                postprocess_kwargs_local["max_sprinkle_area"] = max_sprinkle_area
            if postprocess_kwargs_local:
                low_resolution_masks = generator.image_processor.post_process_masks(
                    low_resolution_masks,
                    original_sizes,
                    mask_threshold=mask_threshold,
                    reshaped_input_sizes=reshaped_input_sizes,
                    binarize=False,
                    **postprocess_kwargs_local,
                )
            masks = generator.image_processor.post_process_masks(
                low_resolution_masks,
                original_sizes,
                mask_threshold=mask_threshold,
                reshaped_input_sizes=reshaped_input_sizes,
                binarize=False,
            )
            iou_scores = model_outputs["iou_scores"]

            num_mask_tokens = generator.model.mask_decoder.num_mask_tokens
            mask_tokens_out = point_embedding[:, :, 1 : (1 + num_mask_tokens), :]
            if iou_scores.shape[2] != mask_tokens_out.shape[2]:
                if iou_scores.shape[2] == mask_tokens_out.shape[2] - 1:
                    mask_tokens_out = mask_tokens_out[:, :, 1 : 1 + iou_scores.shape[2], :]
                else:
                    mask_tokens_out = mask_tokens_out[:, :, : iou_scores.shape[2], :]

            masks, scores, boxes, tokens = _filter_masks_with_tokens(
                masks[0],
                iou_scores[0],
                original_sizes[0],
                input_boxes.to(generator.device)[0],
                pred_iou_thresh,
                stability_score_thresh,
                mask_threshold,
                stability_score_offset,
                mask_tokens_out[0],
            )

            all_scores.append(scores)
            all_masks.extend(masks)
            all_boxes.append(boxes)
            all_tokens.append(tokens)

        if not all_scores:
            return [], []

        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        all_tokens = torch.cat(all_tokens)

        keep_by_nms = batched_nms(
            boxes=all_boxes.float(),
            scores=all_scores,
            idxs=torch.zeros_like(all_boxes[:, 0]),
            iou_threshold=postprocess_kwargs.get("crops_nms_thresh", 0.7),
        )

        all_scores = all_scores[keep_by_nms]
        all_boxes = all_boxes[keep_by_nms]
        all_masks = [all_masks[i] for i in keep_by_nms]
        all_tokens = all_tokens[keep_by_nms]

        output_masks = [_rle_to_mask(rle) for rle in all_masks]
        output_tokens = [t.detach().cpu().numpy() for t in all_tokens]
        return output_masks, output_tokens

    try:
        if return_mask_tokens:
            all_masks, all_tokens = _generate_masks_with_tokens(cropped_image)
        else:
            if points_per_batch is not None:
                results = generator(cropped_image, points_per_batch=points_per_batch)
            else:
                results = generator(cropped_image)
            all_masks = results["masks"]
            all_tokens = []
        print(f"[segment_wing] SAM returned {len(all_masks)} masks total.")

        # Ensure wing_mask matches SAM mask orientation
        if all_masks:
            sam_shape = np.asarray(all_masks[0]).shape
            if wing_mask.shape != sam_shape:
                if wing_mask.shape == sam_shape[::-1]:
                    wing_mask = np.ascontiguousarray(wing_mask.T)
                    print(f"[segment_wing] Transposed wing_mask to {wing_mask.shape}")
                else:
                    raise ValueError(
                        f"wing_mask shape {wing_mask.shape} doesn't match SAM masks {sam_shape}"
                    )

        # ── keep only masks that overlap the wing ──────────────────────
        wing_filtered: list[tuple[np.ndarray, int, np.ndarray | None]] = []
        for idx, mask in enumerate(all_masks):
            mask_np = np.asarray(mask)
            mask_area = int(mask_np.sum())
            if mask_area == 0:
                continue
            overlap = np.logical_and(mask_np, wing_mask).sum() / mask_area
            if overlap >= overlap_threshold:
                token = all_tokens[idx] if return_mask_tokens else None
                wing_filtered.append((mask_np, mask_area, token))

        print(
            f"[segment_wing] {len(wing_filtered)} masks pass wing-overlap "
            f"threshold ({overlap_threshold})."
        )

        # ── deduplicate: prefer smaller masks, subtract overlaps from larger ──
        wing_filtered.sort(key=lambda x: x[1])  # ascending area
        final_masks: list[np.ndarray] = []
        final_tokens: list[np.ndarray] = []

        for mask_np, original_area, token in wing_filtered:
            # Subtract all already-kept (smaller) masks from this one
            remaining = mask_np.copy()
            for kept in final_masks:
                remaining = np.logical_and(remaining, ~kept)

            remaining_area = int(remaining.sum())
            # Keep the remainder if it still has significant area (>10% of original)
            if remaining_area > 0.1 * original_area:
                final_masks.append(remaining.astype(mask_np.dtype))
                if return_mask_tokens and token is not None:
                    final_tokens.append(token)

        print(f"[segment_wing] {len(final_masks)} masks after deduplication.")
        if return_mask_tokens:
            return list(zip(final_masks, final_tokens))
        return final_masks

    finally:
        _unload(generator)


# ── 3. save ────────────────────────────────────────────────────────────────
def save_masks_tiff(
    masks: list[np.ndarray],
    output_path: str,
) -> str:
    """Save a list of binary masks as an OME-TIFF (one channel per mask).

    Parameters
    ----------
    masks : list[np.ndarray]
        Segmentation masks (each 2-D, boolean or uint8).
    output_path : str
        Destination file path.  The extension is normalised to
        ``.ome.tiff`` automatically.

    Returns
    -------
    str
        The (possibly corrected) path the file was actually written to.

    Raises
    ------
    ValueError
        If *masks* is empty.
    ImportError
        If *tifffile* is not installed.
    """
    if not masks:
        raise ValueError("No masks to save.")

    import tifffile

    first = np.asarray(masks[0]).squeeze()
    height, width = first.shape
    num = len(masks)

    # Build a (C, Y, X) uint8 array – 255 where mask is True
    stack = np.zeros((num, height, width), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        m = np.asarray(mask).squeeze()
        stack[idx] = (m > 0).astype(np.uint8) * 255

    # Ensure .ome.tiff extension
    base, ext = os.path.splitext(output_path)
    if not (output_path.endswith(".ome.tiff") or output_path.endswith(".ome.tif")):
        output_path = f"{base}.ome.tiff"

    metadata = {
        "axes": "CYX",
        "Channel": {"Name": [f"mask_{i}" for i in range(num)]},
    }
    tifffile.imwrite(
        output_path,
        stack,
        ome=True,
        photometric="minisblack",
        compression="deflate",
        metadata=metadata,
    )
    print(
        f"[save_masks_tiff] Wrote {output_path} — {num} channels × {height} × {width}"
    )
    return output_path


def save_mask_tokens_npy(
    mask_tokens: list[np.ndarray],
    output_path: str,
) -> str:
    """Save per-mask output tokens as a NumPy .npy file."""
    if not mask_tokens:
        raise ValueError("No mask tokens to save.")

    base, _ = os.path.splitext(output_path)
    if not output_path.endswith(".npy"):
        output_path = f"{base}.npy"

    np.save(output_path, np.stack(mask_tokens, axis=0))
    print(f"[save_mask_tokens_npy] Wrote {output_path} — {len(mask_tokens)} tokens")
    return output_path


def save_masks_png(
    masks: list[np.ndarray],
    output_dir: str,
    prefix: str = "mask_",
) -> list[str]:
    """Save each mask as a numbered PNG in *output_dir*."""
    if not masks:
        raise ValueError("No masks to save.")

    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    for idx, mask in enumerate(masks):
        m = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
        out_path = os.path.join(output_dir, f"{prefix}{idx:03d}.png")
        Image.fromarray(m).save(out_path)
        paths.append(out_path)

    print(f"[save_masks_png] Wrote {len(paths)} mask PNGs → {output_dir}")
    return paths


# ── optional: quick visualisation ──────────────────────────────────────────
def visualise_masks(image: Image.Image, masks: list[np.ndarray]) -> None:
    """Display *masks* overlaid on *image* using matplotlib."""
    import matplotlib.pyplot as plt

    if not masks:
        print("No masks to show.")
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for idx, mask in enumerate(masks):
        mask_np = np.asarray(mask)
        color_rgb = MASK_COLORS[idx % len(MASK_COLORS)]
        rgba = np.array(color_rgb + [0.6])
        h, w = mask_np.shape[-2:]
        overlay = mask_np.reshape(h, w, 1) * rgba.reshape(1, 1, -1)
        ax.imshow(overlay)

        ys, xs = np.where(mask_np.reshape(h, w))
        if len(ys):
            ax.text(
                int(xs.mean()),
                int(ys.mean()),
                str(idx),
                fontsize=12,
                color="white",
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )

    plt.axis("off")
    plt.title("SAM segmentation on cropped wing")
    plt.show()


# ── convenience: run the full pipeline ─────────────────────────────────────
def run_pipeline(
    image_path: str,
    output_cropped_path: str | None = None,
    output_masks_path: str | None = None,
    output_tokens_path: str | None = None,
    output_masks_png_dir: str | None = None,
    text_prompt: str = "wing",
    threshold: float = 0.5,
    overlap_threshold: float = 0.7,
    visualize: bool = True,
    device: str | None = None,
    points_per_batch: int | None = None,
) -> dict | None:
    """Run the full crop → segment → save pipeline.

    Parameters
    ----------
    image_path : str
        Path to the raw input image.
    output_cropped_path : str | None
        If given, save the cropped wing image here.
    output_masks_path : str | None
        If given, save the mask stack as an OME-TIFF here.
    text_prompt : str
        SAM3 text prompt (default ``"wing"``).
    threshold : float
        SAM3 detection threshold.
    overlap_threshold : float
        Minimum wing-overlap ratio for keeping a SAM mask.
    visualize : bool
        Whether to call :func:`visualise_masks` at the end.
    device : str | None
        PyTorch device string.  ``None`` → auto-detect.

    Returns
    -------
    dict | None
        ``{'cropped_image', 'masks', 'wing_mask',
        'cropped_path', 'masks_path'}`` on success, else ``None``.
    """
    _set_deterministic()
    device = _resolve_device(device)

    # 1 — crop
    cropped_image, wing_mask = crop_wing(
        image_path,
        text_prompt=text_prompt,
        threshold=threshold,
        device=device,
    )
    if cropped_image is None:
        print("[pipeline] Wing crop failed – aborting.")
        return None

    if output_cropped_path:
        cropped_image.save(output_cropped_path)
        print(f"[pipeline] Saved cropped image → {output_cropped_path}")

    # 2 — segment
    want_tokens = output_tokens_path is not None or output_masks_png_dir is not None
    if want_tokens:
        masks_with_tokens = segment_wing(
            cropped_image,
            wing_mask,
            overlap_threshold=overlap_threshold,
            device=device,
            return_mask_tokens=True,
            points_per_batch=points_per_batch,
        )
        masks = [m for m, _ in masks_with_tokens]
        mask_tokens = [t for _, t in masks_with_tokens]
    else:
        masks = segment_wing(
            cropped_image,
            wing_mask,
            overlap_threshold=overlap_threshold,
            device=device,
            points_per_batch=points_per_batch,
        )
        mask_tokens = []

    # 3 — save
    masks_path = None
    tokens_path = None
    png_paths: list[str] = []
    if output_masks_path and masks:
        masks_path = save_masks_tiff(masks, output_masks_path)
        output_dir = os.path.dirname(masks_path) or "."
        if output_tokens_path is None and want_tokens:
            base, _ = os.path.splitext(masks_path)
            output_tokens_path = f"{base}_mask_tokens.npy"
        if output_masks_png_dir is None and masks:
            output_masks_png_dir = os.path.join(output_dir, "mask_pngs")

        if output_tokens_path and mask_tokens:
            tokens_path = save_mask_tokens_npy(mask_tokens, output_tokens_path)
        if output_masks_png_dir:
            png_paths = save_masks_png(masks, output_masks_png_dir)

    # 4 — visualise
    if visualize and masks:
        visualise_masks(cropped_image, masks)

    return {
        "cropped_image": cropped_image,
        "masks": masks,
        "mask_tokens": mask_tokens,
        "wing_mask": wing_mask,
        "cropped_path": output_cropped_path,
        "masks_path": masks_path,
        "tokens_path": tokens_path,
        "mask_png_paths": png_paths,
    }


# ── batch processing ────────────────────────────────────────────────────────
def run_batch(
    image_paths: list[str],
    output_dir: str,
    text_prompt: str = "wing",
    threshold: float = 0.5,
    overlap_threshold: float = 0.7,
    device: str | None = None,
) -> list[dict]:
    """Run the pipeline on multiple images.

    Parameters
    ----------
    image_paths : list[str]
        List of paths to input images.
    output_dir : str
        Directory where outputs will be saved. For each input image, a cropped
        image and mask TIFF will be saved using the original filename as a base.
    text_prompt : str
        SAM3 text prompt (default ``"wing"``).
    threshold : float
        SAM3 detection threshold.
    overlap_threshold : float
        Minimum wing-overlap ratio for keeping a SAM mask.
    device : str | None
        PyTorch device string.  ``None`` → auto-detect.

    Returns
    -------
    list[dict]
        List of result dictionaries from :func:`run_pipeline`, one per
        successfully processed image. Failed images are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = _resolve_device(device)

    results = []
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, 1):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n{'=' * 60}")
        print(f"[batch] Processing {idx}/{total}: {image_path}")
        print(f"{'=' * 60}")

        cropped_path = os.path.join(output_dir, f"{basename}_cropped.png")
        masks_path = os.path.join(output_dir, f"{basename}_masks.ome.tiff")

        result = run_pipeline(
            image_path,
            output_cropped_path=cropped_path,
            output_masks_path=masks_path,
            text_prompt=text_prompt,
            threshold=threshold,
            overlap_threshold=overlap_threshold,
            visualize=False,
            device=device,
        )

        if result is not None:
            result["source_path"] = image_path
            results.append(result)
        else:
            print(f"[batch] Skipped {image_path} (no wing detected)")

    print(f"\n{'=' * 60}")
    print(f"[batch] Completed: {len(results)}/{total} images processed successfully")
    print(f"[batch] Outputs saved to: {output_dir}")
    print(f"{'=' * 60}")

    return results


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import glob as glob_module

    parser = argparse.ArgumentParser(description="Wing crop + segmentation pipeline")
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to input image(s). Supports glob patterns (e.g., 'data/*.jpg')",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for batch mode. If provided, runs in batch mode.",
    )
    parser.add_argument(
        "--cropped", help="Save cropped wing to this path (single image mode)"
    )
    parser.add_argument(
        "--masks", help="Save mask OME-TIFF to this path (single image mode)"
    )
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 …")
    parser.add_argument("--prompt", default="wing", help="SAM3 text prompt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--no-viz", action="store_true", help="Disable visualisation")
    args = parser.parse_args()

    # Expand glob patterns in image paths
    expanded_paths = []
    for pattern in args.images:
        matches = glob_module.glob(pattern)
        if matches:
            expanded_paths.extend(sorted(matches))
        elif os.path.isfile(pattern):
            expanded_paths.append(pattern)
        else:
            print(f"Warning: '{pattern}' did not match any files")

    if not expanded_paths:
        parser.error("No valid image files found")

    # Batch mode: multiple images or --output-dir specified
    if args.output_dir or len(expanded_paths) > 1:
        output_dir = args.output_dir or "output"
        run_batch(
            expanded_paths,
            output_dir=output_dir,
            text_prompt=args.prompt,
            threshold=args.threshold,
            overlap_threshold=args.overlap,
            device=args.device,
        )
    else:
        # Single image mode
        run_pipeline(
            expanded_paths[0],
            output_cropped_path=args.cropped,
            output_masks_path=args.masks,
            text_prompt=args.prompt,
            threshold=args.threshold,
            overlap_threshold=args.overlap,
            visualize=not args.no_viz,
            device=args.device,
        )
