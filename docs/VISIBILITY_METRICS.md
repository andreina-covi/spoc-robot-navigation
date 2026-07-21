# Visibility metrics in `navigation-*.csv`

These columns describe how much of each object is visible in the **nav camera** frame.
They are written for **named non-structural** FOV detections so **post-processing** can
decide keep/drop (collection does not hard-filter on visibility thresholds for export).
Numeric-only ids (e.g. `2|4`) and Wall/Floor/etc. are dropped at collection time.

Implementation: `collector.py` (`compute_visibility_metrics`,
`estimate_full_silhouette_from_size_distance`). Suggested thresholds for displacement /
post-processing live in `episode_meta.visibility_filters` and are applied by
`passes_visibility_filters` only for invisible-displacement discovery.

Suggested default thresholds also live in `annotations/episode_meta-*.json` under
`visibility_filters.suggested_postprocess_thresholds`.

Per-run camera geometry (prefer these over hardcoding) is under `episode_meta.camera`:

- `width` / `height` — nav frame size (default 396×224)
- `frame_size_px` — \(W \times H\)
- `fov_vertical_deg` — vertical FOV used for silhouette projection (default 59°)

Agent step sizes used during collection are under `episode_meta.agent`
(`movement_constant`, `rotation_deg`, …).

---

## Inputs

From nav `instance_detections2D`, each object has a 2D box `[cmin, rmin, cmax, rmax]`:

\[
w = cmax - cmin,\quad h = rmax - rmin,\quad A_{\mathrm{bbox}} = w \cdot h
\]

Stretch nav frame size (from `episode_meta.camera`, defaults below): \(W = 396\), \(H = 224\), so

\[
A_{\mathrm{frame}} = 396 \times 224 = 88704
\]

Vertical FOV used for silhouette estimate: \(59^\circ\) (`INTEL_VERTICAL_FOV`).

---

## 1. `visible-area-px`

**Meaning:** how many pixels of the object are currently visible in the image.

**Fast mode (default):**

\[
\text{visible-area-px} = A_{\mathrm{bbox}} = (cmax - cmin)\,(rmax - rmin)
\]

**Strict mode (optional):** uses the instance segmentation mask instead:

\[
\text{visible-area-px} = \#\{\text{pixels where }\mathtt{instance\_masks}[\mathrm{object}] \text{ is True}\}
\]

**Caveat:** the detection bbox is already the box of **visible** pixels, so it shrinks under
occlusion; it is not the full object box.

---

## 2. `visible-frac`

**Meaning:** fraction of the **whole image** covered by that visible area.

\[
\text{visible-frac} = \frac{\text{visible-area-px}}{A_{\mathrm{frame}}}
\]

Example: 800 visible px → \(800 / 88704 \approx 0.009\).

This answers “how big is this blob in the frame?”, **not** “how much of the object is shown”.

---

## 3. `full-silhouette-px`

**Meaning:** estimated area the object would occupy on screen if it were **fully visible /
unoccluded**.

### Default method (size ÷ distance)

From metadata `distance` \(d\) and AABB/OBB `size` \((s_x, s_y, s_z)\):

1. Take the two largest extents \(a \ge b\).
2. Focal length from vertical FOV:

\[
f_y = \frac{H/2}{\tan(\mathrm{FOV}_y / 2)}
\]

3. Project extents:

\[
p_a = f_y \cdot \frac{a}{d},\quad p_b = f_y \cdot \frac{b}{d}
\]

4. Area:

\[
\text{full-silhouette-px} = p_a \cdot p_b
\]

Returned only if \(> 1\); otherwise `None` (missing size/distance or degenerate).

**Caveat:** this is an **approximation** of “full object on screen”, not a second render with
nothing in front. It can overestimate (loose box) or underestimate (object partly out of
frame, odd orientation).

---

## 4. `unoccluded-ratio`

**Meaning:** estimated **shown fraction of the object** (visible vs full).

\[
\text{unoccluded-ratio} =
\min\left(1,\ \frac{\text{visible-area-px}}{\text{full-silhouette-px}}\right)
\]

- `None` if `full-silhouette-px` cannot be estimated.
- Low value ≈ only a small part of a large object is showing (e.g. table corner).
- High value ≈ most of the object’s expected screen footprint is visible.

**Naming note:** the value mixes occlusion, truncation at image borders, and bbox
approximation—it is not a pure raycast “unocclusion” score. A clearer conceptual name is
“shown fraction of the object”.

### Suggested post-process defaults (`Collector` / `episode_meta`)

| Constant | Default | Notes |
|----------|---------|--------|
| `min_frame_fraction` | `0.009` (~0.90% of frame) | Also drives `min_mask_pixels` |
| `min_mask_pixels` | \(A_{\mathrm{frame}} \times 0.009\) ≈ **798** | Absolute visible area |
| `min_bbox_side` | `12` px | Drops thin edge peeks |
| `min_unoccluded_ratio` | `0.4` | Shown fraction of object |

Confirm against `episode_meta.visibility_filters` for a given run (values can be overridden
at `Collector` construction).

---

## Mental model

```text
visible-area-px    →  “how many pixels do I see?”
visible-frac       →  “how much of the image is that?”
full-silhouette-px →  “how big should the whole object look?”
unoccluded-ratio   →  “what fraction of the object am I seeing?”
```

---

## Export vs displacement

| Use | Policy |
|-----|--------|
| `navigation` / `objects` CSV | Export **named** non-structural FOV objects + metrics; post-process filters |
| Invisible displacement “seen” | Uses `filter_recognizable_detections` / `passes_visibility_filters` |

Structural types (Wall, Floor, Ceiling, Room, …) and numeric-only ids (`2|4`) are never
exported as spatial-relation objects.

---

## Example post-process keep rules

```python
df = df[
    (df["visible-area-px"] >= 798)
    & (df["unoccluded-ratio"].fillna(1.0) >= 0.4)
]
```

---

## Possible refinements

1. Prefer **mask pixels** for `visible-area-px` even in fast mode (more accurate than bbox).
2. Cap `full-silhouette-px` by frame size when the object is partly off-screen.
3. Rename `unoccluded-ratio` → `shown-fraction` in CSV/docs if the name confuses consumers.

See also: [DATA_COLLECTION.md](DATA_COLLECTION.md) (export layout, displacement, QA).
