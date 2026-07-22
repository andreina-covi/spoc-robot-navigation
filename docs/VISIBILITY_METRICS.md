# Visibility metrics in `navigation-*.csv`

These columns describe how much of each object is visible in the **nav camera** frame.
Collection exports **all named non-structural** FOV detections that have at least one
mask pixel; it does **not** hard-filter on size or occupancy. Post-processing decides
keep/drop using `visible-pixels`, `min-side`, and `occupancy-ratio`.

Structural meshes (Wall/Floor/Ceiling/…) and numeric-only ids (e.g. `2|4`) are still
dropped at collection via `is_exportable_object`.

Implementation: `collector.py` (`get_visible_pixels_from_bbox`, `get_object_data`).

Per-run camera geometry lives under `episode_meta.camera`:

- `width` / `height` — nav frame size (default 396×224)
- `frame_size_px` — \(W \times H\)
- `fov_vertical_deg` — vertical FOV (default 59°)

---

## Inputs

From nav `instance_detections2D`, each object has a 2D box `[cmin, rmin, cmax, rmax]`:

$$
w = cmax - cmin,\quad h = rmax - rmin,\quad A_{\mathrm{bbox}} = w \cdot h
$$

Visible pixels are counted inside that crop of `instance_segmentation_frame` where the
pixel color matches the object’s instance color (`dict_colors`).

Objects with `visible-pixels == 0` are skipped (no nav row for that object this step).

---

## 1. `visible-pixels`

**Meaning:** how many segmentation pixels of the object are currently visible inside its bbox.

$$
\text{visible-pixels} = \#\{\text{pixels in crop where color} = \text{object color}\}
$$

This is mask-based (not bbox area). It shrinks under occlusion and truncation.

---

## 2. `bbox-area`

**Meaning:** area of the detection box (visible bbox), in pixels.

$$
\text{bbox-area} = (cmax - cmin)\,(rmax - rmin)
$$

Usually \(\text{bbox-area} \ge \text{visible-pixels}\). A large gap often means a loose box
or partial occlusion inside the box.

---

## 3. `min-side`

**Meaning:** shorter side of the bbox, in pixels.

$$
\text{min-side} = \min(cmax - cmin,\ rmax - rmin)
$$

Useful to drop thin edge peeks (e.g. a 200×3 sliver).

---

## 4. `occupancy-ratio`

**Meaning:** fraction of the bbox filled by the object’s mask pixels.

$$
\text{occupancy-ratio} = \mathrm{round}\!\left(\frac{\text{visible-pixels}}{\text{bbox-area}},\ 3\right)
$$

- Near `1` → mask fills most of the box.
- Low value → sparse / heavily occluded content inside a larger box.

---

## Mental model

```text
visible-pixels   →  “how many mask pixels do I see?”
bbox-area        →  “how large is the detection box?”
min-side         →  “is this just a thin edge peek?”
occupancy-ratio  →  “how full is the box with real object pixels?”
```

---

## Export policy

| Output | Policy |
|--------|--------|
| `navigation` / `objects` CSV | All **named non-structural** FOV objects with `visible-pixels > 0` + metrics |
| Keep / drop for training or QA | **Post-processing** using the three metrics above |

Collection does not apply recognizability thresholds to nav export. Displacement tracking
uses FOV presence separately (see [DATA_COLLECTION.md](DATA_COLLECTION.md)).

---

## Example post-process keep rules

Tune cutoffs offline as needed; nothing below is enforced at collection time.

```python
df = df[
    (df["visible-pixels"] >= 100)
    & (df["min-side"] >= 12)
    & (df["occupancy-ratio"] >= 0.3)
]
```

---

See also: [DATA_COLLECTION.md](DATA_COLLECTION.md) (export layout, displacement, QA).
