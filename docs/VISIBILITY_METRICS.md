# Visibility metrics in `navigation-*.csv`

These columns describe how much of each object is visible in the **nav camera** frame.
Collection exports **non-structural** FOV detections that have at least one
mask pixel, including numeric ids such as `6|5`. It does **not** hard-filter on
size or occupancy. Post-processing (cm-benchmark) decides keep/drop from these columns. `mark-x` / `mark-y`
name the mask pixel farthest from the mask edge. Each row also carries the object's
**expected** (unoccluded, projected) box and `in-frame-fraction`, the share of that box
inside the frame.

Wall, Floor, Ceiling, and Room are dropped at collection via `is_exportable_object`.

Implementation: `collector.py` (`get_visible_pixels_from_bbox`, `get_object_data`) and
`geometry.py` (`expected_bbox_from_3d`, `angular_size_deg`).

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

## 5. `mark-x`, `mark-y`

**Meaning:** full-image coordinates of the mask pixel farthest from the mask edge
(the most interior visible pixel of that object in the crop).

Computed from the same crop used for `visible-pixels`
(`np.all(crop == color, axis=2)`). The pixel is the maximum of
`cv2.distanceTransform` on that mask, with a one-pixel zero border so the crop
edge counts, shifted by the bbox origin into full-image coordinates:

$$
\text{mark-x} = cmin + c_{\mathrm{local}},\quad
\text{mark-y} = rmin + r_{\mathrm{local}}
$$

No extra image is rendered. Ties take the first maximum in row-major order
(`np.argmax`).

---

## 6. `expected-bbox-area`, `expected_*`, `ang-width-deg`, `ang-height-deg`

**Meaning:** the box the whole object would cover with nothing in front of it.

The object's 3D box (its oriented box, or the axis-aligned box when there is none) is
projected with the nav camera pose and focal length. `expected_cmin` / `expected_cmax` /
`expected_rmin` / `expected_rmax` are its pixel bounds clamped to the frame, and
`expected-bbox-area` their area. `ang-width-deg` / `ang-height-deg` are the angular size
of that clamped box. Corners closer than 5 cm to the camera plane are left out. All are
empty when the box cannot be projected.

---

## 7. `in-frame-fraction`

**Meaning:** how much of the object's unoccluded projected box lies inside the frame
(1 = not cut by the image border).

$$
\text{in-frame-fraction} = \min\left(1, \frac{\text{expected box area clamped to the frame}}{\text{expected box area unclamped}}\right)
$$

It is **0** when any box corner is behind the camera plane (an object beside or behind
the agent: the unclamped box would cover only its front part and overstate the share),
or when the box cannot be projected. Occlusion does not lower it; `visible-pixels` covers
occlusion. cm-benchmark uses it to require that a named object is mostly inside the frame.

---

## Mental model

```text
visible-pixels   →  “how many mask pixels do I see?”
bbox-area        →  “how large is the detection box?”
min-side         →  “is this just a thin edge peek?”
occupancy-ratio  →  “how full is the box with real object pixels?”
mark-x, mark-y   →  “which mask pixel is farthest from the mask edge?”
in-frame-fraction → “how much of the whole object is inside the frame?”
```

---

## Export policy

| Output | Policy |
|--------|--------|
| `navigation` / `objects` CSV | Non-structural FOV objects with `visible-pixels > 0`, including numeric ids, plus the metrics and mark |
| `object_state` | **Pickupables** only; eligibility uses the same mask-pixel signal (`in_camera_fov`) |
| Keep / drop for training or QA | **Post-processing** (cm-benchmark) using these columns (`mark-x` / `mark-y` are location hints, not keep/drop filters) |

Collection does not apply recognizability thresholds beyond ``visible-pixels > 0``
for nav export. Pickupable tracking uses the same mask-pixel signal
(``in_camera_fov``); hidden tracks live in ``object_state`` (see
[DATA_COLLECTION.md](DATA_COLLECTION.md)).

---

## Where the cutoffs live

Nothing is enforced at collection time. The keep/drop rules and their thresholds are in
cm-benchmark (its visibility filter and `configs/generation.yaml`), so they can be retuned
without collecting again.

---

See also: [DATA_COLLECTION.md](DATA_COLLECTION.md) (export layout, placement trials, QA).
