# Continue slice numbering across multiple slides

## Problem

One brain is often imaged across more than one CZI. The section widget numbers
every load from 1, so exporting the second slide overwrites `s001..sNNN` from the
first. The user wants to section slide A, then slide B, and have B's crops land
after A's in the same `stitched/` tree.

## Approach

Detect the continuation point at export time by looking at what is already on
disk. No cross-file state is kept in the widget, and no session has to be resumed
— the output tree is the source of truth.

Rejected: merging two CZIs into one virtual canvas (large change to the readers,
and stitching geometry is not knowable from the files); carrying a session across
loads (state to persist and invalidate, for no gain over reading the tree).

## Design

### `numbering.next_slice_number(out_base, animal_id) -> int`

Pure function, filesystem-only, no napari.

Globs `out_base/*/{animal_id}_s*.tif` and returns `max(NNN) + 1`, parsing the
digits after the final `_s`. Returns `1` when the directory is missing, empty, or
holds nothing matching the pattern.

The maximum is taken **across all channel subdirectories**, not per channel, so
an interrupted export that wrote `green/` but not `cy3/` cannot cause the
channels to drift apart on the next run.

### Wiring in `_section_widget._export`

After `out_base` is computed:

```python
offset = next_slice_number(out_base, animal_id) - 1
fname = f"{animal_id}_s{str(num + offset).zfill(ZERO_PAD)}.tif"
```

The offset applies **only** to the output filename. `compute_slice_numbers`,
manual click-numbering, the on-screen labels, and the `1..N` contiguity
validation all keep operating on local numbers and are unchanged. This is what
keeps the change small.

The closing notification reports the range written — `wrote 12 slices
(s013-s024) -> {out_base}` — so continuation is visible without opening the
folder.

## Consequences

Re-exporting the same slide **appends rather than overwrites**. Export 12 slices,
spot a bad rectangle, fix it, export again, and you get `s013-s024` beside the
original `s001-s012`; the old behaviour silently overwrote. Append is the correct
default for the multi-slide workflow this serves, and overwriting is the more
destructive default, but a botched export now needs a manual delete. Accepted.

## Testing

`tests/test_numbering.py`, using `tmp_path`:

- missing directory → 1
- empty directory → 1
- `s001..s012` present → 13
- ragged channels (`green/` has `s001..s012`, `cy3/` only `s001..s010`) → 13
- filenames not matching the pattern ignored
- a different `animal_id`'s files ignored
- past the zero-pad width (`s999` → 1000), since `zfill` only pads
