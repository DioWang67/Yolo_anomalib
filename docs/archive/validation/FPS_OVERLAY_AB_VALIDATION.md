# FPS Overlay A/B Validation

## Status

Historical validation record. The one-time validation utility was retired after
the camera capture path stopped drawing preview text onto inference frames.

## Validation

- Date: 2026-06-08
- Target: PCBA1 field images
- Samples: 26
- Comparison: recorded image with the FPS overlay versus a copy with the
  top-left green overlay removed
- Result: 26/26 samples retained the same inspection outcome

The utility compared final status, detected classes and bounding-box centre
drift while running the normal `DetectionSystem` on file-backed frames. Camera
initialisation was disabled during the comparison.

## Resulting rule

Camera frames consumed by inference and colour checks must remain undecorated.
FPS and other status text belong in the preview layer and must be drawn on a
display copy.

## Evidence limitation

The original per-image console output was not retained in the repository. This
record preserves the result referenced by the camera implementation, not the raw
measurement rows. Repeat the validation with current golden images if the frame
capture or overlay path changes again.
