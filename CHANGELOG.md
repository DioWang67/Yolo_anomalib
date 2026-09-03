# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- One place decides whether a stored color baseline may be trusted:
  `core/color_baseline_contract.py` owns the current algorithm version and the
  single reader of the algorithm an artifact records for itself. A baseline is
  not a free-standing table of numbers -- `coverage_mean` and the sampled
  envelopes only mean anything against the crop geometry they were measured on
  -- and pairing statistics from one geometry with a runtime that measures
  another does not fail loudly, it shifts every score. Three parties now consult
  the same function instead of each comparing its own imported constant.
- The station setting `color_baseline_algorithm_enforcement` decides what the
  runtime does with a baseline whose algorithm cannot be shown to be current:
  `warn` (the default) records it and carries on, `strict` refuses to load it.
  The default is permissive on purpose -- a code update alone must not be able
  to stop a line that needs a rebuilt baseline first -- so it is meant to be set
  to `strict` once such a baseline is deployed. An unrecognized value resolves
  to `strict`, because the key is absent unless somebody set it and honouring a
  typo as `warn` would grant the opposite of what the config asked for.

### Fixed
- A disabled position configuration is no longer used as spatial evidence for a
  missing item. `build_missing_item_locations()` read `expected_boxes` without
  consulting `enabled`, so a station that had deliberately turned position
  checking off still got missing parts annotated at coordinates nobody was
  maintaining -- boxes drawn with the authority of a check that was not running.
  `ResultHandler` applies the same gate at its own lookup.
- Repeated expected positions no longer collapse onto one another. `used_keys`
  started empty, so an expected box already claimed by a real detection could be
  handed to a missing item as well; it is now seeded from the detections' own
  `position_expected_key`.
- The annotation panel names every missing item, not only those a location could
  be resolved for. The `Missing:` line was built from `missing_locations`, so an
  item with no usable expected box -- the ordinary case once the gate above
  applies -- vanished from the operator's summary instead of being named without
  a box. `missing_locations=None` still means "the caller resolved nothing, fall
  back to expected boxes", while an empty list is now an explicit "draw no
  locations".
- Operator version activation and rollback no longer drop the station's color
  contract fields. `STATION_LOCAL_FIELDS` in
  `core/services/model_version_registry.py` carried no color field at all while
  activation overwrites the live `config.yaml`, and no version snapshot on disk
  carried `color_roi_policy` -- so switching or rolling back a model version
  silently republished the default geometry over a station running an inset.
  Under v5 the deployed baseline then describes a geometry the station no longer
  measures in: `strict` raises and the line stops, `warn` keeps scoring through
  the wrong crop. For these three fields the live station wins including its
  absence, so a snapshot cannot supply a foreign value either.
- Baseline rebuilds cover the colors the station actually inspects, taken from
  its `expected_items`, instead of a hardcoded five. A station with a different
  palette rebuilt the wrong set: colors it does inspect kept their old
  statistics, and the report named colors it does not have. Repeated positions
  collapse to one color and generic detector classes are excluded, which is the
  same line the runtime draws when it narrows the palette.
- Adding a color the deployed baseline cannot score is caught at startup and
  names the color. The runtime folds an unscoreable candidate into every item's
  verdict, so one unknown or misspelled name in `expected_items` made every
  board fail the color check indefinitely -- under `warn` as much as `strict` --
  and nothing compared the configured colors against the baseline's vocabulary.
  A rebuild still refuses to introduce a color the base baseline lacks, and now
  says why: there is no predecessor to measure drift or regression against, and
  nothing to fall back to when evidence runs short.
- `StatsColorBaselineRebuilder.build()` refuses to write a baseline with no
  `color_roi_policy` stamp. The runtime always resolves a geometry and always
  compares it to the recorded one, so an unstamped artifact loads at no station;
  it was being discovered after the sign-off, when the evidence was gone.
- The acceptance picker and the publication gate now ask what the line asks.
  Both checked only the algorithm label, which says which rules measured the
  statistics and nothing about the geometry or tuning they were measured under,
  so a baseline could be offered, selected, accepted over a full sample set, and
  then refused by the runtime. Both now pass the live station's
  `color_roi_policy` and complete `color_decision_tuning`, resolved through one
  shared `core/services/station_color_settings.py`; the candidate door also
  reads the artifact's own provenance rather than trusting the store's label,
  because a candidate whose statistics were never rewritten has been seen
  carrying a current label. The component catalog stops showing such a candidate
  as VERIFIED.
- `_supported_candidates` no longer collapses "a palette was configured and the
  model can score none of it" into "no restriction". That widened a wrong
  palette into the full vocabulary and reported a measurement taken against
  colors nobody asked for; it now returns an empty palette, which the checker
  fails closed on. `None` stays reserved for a station that configured no color
  palette at all, generic detector classes included, so a station naming only
  those is unaffected.
- `tools/color_verifier.py` stopped applying two rules the runtime no longer
  has. Its envelope report short-circuited on the black and yellow indicators,
  returning that color's raw coverage as a confidence with every other color
  zeroed -- so `envelope_ratios`, documented as a per-color report, carried one
  number and four zeros. Both indicators are now reported without adjusting any
  score. Its sampling margin was 0.12 while the runtime measured at 0.15, which
  made the report and the verdict describe different regions of the same part;
  it now takes the runtime's constant. `ORANGE_RED_TIE_MARGIN` and
  `GREEN_DOMINANCE_RATIO`, left behind unreferenced when the verdict moved to
  the runtime, are gone.
- Stats Color's baseline contract is now `stats-robust-v5`. Calibration and
  runtime share one per-axis center-crop helper, so an elongated 300x100 ROI is
  measured as 210x70 on both sides instead of 210x70 during rebuild and 270x70
  at runtime. Every v4 artifact is intentionally incompatible with v5.
- Baseline rebuild and holdout validation now use the live station's complete
  resolved `color_decision_tuning`, and artifacts record that full effective
  mapping plus the tuning behind any preserved base colors. Strict runtime
  loading rejects missing or different tuning, including a same-path config
  change that would otherwise reuse the cached checker.
- The Yellow shortcut no longer returns before the other colors are scored.
  Its Yellow/Orange ratios remain in diagnostics with no score adjustment so
  Cable1/A regression can measure the shortcut's removal before a replacement
  tie-break is calibrated.
- `StatsColorChecker` now distinguishes no restriction (`allowed_colors=None`)
  from an explicit empty or wholly unsupported vocabulary, which fails closed.
  The unused Black decision knobs and their misleading example configuration
  were removed; v5 Black remains driven solely by learned S/V, LAB and coverage.
- Color verification without an explicit product palette now scores the full
  model vocabulary instead of using YOLO's class as its only candidate, which
  made the verifier circular and could hide a Red/Orange disagreement. Every
  explicitly configured color must now exist in the loaded model; a partial
  intersection no longer silently drops misspelled or absent colors.
- Strict baseline compatibility now binds the artifact to its complete
  `color_roi_policy`, not only to the algorithm label. Rebuild artifacts
  record their geometry and the base geometry behind preserved colors; runtime
  policy changes re-run compatibility even when the model path is unchanged.
- Strict color-baseline enforcement now treats a missing configured color model
  as a configuration error. It no longer silently disables color inspection,
  which lets detector-only bundles fail closed until the station's approved v5
  baseline is present.
- Color baseline rebuilds took their sampling geometry from the model version
  config snapshot, which cannot carry it. `color_roi_policy` is a station-local
  field that a model deployment preserves rather than replaces, so no snapshot
  had ever recorded one, and every rebuild resolved it to the default of no
  inset -- sampling the full detector bbox, the exact geometry the policy exists
  to move away from. It failed silently, because every resulting statistic is
  well-formed and only measured somewhere else. With the correct geometry the
  same 215 samples rebuild all five colors; with the snapshot's default all five
  were rejected by the drift guard and reverted. The guard was right and has not
  been loosened: the input was wrong. The geometry now comes from the live
  station config, a missing one refuses the rebuild rather than defaulting, and
  the report records which file it came from.
- A candidate could claim the current algorithm for statistics it had not
  produced. A rebuild that preserves a color copies that color's numbers from
  the base, so the file honestly records the current algorithm while part of its
  statistics were measured by whatever built the base. One rebuild whose five
  colors were all preserved came out byte-identical to the deployed baseline,
  stamped as current, and passed every gate -- the gate vouching for the thing
  it exists to catch. Candidates now record `preserved_colors` and
  `base_algorithm`, and a mixture is compatible only when the base was current
  too, which still admits the ordinary case of preserving a color that ran short
  of evidence.
- The compatibility gate covered one of the three doors into "compare or publish
  a color baseline". Baseline candidates were checked; color profile packages and
  release publication were not. A package built from an excluded candidate stayed
  selectable while its own source was withheld -- packaging copies the statistics
  but not the geometry behind them -- and because such a package still carries
  every required statistic, it would not have failed closed: it would have
  produced a healthy-looking, wrong comparison. Publication was worse, being the
  only door that reaches the line: the builder authenticated schema and sha256
  and never looked at the algorithm, so any acceptance report written before the
  current algorithm could still publish the baseline it had compared. Both are
  now gated, publication preferring what the report recorded over the artifact as
  it stands today, since an artifact can be rebuilt in place after acceptance ran.
- Acceptance reports record the algorithm behind each stored color model, so a
  report stays judgeable once the rebuild algorithm moves on. Reports written
  before this field simply have no entry, which reads as "cannot be established"
  rather than as compatible.
- The rebuild dialog shows why a color needs a human look. Hue spread, chroma
  collapse, a low dominant fraction and a holdout floor miss were recorded as
  machine codes in the report file only, so `REVIEW_REQUIRED` asked the operator
  for a judgement while withholding what the judgement was about. The measured
  values go to the tooltip so a row stays scannable.
- Cross-implementation conformance tests between the runtime color checker and
  the training-pipeline color gate. The two are separate code bases judging the
  same product, so a decision rule that moves on one side and not the other lets
  a model clear the gate and behave differently on the line, and neither
  repository can notice on its own. Both carry a byte-identical
  `tests/fixtures/color_conformance.json` with the color model embedded, so
  each suite is self-contained; the workspace CI asserts the two copies still
  match. Legitimate differences are recorded under `known_divergences` with the
  reason rather than smoothed away.
- Acceptance color discovery now reports the in-scope artifacts it withholds
  instead of dropping them silently. `discover_color_variants()` returns a
  `ColorVariantDiscovery` carrying both the selectable variants and a
  `ColorVariantExclusion` for each in-scope baseline built by a superseded
  algorithm, and the matrix dialog lists those as unselectable rows stating
  why. Out-of-scope artifacts are deliberately not reported, and a revoked
  revision stays unlisted because the operator already knows it was revoked.
- The matrix dialog states whether the selected scope has any stored color
  model at all, so a station that never had a color baseline is no longer
  indistinguishable from a broken tool.
- The acceptance window's main page now has a 顏色模型 selector, so a single
  inference run can be pinned to one stored color model instead of always using
  whatever the station has active. The default entry keeps the previous
  behavior, unusable entries are listed but cannot be selected, and the chosen
  model travels with its version-matched model config because the override
  cannot be staged without one. Nothing here activates or edits a color
  version.

### Fixed
- Color ROIs are now cropped by a shared, config-driven `ColorRoiPolicy`
  instead of by each caller's own margins. Cable1/A insets 20% from each side
  horizontally, nothing vertically, with an 8 px floor, because the wires run
  horizontally and a box drawn around one routinely catches its neighbour --
  measured at 30-50% of each crop's chromatic content. Inference, the rebuild
  dialog and headless recalibration all read the same policy, so the geometry a
  baseline was calibrated on is the geometry the line measures with, and
  deployment preserves an approved `color_roi_policy` rather than letting a
  training output overwrite the station's sampling geometry silently.
- Black is decided from its learned S/V and LAB envelope, normalized by the
  baseline's `coverage_mean`, in both the runtime and the training gate. The
  hand-written `s < 50 & v < 80` shortcut is gone. It was neither learned nor a
  clean rule: its statistical baseline scored 3.6% on its own holdout while the
  hand-set threshold had about 1% of headroom on real crops, which is what made
  an earlier consistency cleanup able to reject every good board. A missing or
  malformed `coverage_mean` now fails closed instead of scoring against a
  reference that does not exist.
- The rebuilder moves to `stats-robust-v4` and gains an absolute per-color
  holdout floor of 0.90 alongside the existing relative check. The relative
  rule alone kept a baseline that was no better than a bad predecessor: Black
  sat at 3.6% indefinitely because each new proposal was merely "not an
  improvement". Black's hue drift is now reported as `null` rather than 0.0,
  since hue is undefined for it and a zero read as "no drift".

  Measured on the 250 confirmed Cable1/A acceptance samples: false rejects fall
  from 16 to 4 on the deployed baseline -- the gain is the ROI policy and Black
  v4, not the baseline itself -- and to 3 with a v4 rebuild. Escapes stay at
  0/77 throughout. A v4 candidate reaches READY with all five colors REBUILT
  and 100% holdout, no safety preserve and no review-required color.
- Reverted two changes to the black shortcut that looked like cleanups and
  were not. Reporting the fired rule's margin instead of coverage, and sharing
  the center crop with the other paths, each moved black's score by more than
  the ~1% of headroom its threshold has on real crops: a genuine black region
  went from 0.50 to 0.02, and every good board on the Cable1 acceptance set
  was rejected. Coverage is what the threshold was calibrated against and is
  restored; the incoherence of scoring a mean-rule decision with coverage is
  now answered by naming the rules that fired in `debug.black_rules`, which
  costs nothing, rather than by moving the number the verdict depends on.
  Unifying the three center crops remains worth doing, but it is a
  recalibration and has to move the threshold in the same change.
- `tools/color_verifier.py` now reports the verdict the line actually reaches.
  It calls `StatsColorChecker` -- the same object the inspection pipeline uses
  -- instead of carrying its own scoring. The two had drifted into different
  policies and agreed on only 8 of the 15 shared conformance cases: a plain
  bright red came back `Unknown`, and a red region catching an orange edge came
  back `Orange` with high confidence. The envelope check it used to decide with
  survives as its own reported signal, `envelope_ratios` and `envelope_match`,
  so "does this sit inside the baseline's recorded range?" is answered beside
  the runtime's verdict rather than in place of it. The parallel decision chain
  -- `_initial_prediction`, `_apply_color_rules` and the Orange/Red and Green
  rules -- is deleted rather than left unreachable, since an idle second
  opinion is how the two drifted apart. `--edge-margin`, `--sat-threshold` and
  `--min-valid-pixels` now shape only the envelope report;
  `--ratio-threshold` still shapes the verdict, as the runtime's default
  threshold.
- A color's baseline is now sampled from the crop's dominant hue rather than
  from every saturated pixel in it. A detection box drawn around one wire
  routinely catches part of the next, and the sampler fed those neighbours
  straight into the named color's statistics: in the stored evidence this is
  the norm, not an edge case, with Red spanning 2..78 and Green 16..94 in the
  same file and Red's mean landing on 20.7 -- amber. The dominant hue is
  located on a circular histogram so a color sitting on the 0/179 seam is
  found as one cluster instead of being split and half discarded. Crops of a
  single color are unaffected. How much of each crop's chromatic content was
  kept is recorded as `dominant_fraction_mean`, and a color whose evidence was
  mostly *not* the color it names sets the candidate to REVIEW_REQUIRED --
  the baseline is clean either way, but boxes containing more neighbour than
  subject are a detection problem the reviewer should hear about. The value
  reported is the rejected proposal's, not the preserved baseline's, since
  that is the evidence being judged. `ALGORITHM_VERSION` moves to
  `stats-robust-v3`: baselines built before and after are not comparable.
- `tools/color_verifier.py` now states, at the top of the file, that it does
  not answer "what does the line decide?". It applies a stricter
  envelope-based policy than the runtime -- every color matched against the
  recorded `hsv_min`/`hsv_max` box, with anything under `MIN_HSV_MATCH_RATIO`
  zeroed -- while the runtime uses hand-tuned open-ended gates for red, orange
  and green. Measured on the shared conformance cases the two agree on 8 of
  15: the tool reports `Unknown` for a plain bright red whose V sits just past
  the baseline's recorded 99th percentile, and reports `Orange` with high
  confidence for a red region catching an orange edge. Neither policy is wrong
  in itself, but the name invited the other reading and nothing recorded the
  difference. `tests/test_color_verifier_divergence.py` pins it so it stays a
  documented choice rather than something found the hard way during a
  confusing tool run.
- A rebuilt color baseline whose evidence does not look like a single color
  now asks for human review instead of arriving marked READY. The existing
  safety check measures *drift* -- how far the new center moved from the
  approved one -- and misses pollution that leaves the center in place: in the
  stored evidence one Green proposal drifted 13.7, inside the 18.0 limit,
  while its hue ran from 5 to 173. Two complementary measurements are now
  recorded per color and reported in `report.json`: `hue_spread`, the circular
  width of the hue evidence, and `chroma_retention`, the saturation kept
  relative to the approved baseline. Either one out of range sets the
  candidate to REVIEW_REQUIRED, which the rebuild dialog already renders as
  「需人工檢查」. Neither ever rejects a rebuild on its own: the thresholds are
  calibrated on ten stored baselines, and wrongly blocking a good rebuild is a
  production problem too. Applied retrospectively they flag all eight
  problematic stored candidates and leave the one approved baseline untouched.
- A color baseline is no longer built out of background pixels. When fewer
  than `sample_size` pixels in a crop matched the kind of pixel the color is
  made of, `_sample_color_pixels()` discarded its mask and sampled *every*
  pixel instead -- so a crop that is mostly unsaturated background produced a
  "color baseline" made of that background, while the coverage recorded
  alongside it reported 0.00. Such a crop is now dropped, the dropped ids are
  recorded in the summary as `skipped_crops`, `count` reports the crops the
  statistics were actually built from rather than the crops offered, and a
  color with no usable crop at all fails the rebuild instead of producing a
  signable baseline. The pixel floor itself is unchanged and now named, since
  raising it would reject crops the line currently accepts.
- Color check no longer passes a region it never measured. When no pixel
  cleared the saturation gate, `StatsColorChecker` answered with a hard-coded
  `black: 0.7` -- a score picked to clear black's own threshold, invented for a
  ROI carrying no color evidence, and written past the caller's allowed color
  vocabulary. A washed-out or unlit region therefore passed the color check,
  and because the result also read as a trustworthy measurement it overwrote
  the detector's class, carrying the wrong label into the review dataset that
  retrains the detector. The absence is now reported and fails closed,
  including when a product configures a threshold of zero.
- Hue is now averaged on its circle at both ends of the baseline contract.
  OpenCV hue wraps at 0/179, but the mean was taken linearly when the baseline
  was written and again when a region was scored, so red samples at 3 and 178
  averaged to ~90 -- green. A dim red part crossing the seam was classified
  Green outright, and even a bright one lost a quarter of its score to a color
  it does not resemble. Existing `color_stats.json` files keep working; a
  baseline whose samples straddle the seam needs recalibration to benefit.
- A detection whose box is degenerate or lands outside the image no longer
  takes down the frame. Cropping it blind produced an empty array, which
  OpenCV answers with an assertion failure that escaped the whole pipeline: one
  unusable box became a frame-wide ERROR, and in the async pipeline it tripped
  the stop event and halted the line. The item now fails closed on its own and
  the result reports `unmeasurable_roi` rather than claiming every ROI was
  evaluated. Bbox cropping is shared with the crop-saving path, which had the
  same gap.
- Fusion runs measured color from the wrong image. `processed_image` is the
  clean image detections were measured on, and both the color checker and the
  result sink depend on that; the fusion merge pointed it at `result_frame` --
  the YOLO overlay drawn onto the anomalib heatmap -- so color was read from
  heatmap pseudo-color plus the drawn box borders, and the saved annotation was
  drawn over an already-annotated frame. The overlay remains available as
  `result_frame` and the heatmap as `heatmap_path`.
- An activated `global` color revision now reaches the `color_qc` checker.
  `ColorCheckerService` dropped `default_threshold` on that path and
  `ColorQCEnhanced` had no way to accept it, so the revision was silently inert
  for every product on that checker while the run log still announced it as
  applied. It now applies, returns to the model baseline when a later product
  restates nothing, and rejects a malformed value without leaving partial
  state.
- A color whose baseline lacks `hsv_mean` or `lab_mean` no longer outscores one
  that has them. The absent similarity term defaulted to a perfect 1.0 and
  still collected its full 0.2-0.3 weight; absent terms are now dropped and the
  remaining weights renormalized, which is a no-op when every statistic is
  present.
- The hue membership test used for colors without a dedicated rule now survives
  the 0/179 seam, so a margin pushing a range past either end -- or a color
  calibrated around hue 0 -- no longer reads as "never matches".
- Malformed color statistics are rejected at load with the offending key named,
  instead of loading and failing later as an IndexError inside the matcher.
- The black shortcut no longer reports a confidence unrelated to the rule that
  fired, which could produce "this region is black, and black is NG" in one
  result.
- `_compute_hsv3d_hist` no longer swallows an OpenCV failure in silence. The
  pure-Python path exists for environments without cv2, not as a shock absorber
  for runtime errors: it is ~11x slower per ROI and blows the inspection
  latency budget. Anything falling back is now logged, and an empty region is
  rejected before it reaches OpenCV.
- Release activation, rollback history, and retention cleanup now use
  recoverable commit points so database, audit, and filesystem failures cannot
  leave an active blocked release or silently orphan inspection evidence.
- Pilot, preflight, and duplicate-audit evidence now fails closed on wrong
  scope, incomplete inputs, stale configuration, unsafe output paths, and
  runtime policy mismatches.
- Pytest now isolates workspace discovery from live station data and rejects
  environment changes that redirect tests into production paths.
- The acceptance window kept one full-resolution annotated preview per inferred
  sample and never released any of them, so running a batch of a few hundred
  station images accumulated several GB of QPixmap and the process died with no
  traceback. Previews are now stored at display size and capped, evicting the
  least recently viewed. A record whose preview was released says so instead of
  reporting itself as never inferred.
- An acceptance run using a selected color model recorded the deployed color
  model's hash instead of the one it actually used, because the run supplied no
  model identity and the service fell back to the identity on disk. The
  override itself was applied all along; only the recorded evidence disagreed,
  which made a working override look ignored.
- The matrix result table and the version workspace's validation table printed
  a whole-verdict count (誤殺／漏檢) next to a color-only rate (顏色誤殺率／
  顏色逃逸率), so no denominator a reader could guess turned the one into the
  other and identical, reproducible runs read as non-deterministic. Every
  metric now renders as `張數（比率）` in a single cell, the headers name which
  family they belong to and carry the exact denominator as a tooltip, and a
  count whose denominator is empty reports UNKNOWN with the reason instead of
  a zero that reads as a clean sheet.
- The matrix table showed `0` in 相較首組變動 both for the reference row itself
  and for a row that matched the reference. The reference row now says 基準組.
- Committing an acceptance run discarded every result outside that run, so
  推論目前圖片 and 推論未完成圖片 silently wiped the machine verdicts of every
  other sample while reporting a successful atomic commit. The two partial
  buttons could therefore never converge on a complete manifest: each one
  re-created the pending set it had just cleared, leaving 全部重新推論 as the
  only route to a snapshot. A committed batch now clears only results whose
  `artifact_bundle_sha256` differs from the incoming one, because the property a
  formal snapshot needs is that every result came from one identical artifact
  combination — not from one invocation. A formal snapshot accordingly accepts
  several completed runs of one bundle, and still rejects a mixed bundle, an
  untracked result, or a run that never reached COMPLETED. Starting a partial
  run that would discard another combination's results now asks first, so the
  operator decides instead of discovering it from a table that emptied itself.
- `calculate_acceptance_metrics()` raised on a confirmed sample carrying no
  OK/NG verdict, and the acceptance window called it unguarded on every render.
  One hand-edited or restored-from-old-backup row therefore threw out of a Qt
  slot each time the scope was opened, taking the summary bar and snapshot
  comparison with it. Such rows are now counted as `malformed`, excluded from
  `confirmed` and from every rate so no denominator is inflated, and shown in
  the summary bar as 真值異常. Rejection stays at the two decision points that
  can act on it: the acceptance gate and a formal snapshot.
- The fusion→yolo color-scope rule existed as five separate copies across the
  acceptance window, its inference worker, the gate, the matrix, and the matrix
  dialog, and they had drifted: the window's copy did not lowercase the type it
  returned, so a scope lookup could search a differently named scope than the
  same lookup made from the matrix dialog and report a station as having no
  stored color model. All five now call one `color_scope_model_type()`. The
  model-directory lookups in `build_model_variant()` and `load_model_identity()`
  deliberately keep their own copy: they resolve filesystem paths rather than a
  color scope, and lowercasing a directory name is not theirs to do.
- Three symlink guards could never fire, because they tested a path that had
  already been through `resolve()`, which follows the link. They read as
  protection while protecting nothing. The lock helper now checks before
  resolving, where a link is still visible, and the redundant post-resolve test
  in `artifact_ref()` is gone.
- The acceptance gate and the release builder compared bundle values that had
  been stripped and lowercased when the bundle was built against raw caller and
  report values, so a target differing only in whitespace or letter case was
  reported as a mismatched bundle. Both sides are now normalized identically.
- `export_backup_zip()` archived the manifest lock directory. Each file there
  carries one zero-information byte, and on Windows a lock held by another
  process makes it unreadable, which aborted an otherwise valid backup.
  `locks/` is now skipped alongside `backups/`.
- A matrix combination in which every sample failed was publishable. Only the
  combination-level `error` field was checked, but the inference service turns
  each per-image failure into an ERROR outcome and carries on, so the realistic
  failure shape is `error: ""` alongside `errors: 250` and four zeroed confusion
  counts. A release could therefore bind itself to evidence in which nothing was
  ever decided. Publishing now also refuses a combination carrying any per-sample
  error or no decided samples at all.
- Acceptance color discovery offered stored color models without checking they
  can be loaded, so an unusable one was selectable and only failed after every
  image of its combination had been inferred — two of them wasted 500 inferences
  in one run and produced two all-ERROR combinations. A model the stats checker
  cannot load is now reported as a `ColorVariantExclusion` naming the missing
  statistic. Validation runs the real loader rather than re-listing the keys it
  needs, because a separate schema check is free to drift from the loader and
  then a model passes validation and still fails at inference. Candidate
  *status* is deliberately not an exclusion criterion: an INCOMPLETE
  recalibration can still hold usable statistics for the colors it did finish.
- The test suite could publish into live station data. Workspace discovery walks
  *upwards* for `workspace.yaml`, so a pytest `--basetemp` inside the workspace
  resolves a test's own `tmp_path` to the real station data, and code that
  rediscovers paths from a caller-supplied root — as the release builder does
  from `models_root.parent` — writes production evidence. A stub color model
  reached the real profile store this way and was offered as an acceptance
  variant. Two guards now cover this. The suite refuses to start when
  `--basetemp` resolves inside a workspace, which is the whole mechanism and is
  an easy mistake here because this workspace keeps its scratch directories in
  `.tmp/`, inside the workspace. As a backstop for any other route, an autouse
  fixture fails the individual test that creates an entry in a live
  station-data directory, instead of leaving it to be traced weeks later from a
  manifest's recorded source path.
- Every inference outcome cleared and refilled the whole acceptance sample list,
  so the cost of watching a run grew with the square of its length: a few
  hundred station images discarded tens of thousands of rows to show the same
  list back. An outcome can only change its own row and never the order,
  because the visible order follows the manifest, so one row is now repainted in
  place. The full rebuild is still used for the one case a repaint cannot
  express — a new verdict that moves the record in or out of the active filter,
  which shifts every row after it.

- Independent model acceptance workspace with reusable human truth, immutable
  snapshots, backup export, FP/FN metrics and YOLO × color combination tests.
- Versioned inspection-component catalog and atomic inspection releases for
  YOLO, Anomalib, fusion and full Stats Color profiles, including guarded
  activation and complete-combination rollback.
- Five-color Stats Color baseline rebuilding from confirmed OK evidence with
  train/holdout separation, HSV/Lab drift review and immutable candidates.
- Conservative cross-class duplicate-box handling after color verification,
  with report-only/suppress modes, position-check fail-closed protection,
  raw/effective result traceability, GUI evidence overlays and a read-only
  historical replay audit tool.
- Role-based production documentation:
  - `docs/manuals/OPERATOR_MANUAL.md` for daily inspection, history, Excel, review,
    retraining and escalation;
  - `docs/manuals/ENGINEERING_MANUAL.md` for configuration, position gates, deployment,
    SQLite recovery, server synchronization and release acceptance.
- PIN-protected full-width engineering settings and in-window retraining
  workspace.
- Inspection-history page with database-backed filters, evidence preview,
  synchronization status and cancellable Excel export.
- Versioned SQLite inspection database, verified backup/restore tooling,
  retention dry-run and production preflight.
- Local-first company-server synchronization outbox with idempotency,
  revisions, retry leases and dead-letter administration.
- Explicit per-job position-retraining and post-gate activation controls.

- Documentation for the local `yolo_anomalib` conda environment, fusion
  inference usage, and model-level color checker overrides.

- **Path Security Validation Module** (`core/security.py`)
  - `PathValidator` class for preventing directory traversal attacks
  - Global `path_validator` instance for project-wide use
  - Support for multiple allowed root directories
  - Symlink resolution and validation
  - Comprehensive test suite (12/13 tests passing, 1 skipped on Windows)

- **Security Tests** (`tests/test_security.py`)
  - Directory traversal attack prevention tests
  - Path validation boundary tests
  - Global validator integration tests
  - Multiple allowed roots scenario tests

### Changed
- The black shortcut, the yellow shortcut, and the main scoring path in
  `StatsColorChecker` now judge the same center crop. They previously used
  three different crops, so the three decisions could legitimately disagree
  about an elongated region.
- The hue and S/V gates for red, orange, and green moved from module constants
  into `ColorDecisionTuning`, so a product can retune them through
  `color_decision_tuning` in its `config.yaml`. The values were one product's
  calibration hard-coded in the matcher, which contradicted the module's own
  documented contract that threshold changes never require a code release.
  Defaults are unchanged.
- Consolidated engineering version operations into component, candidate,
  validation and deployment views; UI metrics now use 誤殺／漏檢 terminology.
- Acceptance color crops now use the processed-image coordinate space emitted
  by inference. Earlier `stats-robust-v1` candidates are retained for audit but
  marked incompatible; new candidates use `stats-robust-v2`.
- Release timestamps are stored as ISO 8601 UTC and rendered in the station's
  local timezone instead of displaying raw UTC wall time.
- Renamed the existing model IoU control to `YOLO NMS IoU (same-class)` and
  added separately scoped duplicate IoU/geometry controls under model settings.
- Removed unused legacy GUI panel builders and the superseded ad-hoc performance
  benchmark; retained the current GUI, packaging and camera compatibility
  entrypoints.
- Replaced the historical project progress file as a calibration record with
  `docs/records/CALIBRATION_CHANGE_LOG.md`, and moved historical evidence under
  `docs/archive/`.
- Retraining is opened from `工程設定 > 模型補訓`; legacy documentation that
  pointed to the File menu has been corrected.
- Position validation runs only when explicitly selected for the current
  retraining job. Position choices are intentionally not persisted.
- Result reporting uses `inspection_records.sqlite3` as the searchable source
  and GUI-filtered Excel snapshots instead of one fixed workbook.

- Fusion inference and color override handling are documented as model-level
  pipeline behavior, including the fallback behavior when Anomalib or PyYAML is
  unavailable.

- **`core/config.py`**: Integrated path security validation
  - Added path validation in `DetectionConfig.from_yaml()`
  - Prevents loading configs from untrusted external paths
  - Graceful fallback if security module unavailable

- **`main.py`**: Enhanced CLI security
  - Added security validation for `--image` parameter
  - Prevents image loading from untrusted paths
  - Improved error messages for path validation failures

- **`requirements.txt`**: Updated to complete dependency list
  - Expanded from simple `-e .` to 342 lines of pinned dependencies
  - Generated using `pip-compile` from `pyproject.toml`
  - Ensures reproducible installations across environments

### Security
- **防止目錄遍歷攻擊** (Directory Traversal Protection)
  - Blocks paths containing `..` sequences
  - Validates paths are within allowed root directories
  - Protects configuration files, model weights, and input images

- **YAML 安全載入** (Safe YAML Loading)
  - All YAML loading uses `yaml.safe_load()`
  - Verified in: `core/config.py`, `core/services/model_manager.py`, `core/detection_system.py`
  - Prevents arbitrary code execution via malicious YAML files

- **白名單式路徑控制** (Whitelist-based Path Access)
  - Only allows access to predefined project directories
  - Default allowed roots: project root, models directory, Result directory
  - Configurable for different deployment scenarios

## [1.1.0] - 2026-08-13

### Added
- Explicit `status` field on `ColorCheckResult` / the serialized `color_check`
  payload (`evaluated`, `no_detections`), and on `count_check` /
  `position_check` results (`expected_items_lookup_failed`,
  `position_config_lookup_failed`), so downstream consumers can tell a check
  that actually ran from one that could not, instead of reading an unrelated
  FAIL as a pass.
- Shared color-failure classification (`classify_color_check_failure` in
  `core/services/results/customer_message.py`) distinguishing a color
  **mismatch** (wrong color detected) from **low confidence** (right color,
  score below its own threshold), with matching wording reused across the
  customer-facing message, the GUI detail panel and the ASCII image overlay,
  plus new `color_mismatch` / `color_low_confidence` translation strings
  (EN/ZH).
- `ColorQCEnhanced.apply_runtime_configuration()` /
  `reset_runtime_configuration()` and the equivalent `StatsColorChecker`
  methods, which replace (rather than merge) the active thresholds/rules in a
  single validated, all-or-nothing call.
- Expanded pipeline and color-checker test coverage for the new fail-closed
  paths (`tests/test_pipeline_steps.py`,
  `tests/test_color_checker_service.py`).

### Changed
- Color check on a frame with zero detections now fails closed
  (`status=no_detections`) instead of estimating a verdict from the full-frame
  background.
- `ColorCheckerService` reapplies the complete runtime configuration (default
  threshold, overrides, rules) on every invocation, including calls that
  supply none, so a checker instance cached across products can no longer
  carry a previous product's tuning into the next inspection.
- Color-check failure text throughout the GUI and customer message now states
  whether the color was mismatched or just under-confident, instead of
  naming only the detected class.

### Fixed
- Count check and position check no longer silently pass when their
  configuration lookup fails (unreadable expected-items or position-enable
  config); both now fail closed and record why, and `finalize_status` carries
  that verdict through instead of re-deriving a false PASS from the absence of
  a signal.
- Color check candidate lookup failures now respect `color_fail_closed`
  instead of silently falling back to an unrestricted color vocabulary.
- `apply_threshold_overrides`, `apply_color_rules_overrides` and
  `set_default_threshold` now raise on invalid values instead of silently
  discarding them.

## [0.1.0] - 2026-01-06

### 初始版本功能 (Initial Release Features)

#### Core Functionality
- **YOLO11 物件偵測** (Object Detection)
  - Integration with Ultralytics YOLO11
  - Support for custom trained models
  - Confidence and IoU threshold configuration
  - Model caching with LRU eviction (3 models default)
  - GPU warmup for reduced first-frame latency

- **Anomalib 異常檢測** (Anomaly Detection)
  - PatchCore, PaDiM, STFPM, DRAEM support
  - Pixel-level and image-level anomaly detection
  - Heatmap generation and visualization
  - Configurable anomaly thresholds

- **位置驗證** (Position Validation)
  - Expected position checking for detected objects
  - Absolute (pixel) and relative (percentage) tolerance
  - Support for multiple products and areas
  - Auto-generation of position configs from training data

- **顏色檢測** (Color Detection)
  - Statistical color checking for LED components
  - HSV-based color classification
  - Support for multiple color targets per product
  - Color sequence validation for cable products

#### Interfaces
- **命令列介面** (CLI)
  - Interactive mode for product/area selection
'  - Single-shot inference mode with arguments
  - Batch processing support
  - Configurable output formats (Excel, JSON, images)

- **圖形介面** (PyQt5 GUI)
  - Live camera preview
  - Model hot-swapping
  - Real-time inference visualization
  - Results logging and export
  - Multi-threading for responsive UI

#### Industrial Camera Support
- **海康威視 MVS SDK 整合**
  - Automatic device enumeration
  - Exposure and gain control
  - Image acquisition with timeout
  - ROI (Region of Interest) support

#### Pipeline Architecture
- **Modular Step System**
  - Registry-based step loading
  - Configurable pipeline per model
  - Built-in steps: color_check, count_check, sequence_check, position_validation
  - Easy custom step development

#### Quality & Development
- **測試套件** (Test Suite)
  - 52 comprehensive tests
  - Unit, integration, and E2E test coverage
  - Mock camera support for CI environments
  - Performance and robustness tests

- **程式碼品質工具** (Code Quality Tools)
  - Ruff for linting
  - MyPy for type checking  
  - Pytest with coverage reporting
  - Pre-commit hooks configuration

#### Documentation
- **README.md**: 371-line comprehensive project documentation
- **docs/architecture/TECH_GUIDE.md**: 1153-line deep-dive technical guide (JR→SR level)
- **docs/architecture/MODULE_ARCHITECTURE.md**: Architecture diagrams and design patterns
- **config.example.yaml**: Full configuration template with comments

#### Configuration & Flexibility
- **多產品/多站別支援** (Multi-product/Multi-area Support)
  - Hierarchical model organization: `models/{product}/{area}/{type}/`
  - Per-model configuration files
  - Dynamic model loading based on product/area selection

- **靈活的配置系統** (Flexible Configuration)
  - YAML-based global and model-specific configs
  - Environment variable support via `python-dotenv`
  - Pydantic schema validation
  - Hot-reload capabilities

#### Results Management
- **Excel 報表輸出** (Excel Reports)
  - Detailed detection results with timestamps
  - Pass/Fail status per item
  - Color check results
  - Position validation results
  - Anomaly scores and heatmaps

- **影像標註與保存** (Image Annotation)
  - Bounding box visualization
  - Confidence score labels
  - Color-coded pass/fail indicators
  - Original and processed image pairs

---

## Migration Guide

### From Pre-0.1.0 Development Versions

If you're upgrading from an early development version:

1. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Review Configuration**:
   - Compare your `config.yaml` with `config.example.yaml`
   - Add any new required fields (especially security-related)

3. **Test Path Validation**:
   - Ensure your model paths, output directories are within project root
   - Or configure custom allowed roots if needed

4. **Run Tests**:
   ```bash
   pytest -v
   ```

---

## Roadmap

### Unscheduled
- [ ] TensorRT INT8 quantization support
- [ ] Docker deployment guide and Dockerfile
- [ ] REST API service mode
- [ ] Calibration wizard for new cameras
- [ ] Performance profiling dashboard

### Under Consideration
- [ ] Support for additional anomaly models (FastFlow, Reverse Distillation)
- [ ] Multi-camera orchestration
- [ ] Cloud model repository integration
- [ ] Automated retraining pipeline
- [ ] Web-based configuration interface

---

## Contributors

- **DioWang** - Initial development and architecture
- **AI Assistant** - Documentation, testing, and security enhancements

---

## License

Proprietary License - Unauthorized distribution or use is prohibited.

---

## Acknowledgments

This project uses the following open-source packages:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Anomalib](https://github.com/openvinotoolkit/anomalib)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
