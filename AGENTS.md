# MapKV Agent Behavior

These rules apply to every MapKV experiment and report in this repository.

## HTML report defaults

1. Every HTML report must include a synchronized **complete revisit comparison**.
   It must cover the first visit (B1), leaving the region, the return trajectory,
   and the revisit (B2). A short B2/re-entry clip may be added, but must not
   replace the complete revisit video.
2. The newest method and the current experimental focus must be labeled in
   Chinese near the top of the report and in video captions. Keep technical
   English identifiers in parentheses when useful for code/artifact lookup.
3. Report videos stay as relative external files; never base64-embed video.
   Provide synchronized Play/Pause/Reset controls.

## Surfel visualization defaults

1. Do not use chunk-ID center/disk plots as the primary surfel visualization.
2. The primary visualization must use RGB sampled from the actual generated
   historical observations. Never use generated or invented colors to make the
   geometry look plausible.
3. By default show both:
   - an RGB world-space overview; and
   - an RGB target-camera z-buffer render, alongside the target frame and
     projected support mask.
4. Chunk-ID, depth, confidence, and oriented-disk plots remain secondary audit
   views.
5. Until the user selects a preferred RGB style, generate a compact option page
   with multiple real-data renderings and record the choice in
   `mapkv/report_preferences.yaml`.
