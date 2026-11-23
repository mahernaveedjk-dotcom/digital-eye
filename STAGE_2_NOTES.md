# Digital Eye — Stage 2 notes

## Included in this stage

- Added `camera_config.py`, a validated, atomic JSON configuration store for four camera slots.
- Added persistent custom names, sources, enable state, and per-camera capture defaults (width, height, FPS, confidence).
- Updated additional camera slots 2–4 to use saved names and sources.
- Added automatic restoration of previously enabled additional cameras when Camera ON is enabled.
- Added reconnect handling and safer capture cleanup for additional live streams.
- Disconnecting a camera disables its saved auto-reconnect setting.
- Added tests for configuration round-tripping, independent slots, invalid slot numbers, bounds validation, and numeric USB source parsing.

## Important current limitations

- Camera 1 still uses the existing `CameraReader` pipeline and is not yet managed through the new camera configuration UI.
- Cameras 2–4 currently have independent live capture and reconnect only. Their AI detection, event recording, zone processing, and alert pipeline have not yet been unified with Camera 1.
- The existing dashboard still displays the primary camera separately from the additional-camera grid; the final unified automatic 2×2 dashboard is not implemented yet.
- Per-camera width/height/FPS/confidence values are stored and validated, but the current sidebar does not yet expose controls for editing them.
- Actual RTSP/USB camera testing and full Streamlit startup testing must be done on the target Windows computer.

## Run checks

```powershell
python -m compileall -q .
python -m unittest discover -s tests -v
```

## Run the application

```powershell
python -m pip install -r requirements.txt
streamlit run digital_eye.py
```

`camera_config.json` is intentionally ignored by Git because RTSP URLs can contain credentials. Do not share this file publicly.
