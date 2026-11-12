# Spike S1: live capture

**Aim.** Spike S1 tested whether FastRTC inside a Gradio page can stream the webcam and
microphone to Python handlers in real time, with on-screen meters updating, on the project laptop.

**Setup.** The test machine was an Intel Core i7-1255U laptop running Windows 11, using its
built-in camera and microphone. Playwright drove the page in Chromium.
- Two transports were compared: FastRTC (WebRTC, audio-video) and a fallback using Gradio's own
  streaming audio and webcam components.
- Landmarks came from MediaPipe Pose (lite) and Face Landmarker, run on a separate worker thread.
- Only counters and timestamps were logged. No audio or video was stored.

**Pass conditions.** In at least 95 % of seconds after a 5 s warm-up, landmark processing reaches
at least 8 frames per second. The 95th percentile of meter delay, from server to screen, stays
under 300 ms.

**Results.** Both transports passed. FastRTC lost no audio. The fallback is capped at 10 fps and
lost about 5 % of audio time.

**Frame interval.** Landmarks were first run on every 2nd camera frame, about 15 fps, which
passed in room light. In dim light the camera itself slowed to 15 fps, so landmarks fell to
7–8 fps and the run failed. Frames are now sampled by time, targeting 15 fps: every 2nd frame
of a 30 fps camera, and every frame of a slower one. The dim-light rerun then passed, at about
the same CPU cost as room light.

**Limitations.**
- One machine was tested.
- In one run, a face was found in only 80.5 % of frames; this is unexplained.
- Blockiness from a low encoder bitrate was not measured, only resolution (640×360 throughout).
- Running on battery made landmarks markedly slower, which the live app must detect and report.
- The meter delay excludes the time from camera to server.

| Transport | Landmarks | Light | Sampling | Duration | Landmark fps (median / min) | Meter delay p95 | Face found | Result |
|---|---|---|---|---|---|---|---|---|
| FastRTC | stand-in | room | every frame | 180 s | 30 / 29 | 69.3 ms | n/a | Pass |
| Fallback | stand-in | room | every frame | 180 s | 10 / 10 | 91.7 ms | n/a | Pass |
| FastRTC | pose + face | room | every 3rd frame | 180 s | 10 / 9 | 77.9 ms | 97.7 % | Pass |
| FastRTC | pose + face | room | every 2nd frame | 180 s | 15 / 14 | 84.3 ms | 99.8 % | Pass |
| FastRTC | pose + face | dim | every 2nd frame | 60 s | 7 / 7 | 84.6 ms | 99.5 % | Fail |
| FastRTC | pose + face | dim | time-based, 15 fps | 60 s | 14 / 12 | 74.1 ms | 98.4 % | Pass |
