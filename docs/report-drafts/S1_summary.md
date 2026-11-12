# Spike S1: live capture

**Aim.** Spike S1 tested whether FastRTC inside a Gradio page can stream the webcam and microphone
to Python handlers in real time, with on-screen meters updating, on the project laptop.

**Setup.** The test machine was an Intel Core i7-1255U laptop with 15.7 GB RAM, running
Windows 11 on charger, with its built-in camera and microphone. The page ran in Chromium,
driven by Playwright.
- Two transports were compared: FastRTC (WebRTC, audio-video) and a fallback using Gradio's own
  streaming audio and webcam components.
- Landmarks came from MediaPipe Pose (lite) and Face Landmarker, run on a separate worker thread.
- Only counters and timestamps were logged. No audio or video was stored.

**Pass conditions.** In at least 95 % of seconds after a 5 s warm-up, landmark processing reaches
at least 8 frames per second. The 95th percentile of meter delay, from server to screen, stays
under 300 ms.

**Results.** Every run passed (table). FastRTC showed no growing delay over 3 minutes and lost no
audio. The fallback is capped at 10 fps and lost about 5 % of audio time.

**Chosen frame interval.** Landmarks run on every 2nd FastRTC frame, about 15 fps. With every 3rd
frame, the slowest second was 9 fps, one frame above the threshold. Every 2nd frame raised the
minimum to 14 fps. It cost a small amount of CPU and about 5 ms of extra delay.

**Limitations.**
- One machine was tested.
- In one run, a face was found in only 80.5 % of frames; this is unexplained.
- The pixelation check measured resolution only (640×360 throughout), so blockiness from a low
  encoder bitrate was not measured.
- Low light, where webcams often slow down, was not tested.
- The meter delay excludes the time from camera to server.

| Transport | Landmarks | Frame interval | Duration | Landmark fps (median / min) | Meter delay p95 | Face found | Result |
|---|---|---|---|---|---|---|---|
| FastRTC | stand-in | every frame | 180 s | 30 / 29 | 69.3 ms | n/a | Pass |
| Fallback | stand-in | every frame | 180 s | 10 / 10 | 91.7 ms | n/a | Pass |
| FastRTC | pose + face | every 3rd | 180 s | 10 / 9 | 77.9 ms | 97.7 % | Pass |
| FastRTC | pose + face | every 2nd | 60 s | 15 / 14 | 77.6 ms | 100 % | Pass |
