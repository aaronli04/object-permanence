# Occlusion_ball_removed / track_5

![track summary](summary.jpg)

## Summary
- class: kite (33)
- frames: 0-235 (236 frame span)
- hits: 22
- valid track: yes
- had recovery: yes
- relinked from: 6, 11
- fragment track ids: 5, 6, 11
- avg visual similarity: 0.9866
- total misses: 19
- max miss streak: 7

## Label Mix
- kite (33): 8 detections, confidence sum 3.704
- bird (14): 12 detections, confidence sum 3.587
- refrigerator (72): 2 detections, confidence sum 0.557

## Relink Edges
- 5 -> 6 via spatial (score 0.8929)
- 6 -> 11 via dino (score 0.8652)

## Event Timeline
- frame 0: created
- frame 5: activated
- frame 5: closed
- frame 10: lost
- frame 45: created
- frame 50: activated
- frame 85: closed
- frame 90: lost
- frame 160: created
- frame 165: lost
- frame 170: recovered
- frame 180: lost
- frame 185: recovered
- frame 215: lost
- frame 230: recovered
- frame 235: closed

## Key Frames
- start: frame 0, fragment 5, [image](frames/start_f0000.jpg)
- middle: frame 160, fragment 11, [image](frames/middle_f0160.jpg)
- end: frame 235, fragment 11, [image](frames/end_f0235.jpg)

[track metadata](track.json)
