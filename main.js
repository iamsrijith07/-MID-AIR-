/**
 * MID AIR — Gesture Drawing App
 * main.js
 *
 * State Machine:
 *  DRAW  → index finger up only       → draw neon line
 *  ERASE → open palm (all 5 fingers)  → erase in radius
 *  GRAB  → pinch (thumb ↔ index)      → drag strokes
 *  IDLE  → any other pose             → nothing
 */

// ─── Configuration ────────────────────────────────────────────────────────────
const CONFIG = {
  MEDIAPIPE_WASM: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
  MEDIAPIPE_MODEL: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  PINCH_THRESHOLD:    0.07,   // normalised distance for pinch detection
  ERASE_RADIUS:       40,     // pixels
  GRAB_RADIUS:        60,     // pixels — how close a stroke must be to be grabbed
  SMOOTH:             0.35,   // lerp factor for finger smoothing (lower = smoother)
  MIN_DRAW_DIST:      2,      // minimum pixel movement to add a new point
};

// ─── DOM References ───────────────────────────────────────────────────────────
const video       = document.getElementById('webcam');
const canvas      = document.getElementById('drawCanvas');
const ctx         = canvas.getContext('2d');
const cursor      = document.getElementById('fingerCursor');
const loadScreen  = document.getElementById('loadingScreen');
const loaderBar   = document.getElementById('loaderBar');
const loaderMsg   = document.getElementById('loaderMsg');
const statusDot   = document.getElementById('statusDot');
const statusText  = document.getElementById('statusText');
const currentStateEl = document.getElementById('currentState');

// Gesture chips
const chips = {
  draw:  document.getElementById('chip-draw'),
  erase: document.getElementById('chip-erase'),
  grab:  document.getElementById('chip-grab'),
  idle:  document.getElementById('chip-idle'),
};

// ─── App State ────────────────────────────────────────────────────────────────
let handLandmarker = null;
let animFrameId    = null;

// Drawing data: array of stroke objects { color, size, glow, points:[[x,y]] }
let strokes = [];
let currentStroke = null;

// Gesture state machine
let gestureState = 'IDLE'; // 'DRAW' | 'ERASE' | 'GRAB' | 'IDLE'

// Smoothed finger tip positions (canvas coords)
let smoothX = 0, smoothY = 0;
let prevX   = null, prevY = null;

// Grab state
let grabbedStrokes   = [];  // indices of grabbed strokes
let grabOffsets      = [];  // per-stroke {dx, dy} offsets at grab start
let grabStartX       = 0, grabStartY = 0;

// UI state
let activeColor   = '#00FFFF';
let brushSize     = 6;
let glowEnabled   = true;

// ─── Canvas Resize ────────────────────────────────────────────────────────────
function resizeCanvas() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  redrawAll();
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ─── UI Controls ──────────────────────────────────────────────────────────────
document.querySelectorAll('.color-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeColor = btn.dataset.color;
    cursor.style.background      = activeColor;
    cursor.style.boxShadow       = `0 0 8px ${activeColor}, 0 0 24px ${activeColor}`;
  });
});

document.querySelectorAll('.brush-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.brush-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    brushSize = parseInt(btn.dataset.size, 10);
  });
});

document.getElementById('glowToggle').addEventListener('change', e => {
  glowEnabled = e.target.checked;
  redrawAll();
});

document.getElementById('clearBtn').addEventListener('click', () => {
  strokes = [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

document.getElementById('screenshotBtn').addEventListener('click', () => {
  // Merge canvas on top of a black background and save
  const offscreen = document.createElement('canvas');
  offscreen.width  = canvas.width;
  offscreen.height = canvas.height;
  const offCtx = offscreen.getContext('2d');
  offCtx.fillStyle = '#080C12';
  offCtx.fillRect(0, 0, offscreen.width, offscreen.height);
  offCtx.drawImage(canvas, 0, 0);
  const link = document.createElement('a');
  link.download = `midair_${Date.now()}.png`;
  link.href = offscreen.toDataURL('image/png');
  link.click();
});

// ─── Status helpers ───────────────────────────────────────────────────────────
function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}
function setLoader(pct, msg) {
  loaderBar.style.width = `${pct}%`;
  if (msg) loaderMsg.textContent = msg;
}

// ─── MediaPipe Init ───────────────────────────────────────────────────────────
async function initMediaPipe() {
  setStatus('loading', 'Loading model…');
  setLoader(10, 'Importing MediaPipe…');

  try {
    // Dynamic import from CDN
    const vision = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs'
    );
    const { HandLandmarker, FilesetResolver } = vision;

    setLoader(35, 'Fetching WASM runtime…');
    const filesetResolver = await FilesetResolver.forVisionTasks(CONFIG.MEDIAPIPE_WASM);

    setLoader(60, 'Loading hand model…');
    handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: CONFIG.MEDIAPIPE_MODEL,
        delegate: 'GPU',
      },
      runningMode:          'VIDEO',
      numHands:             1,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });

    setLoader(80, 'Starting webcam…');
    await startWebcam();
    setLoader(100, 'Ready!');

    setTimeout(() => {
      loadScreen.style.transition = 'opacity 0.6s';
      loadScreen.style.opacity = '0';
      setTimeout(() => { loadScreen.style.display = 'none'; }, 650);
    }, 400);

    setStatus('ready', 'Tracking active');
    startDetectionLoop();

  } catch (err) {
    console.error('MediaPipe init error:', err);
    setStatus('error', 'Error — see console');
    setLoader(100, `Error: ${err.message}`);
    loaderMsg.style.color = '#FF4444';
  }
}

// ─── Webcam ───────────────────────────────────────────────────────────────────
async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  return new Promise(resolve => { video.onloadedmetadata = resolve; });
}

// ─── Detection Loop ───────────────────────────────────────────────────────────
let lastVideoTime = -1;

function startDetectionLoop() {
  function detect() {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const results = handLandmarker.detectForVideo(video, performance.now());
      processResults(results);
    }
    animFrameId = requestAnimationFrame(detect);
  }
  detect();
}

// ─── Landmark → Canvas Coords ─────────────────────────────────────────────────
function lmToCanvas(lm) {
  // Video is mirrored via CSS, so we mirror X here too
  return {
    x: (1 - lm.x) * canvas.width,
    y: lm.y * canvas.height,
  };
}

// ─── Gesture Classification ───────────────────────────────────────────────────
function classifyGesture(lms) {
  // Finger tip and pip (knuckle) landmark indices
  const tips = [8, 12, 16, 20];  // index, middle, ring, pinky
  const pips = [6, 10, 14, 18];

  const thumbTip   = lms[4];
  const indexTip   = lms[8];

  // Pinch: thumb tip vs index tip distance
  const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
  if (pinchDist < CONFIG.PINCH_THRESHOLD) return 'GRAB';

  // Count extended fingers (tip y < pip y in image space → finger pointing up)
  const extendedFingers = tips.filter((tip, i) => lms[tip].y < lms[pips[i]].y);

  // Thumb extended: tip x further from wrist than the MCP (landmark 2)
  const thumbExtended = lms[4].x < lms[2].x; // mirrored (right hand)

  const totalExtended = extendedFingers.length + (thumbExtended ? 1 : 0);

  // All 5 → erase
  if (totalExtended >= 4 && extendedFingers.length >= 4) return 'ERASE';

  // Only index extended → draw
  const indexUp  = lms[8].y < lms[6].y;
  const middleUp = lms[12].y < lms[10].y;
  const ringUp   = lms[16].y < lms[14].y;
  const pinkyUp  = lms[20].y < lms[18].y;
  if (indexUp && !middleUp && !ringUp && !pinkyUp) return 'DRAW';

  return 'IDLE';
}

// ─── Result Processor ─────────────────────────────────────────────────────────
function processResults(results) {
  if (!results.landmarks || results.landmarks.length === 0) {
    // No hand detected
    if (gestureState === 'DRAW' && currentStroke) finaliseStroke();
    if (gestureState === 'GRAB') releaseGrab();
    setGestureState('IDLE');
    cursor.style.display = 'none';
    prevX = null; prevY = null;
    return;
  }

  const lms = results.landmarks[0];
  const indexTip = lmToCanvas(lms[8]);

  // Smooth cursor
  if (prevX === null) { smoothX = indexTip.x; smoothY = indexTip.y; }
  smoothX += (indexTip.x - smoothX) * CONFIG.SMOOTH;
  smoothY += (indexTip.y - smoothY) * CONFIG.SMOOTH;

  const gesture = classifyGesture(lms);

  // Transition handling
  if (gesture !== gestureState) {
    if (gestureState === 'DRAW' && currentStroke) finaliseStroke();
    if (gestureState === 'GRAB') releaseGrab();
    setGestureState(gesture);
    prevX = null; prevY = null;
  }

  // Show cursor
  cursor.style.display = 'block';
  cursor.style.left = `${smoothX}px`;
  cursor.style.top  = `${smoothY}px`;

  // Execute state actions
  if (gesture === 'DRAW')  handleDraw(smoothX, smoothY);
  if (gesture === 'ERASE') handleErase(smoothX, smoothY);
  if (gesture === 'GRAB')  handleGrab(smoothX, smoothY, lms);

  prevX = smoothX; prevY = smoothY;
}

// ─── State Machine UI ─────────────────────────────────────────────────────────
function setGestureState(state) {
  gestureState = state;
  currentStateEl.textContent = state;

  // Style the state label
  const colors = { DRAW: '#00FFFF', ERASE: '#FF4444', GRAB: '#FF00FF', IDLE: 'rgba(180,210,255,0.45)' };
  currentStateEl.style.color      = colors[state];
  currentStateEl.style.textShadow = state !== 'IDLE' ? `0 0 8px ${colors[state]}, 0 0 20px ${colors[state]}` : 'none';

  // Cursor styling
  cursor.className = '';
  if (state === 'ERASE') cursor.classList.add('erase-mode');
  if (state === 'GRAB')  cursor.classList.add('grab-mode');

  // Chips
  Object.values(chips).forEach(c => c.className = 'gesture-chip');
  if (state === 'DRAW')  chips.draw.classList.add('active-draw');
  if (state === 'ERASE') chips.erase.classList.add('active-erase');
  if (state === 'GRAB')  chips.grab.classList.add('active-grab');
}

// ─── DRAW ─────────────────────────────────────────────────────────────────────
function handleDraw(x, y) {
  if (!currentStroke) {
    currentStroke = {
      color: activeColor,
      size:  brushSize,
      glow:  glowEnabled,
      points: [[x, y]],
    };
    strokes.push(currentStroke);
  } else {
    const last = currentStroke.points[currentStroke.points.length - 1];
    const dist  = Math.hypot(x - last[0], y - last[1]);
    if (dist < CONFIG.MIN_DRAW_DIST) return;
    currentStroke.points.push([x, y]);

    // Incremental draw (only draw the newest segment for performance)
    drawSegment(ctx, currentStroke, currentStroke.points.length - 2);
  }
}

function finaliseStroke() {
  currentStroke = null;
}

function drawSegment(targetCtx, stroke, fromIdx) {
  const pts = stroke.points;
  if (pts.length < 2 || fromIdx >= pts.length - 1) return;

  targetCtx.save();
  targetCtx.lineCap   = 'round';
  targetCtx.lineJoin  = 'round';
  targetCtx.lineWidth = stroke.size;
  targetCtx.strokeStyle = stroke.color;

  if (stroke.glow) {
    targetCtx.shadowColor = stroke.color;
    targetCtx.shadowBlur  = stroke.size * 3;
  }

  targetCtx.beginPath();
  targetCtx.moveTo(pts[fromIdx][0], pts[fromIdx][1]);

  for (let i = fromIdx + 1; i < pts.length; i++) {
    const mx = (pts[i - 1][0] + pts[i][0]) / 2;
    const my = (pts[i - 1][1] + pts[i][1]) / 2;
    targetCtx.quadraticCurveTo(pts[i - 1][0], pts[i - 1][1], mx, my);
  }
  const last = pts[pts.length - 1];
  targetCtx.lineTo(last[0], last[1]);
  targetCtx.stroke();
  targetCtx.restore();
}

// ─── Full Redraw ──────────────────────────────────────────────────────────────
function redrawAll() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  strokes.forEach(stroke => {
    if (stroke.points.length < 2) return;
    ctx.save();
    ctx.lineCap   = 'round';
    ctx.lineJoin  = 'round';
    ctx.lineWidth  = stroke.size;
    ctx.strokeStyle = stroke.color;
    if (stroke.glow) {
      ctx.shadowColor = stroke.color;
      ctx.shadowBlur  = stroke.size * 3;
    }
    ctx.beginPath();
    ctx.moveTo(stroke.points[0][0], stroke.points[0][1]);
    for (let i = 1; i < stroke.points.length; i++) {
      const mx = (stroke.points[i - 1][0] + stroke.points[i][0]) / 2;
      const my = (stroke.points[i - 1][1] + stroke.points[i][1]) / 2;
      ctx.quadraticCurveTo(stroke.points[i - 1][0], stroke.points[i - 1][1], mx, my);
    }
    const last = stroke.points[stroke.points.length - 1];
    ctx.lineTo(last[0], last[1]);
    ctx.stroke();
    ctx.restore();
  });
}

// ─── ERASE ────────────────────────────────────────────────────────────────────
function handleErase(x, y) {
  const r = CONFIG.ERASE_RADIUS;
  let changed = false;

  strokes = strokes.filter(stroke => {
    // Remove any stroke that has a point within the erase radius
    const hit = stroke.points.some(p => Math.hypot(p[0] - x, p[1] - y) < r);
    if (hit) { changed = true; }
    return !hit;
  });

  if (changed) redrawAll();
}

// ─── GRAB ─────────────────────────────────────────────────────────────────────
function handleGrab(x, y, lms) {
  if (grabbedStrokes.length === 0) {
    // Find all strokes within grab radius
    strokes.forEach((stroke, idx) => {
      const close = stroke.points.some(p => Math.hypot(p[0] - x, p[1] - y) < CONFIG.GRAB_RADIUS);
      if (close) {
        grabbedStrokes.push(idx);
        // Compute offset from grab point to each stroke point centroid
        const cx = stroke.points.reduce((s, p) => s + p[0], 0) / stroke.points.length;
        const cy = stroke.points.reduce((s, p) => s + p[1], 0) / stroke.points.length;
        grabOffsets.push({ dx: cx - x, dy: cy - y });
      }
    });
    grabStartX = x; grabStartY = y;
    return;
  }

  // Move grabbed strokes
  const dX = x - grabStartX;
  const dY = y - grabStartY;
  grabStartX = x; grabStartY = y;

  grabbedStrokes.forEach(idx => {
    strokes[idx].points = strokes[idx].points.map(p => [p[0] + dX, p[1] + dY]);
  });
  redrawAll();
}

function releaseGrab() {
  grabbedStrokes = [];
  grabOffsets    = [];
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
(async () => {
  await initMediaPipe();
})();
