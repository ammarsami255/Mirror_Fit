let videoEl, canvas, ctx, detector;
let drawSkeleton = true;
let cmPerPx = null;
let heightSamples = [];

const kpts = {
  nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6, leftHip: 11, rightHip: 12,
  leftKnee: 13, rightKnee: 14, leftAnkle: 15, rightAnkle: 16
};

async function setup() {
  videoEl = document.getElementById('video');
  canvas = document.getElementById('overlay');
  ctx = canvas.getContext('2d');

  document.getElementById('startBtn').onclick = startCamera;
  document.getElementById('toggleSkel').onclick = () => drawSkeleton = !drawSkeleton;
  document.getElementById('snapshotBtn').onclick = snapshot;
  document.getElementById('autoCalib').onclick = autoCalibrate;
  document.getElementById('resetCalib').onclick = () => { cmPerPx = null; document.getElementById('cmPerPx').value = ''; };
  document.getElementById('cmPerPx').oninput = (e) => { cmPerPx = parseFloat(e.target.value) || null; };

  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('TensorFlow.js backend initialized:', tf.getBackend());

    if (typeof poseDetection === 'undefined') {
      console.error('poseDetection is not defined.');
      alert('تعذر تحميل مكتبة الكشف عن الوضعيات.');
      return;
    }

    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER }
    );
    console.log('Pose detector loaded successfully');
  } catch (err) {
    console.error('Error during setup:', err);
    alert('حدث خطأ أثناء تهيئة النموذج: ' + err.message);
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    videoEl.srcObject = stream;
    await videoEl.play();
    resizeCanvas();
    requestAnimationFrame(loop);
  } catch (err) {
    console.error('Error accessing camera:', err);
    alert('تعذر الوصول إلى الكاميرا: ' + err.message);
  }
}

function resizeCanvas() {
  const rect = videoEl.getBoundingClientRect();
  canvas.width = videoEl.videoWidth || rect.width;
  canvas.height = videoEl.videoHeight || rect.height;
}

function dist(a, b) { const dx = a.x - b.x, dy = a.y - b.y; return Math.hypot(dx, dy); }
function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }

function pxToCm(px) { return cmPerPx ? (px * cmPerPx) : null; }

function drawGuideBox() {
  ctx.save();
  ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
  ctx.lineWidth = 2;
  ctx.strokeRect(canvas.width * 0.2, canvas.height * 0.2, canvas.width * 0.6, canvas.height * 0.6);
  ctx.restore();
}

function drawKeypointsAndEdges(keypoints) {
  for (const kp of keypoints) {
    if (kp.score < 0.3) continue;
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  const pairs = [
    [kpts.leftShoulder, kpts.rightShoulder],
    [kpts.leftHip, kpts.rightHip],
    [kpts.leftShoulder, kpts.leftHip],
    [kpts.rightShoulder, kpts.rightHip],
    [kpts.leftHip, kpts.leftKnee],
    [kpts.rightHip, kpts.rightKnee],
    [kpts.leftKnee, kpts.leftAnkle],
    [kpts.rightKnee, kpts.rightAnkle],
    [kpts.nose, kpts.leftShoulder],
    [kpts.nose, kpts.rightShoulder]
  ];
  for (const [i, j] of pairs) {
    const a = keypoints[i], b = keypoints[j];
    if (a?.score > 0.3 && b?.score > 0.3) {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }
}

function postureScore(nose, midShoulders, midHips) {
  const vxNeck = nose.x - midShoulders.x;
  const vyNeck = midShoulders.y - nose.y;
  const neckAngle = Math.atan2(Math.abs(vxNeck), Math.max(1, vyNeck)) * 180 / Math.PI;
  const neckScore = Math.round(100 * (1 - Math.min(30, Math.max(0, neckAngle)) / 30));

  let backScore = 100;
  if (midHips) {
    const vxBack = midShoulders.x - midHips.x;
    const vyBack = midHips.y - midShoulders.y;
    const backAngle = Math.atan2(Math.abs(vxBack), Math.max(1, vyBack)) * 180 / Math.PI;
    backScore = Math.round(100 * (1 - Math.min(15, Math.max(0, backAngle)) / 15));
  }

  const score = Math.round((neckScore + backScore) / 2);
  return { score, neckAngle: neckAngle.toFixed(1) };
}

async function loop() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGuideBox();

  if (!detector) {
    console.warn('Detector is not initialized yet.');
    requestAnimationFrame(loop);
    return;
  }

  let poses = [];
  try {
    poses = await detector.estimatePoses(videoEl, { flipHorizontal: true });
  } catch (e) {
    console.error('Error estimating poses:', e);
  }

  if (poses[0]) {
    const kp = poses[0].keypoints;
    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(124,197,255,0.95)';
    ctx.fillStyle = 'rgba(124,197,255,0.95)';

    if (drawSkeleton) drawKeypointsAndEdges(kp);

    const ls = kp[kpts.leftShoulder], rs = kp[kpts.rightShoulder];
    const la = kp[kpts.leftAnkle], ra = kp[kpts.rightAnkle];
    const lh = kp[kpts.leftHip], rh = kp[kpts.rightHip];
    const ns = kp[kpts.nose];
    const ms = (ls?.score > 0.3 && rs?.score > 0.3) ? mid(ls, rs) : null;
    const mh = (lh?.score > 0.3 && rh?.score > 0.3) ? mid(lh, rh) : null;

    let shoulderPx = null, heightPx = null;

    if (ls?.score > 0.3 && rs?.score > 0.3) {
      shoulderPx = dist(ls, rs);
      ctx.beginPath();
      ctx.moveTo(ls.x, ls.y); ctx.lineTo(rs.x, rs.y); ctx.stroke();
    }

    if (ns?.score > 0.3 && la?.score > 0.3 && ra?.score > 0.3) {
      const midAnk = mid(la, ra);
      const currentHeightPx = dist(ns, midAnk) * 1.15;
      heightSamples.push(currentHeightPx);
      if (heightSamples.length > 10) heightSamples.shift();
      heightPx = heightSamples.reduce((a, b) => a + b, 0) / heightSamples.length;
      ctx.beginPath();
      ctx.moveTo(ns.x, ns.y); ctx.lineTo(midAnk.x, midAnk.y); ctx.setLineDash([8, 6]); ctx.stroke();
      ctx.setLineDash([]);
    }

    if (ns?.score > 0.3 && ms) {
      const p = postureScore(ns, ms, mh);
      document.getElementById('postureScore').textContent = `${p.score}/100`;
      const badge = document.getElementById('postureScore').parentElement;
      badge.style.filter = (p.score >= 80) ? 'drop-shadow(0 0 10px rgba(58,210,159,0.35))' :
                           (p.score >= 50) ? 'drop-shadow(0 0 10px rgba(255,204,102,0.25))' :
                           'drop-shadow(0 0 10px rgba(255,107,107,0.25))';
    } else {
      document.getElementById('postureScore').textContent = '—/100';
    }

    const swPxEl = document.getElementById('shoulderWidthPx');
    const swCmEl = document.getElementById('shoulderWidthCm');
    const hPxEl = document.getElementById('heightPx');
    const hCmEl = document.getElementById('heightCm');

    if (shoulderPx) {
      swPxEl.textContent = `${shoulderPx.toFixed(1)} px`;
      swCmEl.textContent = cmPerPx ? `${(pxToCm(shoulderPx)).toFixed(1)} سم` : '— سم';
    } else {
      swPxEl.textContent = '— px'; swCmEl.textContent = '— سم';
    }

    if (heightPx) {
      hPxEl.textContent = `${heightPx.toFixed(1)} px`;
      hCmEl.textContent = cmPerPx ? `${(pxToCm(heightPx)).toFixed(1)} سم` : '— سم';
    } else {
      hPxEl.textContent = '— px'; hCmEl.textContent = '— سم';
    }

    ctx.restore();
  }

  requestAnimationFrame(loop);
}

function snapshot() {
  const temp = document.createElement('canvas');
  temp.width = canvas.width; temp.height = canvas.height;
  const tctx = temp.getContext('2d');
  tctx.drawImage(videoEl, 0, 0, temp.width, temp.height);
  tctx.drawImage(canvas, 0, 0);
  const a = document.createElement('a');
  a.download = `mirrorfit_${Date.now()}.png`;
  a.href = temp.toDataURL('image/png');
  a.click();
}

function autoCalibrate() {
  const swText = document.getElementById('shoulderWidthPx').textContent;
  const m = /([\d.]+)\s*px/.exec(swText);
  if (!m) { alert('قِف قدّام الكاميرا بحيث كتافك باينين الأول.'); return; }
  const shoulderPx = parseFloat(m[1]);
  if (shoulderPx <= 0) { alert('تعذر القياس، جرّب تاني.'); return; }
  const shoulderCm = parseFloat(prompt('أدخل عرض الكتف بالسنتيمتر (افتراضي 40 سم):', '40')) || 40;
  cmPerPx = shoulderCm / shoulderPx;
  document.getElementById('cmPerPx').value = cmPerPx.toFixed(4);
}

window.addEventListener('load', setup);
