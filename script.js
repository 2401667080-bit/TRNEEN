(function(){
  const canvas = document.getElementById('curve');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('status');
  const scoreEl = document.getElementById('score');
  const readoutEl = document.getElementById('readout');
  const uploadBtn = document.getElementById('uploadBtn');
  const fileInput = document.getElementById('fileInput');
  const playBtn = document.getElementById('playBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const resumeBtn = document.getElementById('resumeBtn');
  const recordBtn = document.getElementById('recordBtn');
  const resetBtn = document.getElementById('resetBtn');
  const glyphKey = document.getElementById('glyphKey');
  const modeTraditionalBtn = document.getElementById('modeTraditional');
  const modePreciseBtn = document.getElementById('modePrecise');
  const verseTextarea = document.getElementById('verse');
  const wordRow = document.getElementById('wordRow');
  const alignNote = document.getElementById('alignNote');
  const engineBadge = document.getElementById('engineBadge');
  const sensitivitySlider = document.getElementById('sensitivitySlider');
  const sensitivityValue = document.getElementById('sensitivityValue');

  let viewMode = 'traditional';
  let audioCtx = null;
  let referenceCurve = null;
  let referenceDuration = 0;
  let referenceAudioBuffer = null;
  let liveCurve = [];
  let isRecording = false;
  let recordStartTime = 0;
  let playheadTime = null;
  let playing = false;
  let isPaused = false;
  let playbackOffset = 0;
  let playSource = null;
  let playStartCtxTime = 0;
  let mediaStream = null;
  let scriptNode = null;
  let micSource = null;
  let liveMaxRms = 0;
  let alignedWords = [];

  let displayMin = 55, displayMax = 79;
  const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

  // ---------- CREPE (pretrained pitch model), loaded and run directly ----------
  // Runs the raw TensorFlow.js model ourselves (no streaming wrapper), so
  // file analysis processes a fixed set of frames deterministically —
  // it can't fall behind real-time playback, can't skip the tail, and
  // doesn't need to actually play the audio through the graph at all
  // (which was the source of the lag/heaviness/incomplete-tail issues).
  const CREPE_MODEL_PATH = './crepe-model/model.json';
  const CREPE_SAMPLE_RATE = 16000;
  const CREPE_FRAME_SIZE = 1024;
  let crepeSupported = null; // null = unknown, true/false once tested
  let crepeModelPromise = null;

  function setEngineBadge(text, ok){
    engineBadge.textContent = text;
    engineBadge.classList.toggle('crepe', !!ok);
  }

  function getCrepeModel(){
    if (!crepeModelPromise){
      crepeModelPromise = (async () => {
        if (typeof tf === 'undefined'){
          console.warn('[مسار الصوت] tf.js لم يُحمَّل — يُستخدم الكاشف البديل.');
          return null;
        }
        try{ return await tf.loadLayersModel(CREPE_MODEL_PATH); }
        catch(e){ console.error('[مسار الصوت] فشل تحميل نموذج CREPE:', e); return null; }
      })();
    }
    return crepeModelPromise;
  }

  // Standard CREPE output decoding (360 pitch-salience bins spanning ~6
  // octaves in 20-cent steps) — published as part of the original CREPE
  // paper/reference implementation, not something specific to this app.
  const CENTS_MAPPING = (() => {
    const arr = new Float32Array(360);
    for (let i=0;i<360;i++) arr[i] = (7180*i/359) + 1997.3794084376191;
    return arr;
  })();
  function decodeCrepeActivation(activations){
    let maxIdx=0, maxVal=-Infinity;
    for (let i=0;i<activations.length;i++){ if (activations[i]>maxVal){ maxVal=activations[i]; maxIdx=i; } }
    const start=Math.max(0,maxIdx-4), end=Math.min(activations.length-1,maxIdx+4);
    let num=0, den=0;
    for (let i=start;i<=end;i++){ num += activations[i]*CENTS_MAPPING[i]; den += activations[i]; }
    const cents = den>0 ? num/den : CENTS_MAPPING[maxIdx];
    return { freq: 10*Math.pow(2, cents/1200), confidence: maxVal };
  }
  function normalizeFrame(frame){
    let mean=0; for (let i=0;i<frame.length;i++) mean+=frame[i]; mean/=frame.length;
    let variance=0; for (let i=0;i<frame.length;i++){ const d=frame[i]-mean; variance+=d*d; } variance/=frame.length;
    const std = Math.sqrt(variance) || 1e-8;
    const out = new Float32Array(frame.length);
    for (let i=0;i<frame.length;i++) out[i] = (frame[i]-mean)/std;
    return out;
  }
  function resampleLinear(src, fromRate, toRate){
    if (fromRate === toRate) return src;
    const ratio = fromRate/toRate;
    const newLen = Math.max(1, Math.round(src.length/ratio));
    const out = new Float32Array(newLen);
    for (let i=0;i<newLen;i++){
      const p=i*ratio, i0=Math.floor(p), i1=Math.min(i0+1, src.length-1), f=p-i0;
      out[i] = src[i0]*(1-f) + src[i1]*f;
    }
    return out;
  }
  async function resampleBufferTo16k(audioBuffer){
    const targetLen = Math.ceil(audioBuffer.duration * CREPE_SAMPLE_RATE);
    const offlineCtx = new OfflineAudioContext(1, targetLen, CREPE_SAMPLE_RATE);
    const src = offlineCtx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(offlineCtx.destination);
    src.start();
    const rendered = await offlineCtx.startRendering();
    return rendered.getChannelData(0);
  }
  async function predictSingleFrame(model, frame1024){
    const norm = normalizeFrame(frame1024);
    const inputTensor = tf.tensor2d(norm, [1, CREPE_FRAME_SIZE]);
    const outputTensor = model.predict(inputTensor);
    const outputData = await outputTensor.data();
    inputTensor.dispose(); outputTensor.dispose();
    return decodeCrepeActivation(outputData);
  }

  // Quick background probe on page load so the badge reflects reality
  // before the user even uploads anything.
  (async function probeCrepe(){
    const model = await getCrepeModel();
    crepeSupported = !!model;
    setEngineBadge(crepeSupported ? 'محرك CREPE (ذكاء اصطناعي مُدرَّب) نشط' : 'تعذّر تحميل CREPE — تعمل الخوارزمية البديلة', crepeSupported);
  })();

  modeTraditionalBtn.addEventListener('click', () => setMode('traditional'));
  modePreciseBtn.addEventListener('click', () => setMode('precise'));
  sensitivitySlider.addEventListener('input', () => {
    sensitivity = Number(sensitivitySlider.value)/100;
    sensitivityValue.textContent = sensitivity<0.34 ? 'منخفضة' : sensitivity<0.67 ? 'متوسطة' : 'عالية';
    refPiecesCache = null; livePiecesCache = null;
    render();
  });
  function setMode(m){
    viewMode = m;
    modeTraditionalBtn.classList.toggle('active', m==='traditional');
    modePreciseBtn.classList.toggle('active', m==='precise');
    glyphKey.style.display = (m==='traditional') ? 'flex' : 'none';
    render();
  }

  function ensureAudioCtx(){
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return audioCtx;
  }
  function freqToMidi(freq){ if (!freq || freq<=0) return null; return 69 + 12*Math.log2(freq/440); }
  function midiLabel(m){
    const idx=((Math.round(m)%12)+12)%12, oct=Math.floor(Math.round(m)/12)-1;
    return NOTE_NAMES[idx]+oct;
  }

  // ---------- Fallback: autocorrelation pitch detector (ACF2+ style) ----------
  function autoCorrelate(buf, sampleRate){
    let SIZE=buf.length, rms=0;
    for (let i=0;i<SIZE;i++){ const v=buf[i]; rms+=v*v; }
    rms=Math.sqrt(rms/SIZE);
    if (rms<0.004) return { freq:-1, clarity:0, rms };
    let r1=0, r2=SIZE-1; const thres=0.2;
    for (let i=0;i<SIZE/2;i++){ if (Math.abs(buf[i])<thres){ r1=i; break; } }
    for (let i=1;i<SIZE/2;i++){ if (Math.abs(buf[SIZE-i])<thres){ r2=SIZE-i; break; } }
    const trimmed=buf.slice(r1,r2), N=trimmed.length;
    if (N<8) return { freq:-1, clarity:0, rms };
    const c=new Float32Array(N);
    for (let i=0;i<N;i++){ let sum=0; for (let j=0;j<N-i;j++) sum+=trimmed[j]*trimmed[j+i]; c[i]=sum; }
    let d=0; while (d<N-1 && c[d]>c[d+1]) d++;
    let maxVal=-1, maxPos=-1;
    for (let i=d;i<N;i++){ if (c[i]>maxVal){ maxVal=c[i]; maxPos=i; } }
    if (maxPos<=0 || !c[0]) return { freq:-1, clarity:0, rms };
    const clarity=maxVal/c[0];
    let T0=maxPos;
    const x1=c[T0-1]||0, x2=c[T0], x3=(T0+1<N?c[T0+1]:0);
    const a=(x1+x3-2*x2)/2, b=(x3-x1)/2;
    if (a) T0=T0-b/(2*a);
    const freq=sampleRate/T0;
    if (freq<60 || freq>700) return { freq:-1, clarity, rms };
    return { freq, clarity, rms };
  }
  function bufRms(buf){
    let rms=0; for (let i=0;i<buf.length;i++) rms += buf[i]*buf[i];
    return Math.sqrt(rms/buf.length);
  }

  // ---------- Offline analysis of uploaded file ----------
  // Tries CREPE first as a deterministic batch pass over every frame in
  // the file (not tied to real-time playback). Falls back to the
  // autocorrelation pass if the model isn't available.
  async function analyzeFile(arrayBuffer){
    const ctxA = ensureAudioCtx();
    if (ctxA.state==='suspended') await ctxA.resume();
    const audioBuffer = await ctxA.decodeAudioData(arrayBuffer);
    referenceAudioBuffer = audioBuffer;
    referenceDuration = audioBuffer.duration;

    let raw = null;
    if (crepeSupported !== false){
      raw = await analyzeWithCrepeDirect(audioBuffer, (pct) => {
        statusEl.textContent = `جارٍ التحليل عبر CREPE... ${pct}%`;
      });
      if (raw){ crepeSupported = true; setEngineBadge('محرك CREPE (ذكاء اصطناعي مُدرَّب) نشط', true); }
    }
    if (!raw){
      if (crepeSupported === null){ crepeSupported = false; setEngineBadge('تعذّر تحميل CREPE — تعمل الخوارزمية البديلة', false); }
      statusEl.textContent = 'جارٍ التحليل بالخوارزمية البديلة...';
      raw = analyzeWithAutocorrelate(audioBuffer);
    }
    finalizeCurve(raw);
    computeWordAlignment();
  }

  async function analyzeWithCrepeDirect(audioBuffer, onProgress){
    const model = await getCrepeModel();
    if (!model) return null;

    const data16k = await resampleBufferTo16k(audioBuffer);
    const hop = 640; // 40ms hop at 16kHz — matches prior ~25 points/sec resolution
    const totalFrames = Math.max(1, Math.floor((data16k.length - CREPE_FRAME_SIZE) / hop) + 1);
    const BATCH = 24;
    const raw = [];

    for (let b=0; b<totalFrames; b+=BATCH){
      const count = Math.min(BATCH, totalFrames-b);
      const batchData = new Float32Array(count*CREPE_FRAME_SIZE);
      const rmsList = [];
      for (let k=0;k<count;k++){
        const start = (b+k)*hop;
        let frame = data16k.subarray(start, start+CREPE_FRAME_SIZE);
        if (frame.length < CREPE_FRAME_SIZE){
          const padded = new Float32Array(CREPE_FRAME_SIZE);
          padded.set(frame);
          frame = padded;
        }
        rmsList.push(bufRms(frame));
        batchData.set(normalizeFrame(frame), k*CREPE_FRAME_SIZE);
      }
      const inputTensor = tf.tensor2d(batchData, [count, CREPE_FRAME_SIZE]);
      const outputTensor = model.predict(inputTensor);
      const outputData = await outputTensor.data();
      inputTensor.dispose(); outputTensor.dispose();

      for (let k=0;k<count;k++){
        const activations = outputData.subarray(k*360, (k+1)*360);
        const { freq, confidence } = decodeCrepeActivation(activations);
        const t = (b+k)*hop/CREPE_SAMPLE_RATE;
        raw.push({ t, freq: confidence>0.5 ? freq : -1, rms: rmsList[k] });
      }
      if (onProgress) onProgress(Math.min(100, Math.round(((b+count)/totalFrames)*100)));
      await new Promise(r => setTimeout(r, 0)); // yield so the page stays responsive
    }
    return raw;
  }

  function analyzeWithAutocorrelate(audioBuffer){
    const data = audioBuffer.getChannelData(0);
    const sr = audioBuffer.sampleRate;
    const duration = audioBuffer.duration;
    const frameSize = 1024;
    const points = Math.max(80, Math.min(900, Math.round(duration*35)));
    function getFrame(t){
      const start=Math.floor(t*sr);
      if (start+frameSize<=data.length) return data.subarray(start, start+frameSize);
      const frame=new Float32Array(frameSize);
      frame.set(data.subarray(start, data.length));
      return frame;
    }
    const raw=[];
    for (let i=0;i<points;i++){
      const t=(i/(points-1))*duration;
      const { freq, rms } = autoCorrelate(getFrame(t), sr);
      raw.push({ t, freq, rms });
    }
    return raw;
  }

  function finalizeCurve(raw){
    const sortedRms = raw.map(p=>p.rms).slice().sort((a,b)=>a-b);
    const loudRef = sortedRms[Math.floor(sortedRms.length*0.85)] || 0;
    const energyGate = loudRef * 0.28;
    const curve = raw.map(p => {
      const voiced = p.freq>0 && p.rms>=energyGate;
      return { t:p.t, semi: voiced ? freqToMidi(p.freq) : null };
    });
    for (let i=1;i<curve.length-1;i++){
      if (curve[i].semi===null) continue;
      const prev=curve[i-1].semi, next=curve[i+1].semi;
      if (prev===null && next===null) continue;
      const nAvg=(prev!==null&&next!==null)?(prev+next)/2:(prev!==null?prev:next);
      if (Math.abs(curve[i].semi-nAvg) > 9) curve[i].semi = null;
    }
    referenceCurve = curve;
    const voicedVals = curve.filter(p=>p.semi!==null).map(p=>p.semi);
    if (voicedVals.length){
      displayMin = Math.floor(Math.min(...voicedVals)) - 1;
      displayMax = Math.ceil(Math.max(...voicedVals)) + 1;
      if (displayMax-displayMin<3){ displayMin-=1; displayMax+=1; }
    }
    render();
    statusEl.textContent = 'تم رسم المسار الصوتي — استمع للأصل أو سجّل صوتك';
    playBtn.disabled=false; recordBtn.disabled=false;
  }

  // ---------- Shared geometry ----------
  function fitCanvas(){
    const rect=canvas.getBoundingClientRect(); const dpr=window.devicePixelRatio||1;
    canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  function semiToY(semi,h){
    const clamped=Math.max(displayMin, Math.min(displayMax, semi));
    const ratio=(clamped-displayMin)/(displayMax-displayMin);
    return h - ratio*h*0.78 - h*0.10;
  }
  function drawNoteAxis(w,h){
    ctx.font='10px Cairo, sans-serif'; ctx.textAlign='left'; ctx.textBaseline='middle';
    for (let m=Math.ceil(displayMin); m<=Math.floor(displayMax); m++){
      const y=semiToY(m,h);
      ctx.strokeStyle='rgba(169,120,47,0.18)'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
      ctx.fillStyle='rgba(30,27,22,0.5)'; ctx.fillText(midiLabel(m), 4, y);
    }
  }
  function drawPreciseCurve(points,w,h,duration,color,lineWidth){
    ctx.strokeStyle=color; ctx.lineWidth=lineWidth; ctx.lineJoin='round'; ctx.lineCap='round';
    let drawing=false, prevY=0;
    ctx.beginPath();
    for (const p of points){
      const x=(p.t/duration)*w;
      if (p.semi===null){ drawing=false; continue; }
      const y=semiToY(p.semi,h);
      if (!drawing){ ctx.moveTo(x,y); drawing=true; } else { ctx.lineTo(x,prevY); ctx.lineTo(x,y); }
      prevY=y;
    }
    ctx.stroke();
  }

  // ---------- Traditional (symbolic) analysis + drawing ----------
  function splitRuns(curve){
    const runs=[]; let cur=[];
    for (const p of curve){
      if (p.semi===null){ if (cur.length>3) runs.push(cur); cur=[]; }
      else cur.push(p);
    }
    if (cur.length>3) runs.push(cur);
    return runs;
  }
  function smoothRun(run){
    return run.map((p,i)=>{
      const a=run[Math.max(0,i-1)].semi, c=run[Math.min(run.length-1,i+1)].semi;
      return { t:p.t, semi: p.semi*0.5 + a*0.25 + c*0.25 };
    });
  }
  // Detection sensitivity (0..1), user-adjustable via the slider — scales
  // between conservative (fewer false positives) and sensitive (catches
  // subtler ornaments) without needing to re-analyze the audio.
  let sensitivity = 0.5;
  function sensParams(){
    // At sensitivity 0: strict (amp 2.6, dur 0.32, rdp 8, minPts 5, minDur .38)
    // At sensitivity 1: loose  (amp 1.0, dur 0.12, rdp 3, minPts 3, minDur .18)
    const t = sensitivity;
    return {
      scoopAmp: 2.6 - 1.6*t,
      scoopDurMin: 0.32 - 0.20*t,
      rdpEps: 8 - 5*t,
      zigzagMinPts: Math.round(5 - 2*t),
      zigzagMinDur: 0.38 - 0.20*t
    };
  }
  function findScoopZones(run){
    const { scoopAmp, scoopDurMin } = sensParams();
    const zones=[]; let i=0;
    while (i<run.length-2){
      let j=i; while (j<run.length-1 && run[j+1].semi>=run[j].semi-0.05) j++;
      const peak=j;
      let k=peak; while (k<run.length-1 && run[k+1].semi<=run[k].semi+0.05) k++;
      const end=k;
      if (peak>i && end>peak){
        const amp=run[peak].semi-Math.min(run[i].semi, run[end].semi);
        const dur=run[end].t-run[i].t;
        if (amp>scoopAmp && dur>scoopDurMin && dur<2.2){ zones.push({start:i,peak,end}); i=end; continue; }
      }
      i++;
    }
    return zones;
  }
  function rdpSimplify(pts, epsilon){
    if (pts.length<3) return pts.slice();
    function perpDist(p,a,b){
      const dx=b.x-a.x, dy=b.y-a.y, norm=Math.hypot(dx,dy);
      if (norm===0) return Math.hypot(p.x-a.x,p.y-a.y);
      return Math.abs(dy*p.x - dx*p.y + b.x*a.y - b.y*a.x)/norm;
    }
    function rec(arr){
      let dmax=0, idx=0; const end=arr.length-1;
      for (let i=1;i<end;i++){ const d=perpDist(arr[i],arr[0],arr[end]); if (d>dmax){ dmax=d; idx=i; } }
      if (dmax>epsilon){
        const left=rec(arr.slice(0,idx+1)), right=rec(arr.slice(idx));
        return left.slice(0,-1).concat(right);
      }
      return [arr[0], arr[end]];
    }
    return rec(pts);
  }
  function buildTraditionalRuns(curve, w, h, duration){
    const runs = splitRuns(curve);
    const out = [];
    for (const rawRun of runs){
      const run = smoothRun(smoothRun(rawRun));
      const scoops = findScoopZones(run);
      let cursor=0;
      const rawPieces=[];
      for (const z of scoops){
        if (z.start>cursor) rawPieces.push({ kind:'line', pts: run.slice(cursor, z.start+1) });
        rawPieces.push({ kind:'scoop', pts: run.slice(z.start, z.end+1), peak: run[z.peak] });
        cursor = z.end;
      }
      if (cursor < run.length-1) rawPieces.push({ kind:'line', pts: run.slice(cursor) });
      const toXY = p => ({ x:(p.t/duration)*w, y:semiToY(p.semi,h), t:p.t, semi:p.semi });
      const pieces = [];
      for (const rp of rawPieces){
        if (rp.pts.length<2) continue;
        if (rp.kind==='scoop'){
          const a=toXY(rp.pts[0]), b=toXY(rp.peak), c=toXY(rp.pts[rp.pts.length-1]);
          pieces.push({ type:'scoop', a, b, c, tStart:a.t, tEnd:c.t, xMid:(a.x+c.x)/2, yTag:Math.min(a.y,b.y,c.y) });
        } else {
          const px = rp.pts.map(toXY);
          const { rdpEps, zigzagMinPts, zigzagMinDur } = sensParams();
          const simplified = rdpSimplify(px, rdpEps);
          const netSlope = simplified[simplified.length-1].semi - simplified[0].semi;
          const pieceDur = px[px.length-1].t - px[0].t;
          const type = (simplified.length>=zigzagMinPts && pieceDur>=zigzagMinDur) ? 'zigzag'
                     : Math.abs(netSlope)>1.6 ? (netSlope>0?'rise':'fall')
                     : 'flat';
          const mid = simplified[Math.floor(simplified.length/2)];
          pieces.push({ type, points:simplified, tStart:px[0].t, tEnd:px[px.length-1].t, xMid:mid.x, yTag:Math.min(...simplified.map(p=>p.y)) });
        }
      }
      out.push(pieces);
    }
    return out;
  }
  let refPiecesCache = null;  // { w, h, duration, curve, sensitivity, pieces }
  let livePiecesCache = null;

  // Draws from already-computed pieces (no recomputation) — used on every
  // animation frame during playback so redraws stay cheap.
  function strokeTraditionalFromRunGroups(runGroups, color, lineWidth){
    ctx.strokeStyle=color; ctx.lineWidth=lineWidth; ctx.lineJoin='round'; ctx.lineCap='round';
    const annotations = [];
    for (const pieces of runGroups){
      ctx.beginPath();
      let started=false;
      for (const piece of pieces){
        if (piece.type==='scoop'){
          if (!started){ ctx.moveTo(piece.a.x, piece.a.y); started=true; } else ctx.lineTo(piece.a.x, piece.a.y);
          const ctrlY = piece.b.y - Math.abs(piece.a.y-piece.b.y)*0.35 - 4;
          ctx.quadraticCurveTo(piece.b.x, ctrlY, piece.c.x, piece.c.y);
        } else {
          for (const pt of piece.points){
            if (!started){ ctx.moveTo(pt.x, pt.y); started=true; } else ctx.lineTo(pt.x, pt.y);
          }
        }
        if (piece.type !== 'flat') annotations.push(piece);
      }
      ctx.stroke();
    }
    return annotations;
  }
  const TYPE_LABEL = { rise:'صعود', fall:'هبوط', zigzag:'زخرفة', scoop:'اغتراف' };
  function drawAnnotations(annotations, color){
    ctx.font = '10px Cairo, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = color;
    for (const a of annotations){
      const label = TYPE_LABEL[a.type];
      if (!label) continue;
      ctx.fillText(label, a.xMid, Math.max(11, a.yTag - 7));
    }
  }

  function render(){
    fitCanvas();
    const rect = canvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    ctx.clearRect(0,0,w,h);
    const duration = Math.max(referenceDuration, isRecording ? (Date.now()-recordStartTime)/1000 : 0, 3);
    if (viewMode==='precise') drawNoteAxis(w,h);
    const inkColor = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
    const redColor = getComputedStyle(document.documentElement).getPropertyValue('--tajweed-red').trim();
    if (referenceCurve){
      if (viewMode==='traditional'){
        let ann;
        if (refPiecesCache && refPiecesCache.curve===referenceCurve && refPiecesCache.w===w && refPiecesCache.h===h && refPiecesCache.duration===duration && refPiecesCache.sensitivity===sensitivity){
          ann = strokeTraditionalFromRunGroups(refPiecesCache.pieces, inkColor, 2.4);
        } else {
          const pieces = buildTraditionalRuns(referenceCurve, w, h, duration);
          refPiecesCache = { w, h, duration, curve:referenceCurve, sensitivity, pieces };
          ann = strokeTraditionalFromRunGroups(pieces, inkColor, 2.4);
        }
        drawAnnotations(ann, 'rgba(30,27,22,0.65)');
      } else drawPreciseCurve(referenceCurve, w, h, duration, inkColor, 2.2);
    }
    if (liveCurve.length){
      if (viewMode==='traditional'){
        let ann2;
        if (!isRecording && livePiecesCache && livePiecesCache.curve===liveCurve && livePiecesCache.w===w && livePiecesCache.h===h && livePiecesCache.duration===duration && livePiecesCache.sensitivity===sensitivity){
          ann2 = strokeTraditionalFromRunGroups(livePiecesCache.pieces, redColor, 2.4);
        } else {
          const pieces = buildTraditionalRuns(liveCurve, w, h, duration);
          if (!isRecording) livePiecesCache = { w, h, duration, curve:liveCurve, sensitivity, pieces };
          ann2 = strokeTraditionalFromRunGroups(pieces, redColor, 2.4);
        }
        drawAnnotations(ann2, 'rgba(165,52,44,0.7)');
      } else drawPreciseCurve(liveCurve, w, h, duration, redColor, 2.2);
    }
    if (playheadTime !== null){
      const x=(playheadTime/duration)*w;
      ctx.strokeStyle='rgba(169,120,47,0.65)'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
    }
    renderWordRow(duration);
  }
  function loop(){ render(); if (isRecording || playing) requestAnimationFrame(loop); }
  window.addEventListener('resize', render);

  function updateReadout(freq, midi){
    if (freq && freq>0 && midi!==null){
      readoutEl.innerHTML = `<span class="note">${midiLabel(midi)}</span><span class="freq">${freq.toFixed(1)}Hz</span>`;
    } else {
      readoutEl.innerHTML = `<span class="placeholder">لا يلتقط صوتاً واضحاً الآن</span>`;
    }
  }

  // ---------- Word alignment (heuristic — not real speech recognition) ----------
  function computeWordAlignment(){
    const text = verseTextarea.value.trim();
    if (!text || !referenceCurve){ alignedWords=[]; renderWordRow(referenceDuration||3); return; }
    const words = text.split(/\s+/).filter(Boolean);
    const runs = splitRuns(referenceCurve);
    if (!words.length || !runs.length){ alignedWords=[]; renderWordRow(referenceDuration||3); return; }
    const runDurations = runs.map(r => r[r.length-1].t - r[0].t);
    const totalDur = runDurations.reduce((a,b)=>a+b,0) || 1;
    let counts = runDurations.map(d => Math.max(1, Math.round(words.length*d/totalDur)));
    let diff = words.length - counts.reduce((a,b)=>a+b,0);
    let gi=0, guard=0;
    while (diff!==0 && guard<2000){
      guard++; const i=gi%counts.length;
      if (diff>0){ counts[i]++; diff--; } else if (counts[i]>1){ counts[i]--; diff++; }
      gi++;
    }
    const result=[]; let wPtr=0;
    runs.forEach((run,ri)=>{
      const n=counts[ri]; const rStart=run[0].t, rEnd=run[run.length-1].t, span=rEnd-rStart;
      for (let k=0;k<n;k++){
        if (wPtr>=words.length) break;
        result.push({ word:words[wPtr], tStart: rStart+span*k/n, tEnd: rStart+span*(k+1)/n });
        wPtr++;
      }
    });
    while (wPtr < words.length){ result.push({ word:words[wPtr], tStart: referenceDuration, tEnd: referenceDuration }); wPtr++; }
    alignedWords = result;
    renderWordRow(referenceDuration||3);
  }
  verseTextarea.addEventListener('input', () => { if (referenceCurve) computeWordAlignment(); });

  function renderWordRow(duration){
    wordRow.innerHTML = '';
    if (!alignedWords.length){ alignNote.style.display='none'; return; }
    alignNote.style.display = 'block';
    for (const wd of alignedWords){
      const el = document.createElement('span');
      el.className = 'word';
      el.textContent = wd.word;
      const xPct = (wd.tStart/duration)*100;
      el.style.right = `calc(${100-xPct}% )`;
      el.style.left = 'auto';
      el.style.transform = 'translateX(50%)';
      if (playheadTime!==null && playheadTime>=wd.tStart && playheadTime<wd.tEnd) el.classList.add('playing');
      el.addEventListener('click', () => { seekAndPlay(wd.tStart); });
      wordRow.appendChild(el);
    }
  }

  // ---------- Upload ----------
  uploadBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    statusEl.textContent = 'جارٍ تحليل المقطع الصوتي...';
    playBtn.disabled=true; recordBtn.disabled=true; pauseBtn.disabled=true; resumeBtn.disabled=true;
    try{
      const buf = await file.arrayBuffer();
      await analyzeFile(buf);
    }catch(err){
      console.error(err);
      statusEl.textContent = 'تعذّر قراءة هذا الملف الصوتي، جرّب ملفاً آخر';
    }
  });

  // ---------- Playback: play / pause / resume / seek ----------
  async function playFrom(offset){
    if (!referenceAudioBuffer) return;
    const ctxA = ensureAudioCtx();
    if (ctxA.state==='suspended') await ctxA.resume();
    if (playSource){ try{ playSource.stop(); }catch(e){} playSource.disconnect(); playSource=null; }
    const src = ctxA.createBufferSource();
    src.buffer = referenceAudioBuffer;
    src.connect(ctxA.destination);
    playStartCtxTime = ctxA.currentTime - offset;
    playSource = src;
    playing = true; isPaused = false;
    src.start(0, Math.max(0, offset));
    updatePlaybackButtons();
    statusEl.textContent = 'يتم الآن تشغيل التلاوة الأصلية...';
    src.onended = () => {
      if (playing){ playing=false; playheadTime=null; playbackOffset=0; isPaused=false; updatePlaybackButtons(); statusEl.textContent='انتهى التشغيل'; render(); }
    };
    loop();
    const tick = () => {
      if (!playing) return;
      playheadTime = ctxA.currentTime - playStartCtxTime;
      if (playheadTime >= referenceDuration) return;
      requestAnimationFrame(tick);
    };
    tick();
  }
  function pausePlayback(){
    if (!playing) return;
    playbackOffset = playheadTime || 0;
    isPaused = true; playing = false;
    if (playSource){ try{ playSource.stop(); }catch(e){} playSource.disconnect(); playSource=null; }
    statusEl.textContent = 'تم الإيقاف المؤقت';
    updatePlaybackButtons(); render();
  }
  function seekAndPlay(t){
    playbackOffset = Math.max(0, Math.min(referenceDuration, t));
    playFrom(playbackOffset);
  }
  function updatePlaybackButtons(){
    pauseBtn.disabled = !playing;
    resumeBtn.disabled = !(isPaused && !playing && playbackOffset>0);
    playBtn.disabled = !referenceAudioBuffer;
  }
  playBtn.addEventListener('click', () => seekAndPlay(0));
  pauseBtn.addEventListener('click', pausePlayback);
  resumeBtn.addEventListener('click', () => playFrom(playbackOffset));
  canvas.addEventListener('click', (e) => {
    if (!referenceAudioBuffer) return;
    const rect = canvas.getBoundingClientRect();
    const duration = Math.max(referenceDuration, 3);
    const t = ((e.clientX-rect.left)/rect.width) * duration;
    seekAndPlay(t);
  });

  // ---------- Recording (CREPE first, autocorrelation fallback) ----------
  recordBtn.addEventListener('click', async () => {
    if (!isRecording){
      try{ mediaStream = await navigator.mediaDevices.getUserMedia({ audio:true }); }
      catch(err){ statusEl.textContent='تعذّر الوصول إلى الميكروفون — تأكد من منح الإذن، ومن فتح الصفحة عبر رابط https'; return; }
      const ctxA = ensureAudioCtx();
      if (ctxA.state==='suspended') await ctxA.resume();

      liveCurve=[]; isRecording=true; recordStartTime=Date.now(); liveMaxRms=0;
      recordBtn.textContent='إيقاف التسجيل'; recordBtn.classList.add('active');
      scoreEl.textContent=''; loop();

      let crepeModel = null;
      if (crepeSupported !== false){
        statusEl.textContent = 'جارٍ تجهيز محرك CREPE للتسجيل الحي...';
        crepeModel = await getCrepeModel();
        if (crepeModel){ crepeSupported = true; setEngineBadge('محرك CREPE (ذكاء اصطناعي مُدرَّب) نشط', true); }
      }

      if (crepeModel){
        statusEl.textContent = 'يتم التسجيل الآن (CREPE)... اتّبع المسار الأسود';
        micSource = ctxA.createMediaStreamSource(mediaStream);
        scriptNode = ctxA.createScriptProcessor(4096,1,1);
        const silentGain = ctxA.createGain(); silentGain.gain.value=0;
        let busy = false;
        scriptNode.onaudioprocess = (ev) => {
          if (!isRecording || busy) return;
          busy = true;
          const input = ev.inputBuffer.getChannelData(0).slice();
          const rms = bufRms(input);
          liveMaxRms = Math.max(liveMaxRms*0.985, rms);
          const resampled = resampleLinear(input, ctxA.sampleRate, CREPE_SAMPLE_RATE);
          let frame;
          if (resampled.length >= CREPE_FRAME_SIZE){
            frame = resampled.subarray(resampled.length - CREPE_FRAME_SIZE);
          } else {
            frame = new Float32Array(CREPE_FRAME_SIZE);
            frame.set(resampled, CREPE_FRAME_SIZE - resampled.length);
          }
          const t = (Date.now()-recordStartTime)/1000;
          predictSingleFrame(crepeModel, frame).then(({ freq, confidence }) => {
            const voiced = confidence>0.5 && rms>=liveMaxRms*0.28;
            const midi = voiced ? freqToMidi(freq) : null;
            liveCurve.push({ t, semi: midi });
            updateReadout(voiced?freq:null, midi);
            busy = false;
          }).catch(() => { busy = false; });
        };
        micSource.connect(scriptNode); scriptNode.connect(silentGain); silentGain.connect(ctxA.destination);
      } else {
        if (crepeSupported === null){ crepeSupported=false; setEngineBadge('تعذّر تحميل CREPE — تعمل الخوارزمية البديلة', false); }
        statusEl.textContent = 'يتم التسجيل الآن (الخوارزمية البديلة)... اتّبع المسار الأسود';
        micSource = ctxA.createMediaStreamSource(mediaStream);
        scriptNode = ctxA.createScriptProcessor(4096,1,1);
        const silentGain = ctxA.createGain(); silentGain.gain.value=0;
        scriptNode.onaudioprocess = (ev) => {
          if (!isRecording) return;
          const input = ev.inputBuffer.getChannelData(0);
          const { freq, clarity, rms } = autoCorrelate(input, ctxA.sampleRate);
          liveMaxRms = Math.max(liveMaxRms*0.985, rms);
          const voiced = freq>0 && clarity>=0.3 && rms>=liveMaxRms*0.28;
          const midi = voiced ? freqToMidi(freq) : null;
          const t = (Date.now()-recordStartTime)/1000;
          liveCurve.push({ t, semi: midi });
          updateReadout(voiced?freq:null, midi);
        };
        micSource.connect(scriptNode); scriptNode.connect(silentGain); silentGain.connect(ctxA.destination);
      }
    } else {
      isRecording=false;
      recordBtn.textContent='تسجيل صوتك'; recordBtn.classList.remove('active');
      if (scriptNode){ scriptNode.disconnect(); scriptNode=null; }
      if (micSource){ micSource.disconnect(); micSource=null; }
      if (mediaStream) mediaStream.getTracks().forEach(t=>t.stop());
      statusEl.textContent='تم إيقاف التسجيل';
      updateReadout(null,null);
      computeScore(); render();
    }
  });

  // ---------- Similarity score ----------
  function resampleVoiced(curve,n,duration){
    const out=new Array(n).fill(null);
    for (let i=0;i<n;i++){
      const t=(i/(n-1))*duration; let before=null, after=null;
      for (const p of curve){
        if (p.semi===null) continue;
        if (p.t<=t) before=p;
        if (p.t>=t && after===null) after=p;
      }
      if (before && after && before!==after){
        const span=after.t-before.t, ratio=span>0?(t-before.t)/span:0;
        out[i]=before.semi+(after.semi-before.semi)*ratio;
      } else if (before) out[i]=before.semi; else if (after) out[i]=after.semi;
    }
    return out;
  }
  function computeScore(){
    if (!referenceCurve || liveCurve.length<5){ scoreEl.textContent=''; return; }
    const N=80; const duration=Math.max(referenceDuration, liveCurve[liveCurve.length-1].t);
    const a=resampleVoiced(referenceCurve,N,duration), b=resampleVoiced(liveCurve,N,duration);
    const pairs=[];
    for (let i=0;i<N;i++) if (a[i]!==null && b[i]!==null) pairs.push([a[i],b[i]]);
    if (pairs.length<10){ scoreEl.innerHTML='لم يُلتقط صوت كافٍ للمقارنة — حاول مرة أخرى'; return; }
    const meanA=pairs.reduce((s,p)=>s+p[0],0)/pairs.length, meanB=pairs.reduce((s,p)=>s+p[1],0)/pairs.length;
    let num=0,denA=0,denB=0;
    for (const [x,y] of pairs){ num+=(x-meanA)*(y-meanB); denA+=(x-meanA)**2; denB+=(y-meanB)**2; }
    const corr=(denA>0&&denB>0)?num/Math.sqrt(denA*denB):0;
    const pct=Math.round(Math.max(0,corr)*100);
    let label='حاول مرة أخرى';
    if (pct>=85) label='ممتاز'; else if (pct>=65) label='جيد جداً'; else if (pct>=45) label='جيد، استمر بالتدريب';
    scoreEl.innerHTML = `نسبة تطابق اللحن: <b>${pct}%</b> — ${label}`;
  }

  // ---------- Reset ----------
  resetBtn.addEventListener('click', () => {
    referenceCurve=null; referenceDuration=0; referenceAudioBuffer=null;
    liveCurve=[]; isRecording=false; playing=false; isPaused=false; playbackOffset=0; playheadTime=null;
    displayMin=55; displayMax=79; alignedWords=[]; refPiecesCache=null; livePiecesCache=null;
    playBtn.disabled=true; recordBtn.disabled=true; pauseBtn.disabled=true; resumeBtn.disabled=true;
    recordBtn.textContent='تسجيل صوتك'; recordBtn.classList.remove('active');
    scoreEl.textContent=''; verseTextarea.value='';
    statusEl.textContent='ابدأ برفع تلاوة لرسم مسارها الصوتي';
    updateReadout(null,null);
    fileInput.value=''; wordRow.innerHTML=''; alignNote.style.display='none';
    render();
  });

  render();
})();
