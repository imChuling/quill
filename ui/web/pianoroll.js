/* Quill Piano Roll v4 — mini DAW 式可编辑 MIDI 卷帘。
   ① 录制(屏幕/电脑键盘/外接 MIDI)实时回显 + 时间/力度;节拍器 click + 预备拍 count-in;BPM 驱动网格。
   ② 增删/编辑:悬停提示光标,拖移动、拖右端改长度、右键/双击删除。
   ③ 回放经真实引擎,播放头跟随。
   ④ 视口:时间(横)+ 音高(纵)双轴可滚动(滚轮纵向音高、Shift/横向滚动时间),全程缓动丝滑。 */
(function () {
  function init() {
    const cv = document.getElementById('prCanvas');
    if (!cv) return false;
    const ctx = cv.getContext('2d');
    const SNAP = 0.25, KEYW = 42;
    const PITCH_LO = 21, PITCH_HI = 108;              // 总音域 A0..C8(可滚动到)
    const dpr = window.devicePixelRatio || 1;
    let bpm = 120;
    let VIEW_ROWS = 24;                               // 可见音高行数(随画布高度在 fit() 里调)
    let viewLen = 8;                                  // 可见拍数
    let BPB = 4;                                       // 每小节拍数(拍号:2/4、3/4、4/4、6/4…)
    let viewStart = 0, viewStartTarget = 0;          // 横向(拍)当前/目标
    let viewLo = 48, viewLoTarget = 48;              // 纵向:底部可见 MIDI(浮点=平滑)当前/目标
    let notes = [], drag = null, preview = -1;
    let playing = false, recording = false, recHeadBeat = -1, playBeat = 0;
    let t0 = 0, recT0 = 0, timers = [], renderRAF = null;

    const api = () => window.pywebview && window.pywebview.api;
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const gridX = () => cv.width - KEYW * dpr;
    const rowH = () => cv.height / VIEW_ROWS;
    const topF = () => viewLo + VIEW_ROWS - 1;        // 顶部可见 MIDI(浮点)
    const yOfMidi = (m) => (topF() - m) * rowH();
    const midiOfY = (py) => Math.round(topF()) - Math.floor(py / rowH());
    const xOfBeat = (b) => KEYW * dpr + ((b - viewStart) / viewLen) * gridX();
    const beatOfX = (px) => viewStart + ((px - KEYW * dpr) / gridX()) * viewLen;
    const isBlack = (m) => [1, 3, 6, 8, 10].includes(((m % 12) + 12) % 12);
    const snap = (b) => Math.round(b / SNAP) * SNAP;
    const velColor = (v) => { const t = clamp(v / 127, 0, 1); return `hsl(${44 - t * 16},${52 + t * 32}%,${50 + t * 16}%)`; };
    const lastEnd = () => notes.length ? Math.max.apply(null, notes.map(n => n.start + n.len)) : 0;
    const timelineBeats = () => Math.max(viewLen, lastEnd() + 4, recHeadBeat + 2);
    const maxViewLo = () => PITCH_HI - VIEW_ROWS + 1;

    function centerOnNotes() {                         // 录后把纵向视口居中到音符
      if (!notes.length) { viewLoTarget = 48; return; }
      const lo = Math.min.apply(null, notes.map(n => n.midi)), hi = Math.max.apply(null, notes.map(n => n.midi));
      viewLoTarget = clamp(Math.round((lo + hi) / 2 - VIEW_ROWS / 2), PITCH_LO, maxViewLo());
    }
    function followHead(beat) {                        // 横向跟随播放/录制头(设目标,缓动)
      const t = beat - viewLen * 0.6;
      if (t > viewStartTarget) viewStartTarget = Math.max(0, t);
    }

    // ---- 单一缓动渲染循环(播放头 + 视口平滑 + 重绘)---- //
    function renderLoop() {
      if (playing) { playBeat = (performance.now() - t0) / 1000 / (60 / bpm); if (playBeat > lastEnd() + 0.12) stop(); else followHead(playBeat); }
      if (recording === true) { recHeadBeat = (performance.now() - recT0) / 1000 / (60 / bpm); followHead(recHeadBeat); }
      let moving = false;
      viewStart += (viewStartTarget - viewStart) * 0.25; if (Math.abs(viewStartTarget - viewStart) < 0.003) viewStart = viewStartTarget; else moving = true;
      viewLo += (viewLoTarget - viewLo) * 0.25; if (Math.abs(viewLoTarget - viewLo) < 0.01) viewLo = viewLoTarget; else moving = true;
      draw();
      if (playing || recording === true || moving) renderRAF = requestAnimationFrame(renderLoop); else renderRAF = null;
    }
    function kick() { if (!renderRAF) renderRAF = requestAnimationFrame(renderLoop); }

    function fit() {
      const r = cv.getBoundingClientRect();
      cv.width = Math.max(1, r.width * dpr); cv.height = Math.max(1, r.height * dpr);
      VIEW_ROWS = clamp(Math.round((r.height) / 9), 16, 30);   // 行高 ~9px
      viewLo = clamp(viewLo, PITCH_LO, maxViewLo()); viewLoTarget = clamp(viewLoTarget, PITCH_LO, maxViewLo());
      draw();
    }

    function drawKeyboard() {                          // 真钢琴排布:白键长连续、黑键短潜入(配色同底部键盘)
      const kw = KEYW * dpr, H = cv.height, rh = rowH();
      const wg = ctx.createLinearGradient(0, 0, kw, 0);
      wg.addColorStop(0, '#f4eee4'); wg.addColorStop(1, '#e2d8c0');
      ctx.fillStyle = wg; ctx.fillRect(0, 0, kw, H);
      const m0 = Math.floor(viewLo) - 1, m1 = Math.ceil(topF()) + 1;
      ctx.strokeStyle = 'rgba(120,100,60,.32)'; ctx.lineWidth = dpr;
      for (let m = m0; m <= m1; m++) {
        const pc = ((m % 12) + 12) % 12, y = yOfMidi(m);
        let ly = null;
        if (pc === 0 || pc === 5) ly = y + rh; else if (isBlack(m)) ly = y + rh / 2;
        if (ly !== null) { ctx.beginPath(); ctx.moveTo(0, ly); ctx.lineTo(kw, ly); ctx.stroke(); }
      }
      const fl = ctx.createLinearGradient(kw - 7 * dpr, 0, kw, 0);
      fl.addColorStop(0, 'rgba(120,100,60,0)'); fl.addColorStop(1, 'rgba(120,100,60,.20)');
      ctx.fillStyle = fl; ctx.fillRect(kw - 7 * dpr, 0, 7 * dpr, H);
      const bw = kw * 0.6, bg = ctx.createLinearGradient(0, 0, bw, 0);
      bg.addColorStop(0, '#2a2418'); bg.addColorStop(1, '#141009');
      for (let m = m0; m <= m1; m++) if (isBlack(m)) {
        const y = yOfMidi(m) + rh * 0.06, hh = rh * 0.88;
        ctx.fillStyle = bg; ctx.fillRect(0, y, bw, hh);
        ctx.fillStyle = 'rgba(255,246,224,.12)'; ctx.fillRect(0, y, bw, Math.max(dpr, hh * 0.22));
        ctx.fillStyle = 'rgba(0,0,0,.5)'; ctx.fillRect(bw - dpr, y, dpr, hh);
      }
      ctx.strokeStyle = 'rgba(20,16,10,.5)'; ctx.lineWidth = dpr;
      ctx.beginPath(); ctx.moveTo(kw, 0); ctx.lineTo(kw, H); ctx.stroke();
      for (let m = m0; m <= m1; m++) if (m % 12 === 0) {
        const y = yOfMidi(m);
        ctx.fillStyle = '#7a5e26'; ctx.font = '600 ' + (8 * dpr) + 'px ui-sans-serif,system-ui';
        ctx.fillText('C' + (m / 12 - 1), kw * 0.66, y + rh * 0.74);
      }
    }

    function draw() {
      const W = cv.width, H = cv.height, kw = KEYW * dpr, rh = rowH();
      ctx.clearRect(0, 0, W, H);
      ctx.save(); ctx.beginPath(); ctx.rect(kw, 0, W - kw, H); ctx.clip();
      const m0 = Math.floor(viewLo) - 1, m1 = Math.ceil(topF()) + 1;
      for (let m = m0; m <= m1; m++) {                          // 行底色(黑键行略深)
        const y = yOfMidi(m);
        ctx.fillStyle = isBlack(m) ? 'rgba(0,0,0,.20)' : 'rgba(255,246,224,.03)';
        ctx.fillRect(kw, y, W - kw, rh);
      }
      const bFrom = Math.floor(viewStart), bTo = Math.ceil(viewStart + viewLen);
      ctx.strokeStyle = 'rgba(176,141,79,.07)'; ctx.lineWidth = dpr;
      for (let b = bFrom; b <= bTo; b += SNAP) { if (b % 1 === 0) continue; const x = xOfBeat(b); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      for (let b = bFrom; b <= bTo; b++) {
        const x = xOfBeat(b), bar = ((((b % BPB) + BPB) % BPB) === 0);
        ctx.strokeStyle = bar ? 'rgba(176,141,79,.5)' : 'rgba(176,141,79,.18)';
        ctx.lineWidth = bar ? 1.3 * dpr : dpr;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        if (bar && b >= 0) { ctx.fillStyle = 'rgba(176,141,79,.5)'; ctx.font = '600 ' + (8 * dpr) + 'px ui-sans-serif'; ctx.fillText('' + (b / BPB + 1), x + 3 * dpr, 10 * dpr); }
      }
      for (let m = m0; m <= m1; m++) if (m % 12 === 0) {        // C 水平参考线
        const y = yOfMidi(m); ctx.strokeStyle = 'rgba(176,141,79,.12)'; ctx.lineWidth = dpr;
        ctx.beginPath(); ctx.moveTo(kw, y); ctx.lineTo(W, y); ctx.stroke();
      }
      notes.forEach(n => {
        if (n.start + n.len < viewStart || n.start > viewStart + viewLen || n.midi < viewLo - 1 || n.midi > topF() + 1) return;
        const x = xOfBeat(n.start), y = yOfMidi(n.midi), w = Math.max(2 * dpr, (n.len / viewLen) * gridX());
        ctx.fillStyle = velColor(n.vel); ctx.fillRect(x, y + dpr, w, rh - 2 * dpr);
        ctx.fillStyle = 'rgba(255,255,255,.18)'; ctx.fillRect(x, y + dpr, w, Math.max(dpr, rh * 0.13));
        if (n.art === 'slide') { ctx.fillStyle = '#7ec8c9'; ctx.fillRect(x, y + dpr, 2.5 * dpr, rh - 2 * dpr); }        // 滑音=青条
        else if (n.art === 'vibrato') { ctx.fillStyle = '#c9a0e8'; ctx.fillRect(x, y + dpr, 2.5 * dpr, rh - 2 * dpr); } // 揉弦=紫条
        else if (n.art === 'legato') { ctx.fillStyle = '#f0e6cc'; ctx.fillRect(x, y + dpr, 2.5 * dpr, rh - 2 * dpr); }  // 连音=奶白条
        ctx.strokeStyle = 'rgba(40,30,12,.75)'; ctx.lineWidth = dpr; ctx.strokeRect(x + .5 * dpr, y + dpr, w - dpr, rh - 2 * dpr);
      });
      if (recording === true && recHeadBeat >= 0) { const x = xOfBeat(recHeadBeat); ctx.strokeStyle = '#d9544e'; ctx.lineWidth = 1.5 * dpr; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      if (playing) { const x = xOfBeat(playBeat); ctx.strokeStyle = '#f0e6cc'; ctx.lineWidth = 1.5 * dpr; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      ctx.restore();
      drawKeyboard();
    }

    // ---- 编辑 ---- //
    function evPos(e) { const r = cv.getBoundingClientRect(); return { px: (e.clientX - r.left) * dpr, py: (e.clientY - r.top) * dpr }; }
    function hitNote(beat, midi) { return notes.find(n => n.midi === midi && beat >= n.start && beat <= n.start + n.len); }
    function nearEnd(n, px) { return Math.abs(px - xOfBeat(n.start + n.len)) < 7 * dpr; }
    cv.addEventListener('mousedown', e => {
      if (recording) return;
      const { px, py } = evPos(e);
      if (px < KEYW * dpr) return;
      const beat = beatOfX(px), midi = midiOfY(py);
      if (midi < PITCH_LO || midi > PITCH_HI || beat < 0) return;
      const hit = hitNote(beat, midi);
      if (hit) drag = nearEnd(hit, px) ? { mode: 'resize', n: hit } : { mode: 'move', n: hit, gb: beat - hit.start };
      else { const n = { midi, start: snap(beat), len: SNAP, vel: 100 }; notes.push(n); drag = { mode: 'resize', n }; if (api()) { window.pywebview.api.note_on(midi, 100); preview = midi; } }
      draw(); e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
      if (!drag) return;
      const { px, py } = evPos(e);
      const beat = beatOfX(px), midi = clamp(midiOfY(py), PITCH_LO, PITCH_HI);
      if (drag.mode === 'resize') drag.n.len = Math.max(SNAP, snap(beat - drag.n.start));
      else { drag.n.start = Math.max(0, snap(beat - drag.gb)); if (midi !== drag.n.midi) { drag.n.midi = midi; if (api()) { window.pywebview.api.note_off(preview); window.pywebview.api.note_on(midi, drag.n.vel); preview = midi; } } }
      draw();
    });
    window.addEventListener('mouseup', () => { if (preview >= 0 && api()) window.pywebview.api.note_off(preview); preview = -1; drag = null; });
    cv.addEventListener('mousemove', e => {                     // 悬停光标提示
      if (drag || recording) return;
      const { px, py } = evPos(e);
      if (px < KEYW * dpr) { cv.style.cursor = 'default'; return; }
      const hit = hitNote(beatOfX(px), midiOfY(py));
      cv.style.cursor = hit ? (nearEnd(hit, px) ? 'ew-resize' : 'move') : 'crosshair';
    });
    function delAt(e) { const { px, py } = evPos(e); const hit = hitNote(beatOfX(px), midiOfY(py)); if (hit) { notes = notes.filter(n => n !== hit); draw(); } return !!hit; }
    cv.addEventListener('dblclick', delAt);

    // ---- 右键菜单:演奏法标注 / 删除 ---- //
    let ctxMenu = null;
    function closeCtx() { if (ctxMenu) { ctxMenu.remove(); ctxMenu = null; } }
    function showArtMenu(e, note) {
      e.preventDefault(); closeCtx();
      const m = document.createElement('div');
      m.style.cssText = 'position:fixed;z-index:9999;background:#1e1c16;border:1px solid rgba(176,141,79,.4);border-radius:6px;padding:4px 0;font-family:"Spline Sans Mono",monospace;font-size:11px;color:#e8e0cd;box-shadow:0 6px 20px rgba(0,0,0,.6);min-width:120px;';
      const items = [
        { label: 'Plain', art: 'pluck', color: '#cdc3a8' },
        { label: '~ Vibrato', art: 'vibrato', color: '#c9a0e8' },
        { label: '⤡ Slide', art: 'slide', color: '#7ec8c9' },
        { label: '⌇ Legato', art: 'legato', color: '#f0e6cc' },
        null,
        { label: '✕ Delete', action: 'del', color: '#d9544e' },
      ];
      items.forEach(it => {
        if (!it) { const hr = document.createElement('div'); hr.style.cssText = 'height:1px;margin:3px 8px;background:rgba(176,141,79,.2);'; m.appendChild(hr); return; }
        const d = document.createElement('div');
        d.textContent = it.label;
        const active = it.art && (note.art || 'pluck') === it.art;
        d.style.cssText = 'padding:5px 14px;cursor:pointer;color:' + it.color + ';' + (active ? 'background:rgba(176,141,79,.15);' : '');
        d.onmouseenter = () => { d.style.background = 'rgba(176,141,79,.18)'; };
        d.onmouseleave = () => { d.style.background = active ? 'rgba(176,141,79,.15)' : ''; };
        d.onclick = () => {
          closeCtx();
          if (it.action === 'del') { notes = notes.filter(n => n !== note); }
          else { note.art = it.art; }
          draw();
        };
        m.appendChild(d);
      });
      m.style.left = Math.min(e.clientX, window.innerWidth - 140) + 'px';
      m.style.top = Math.min(e.clientY, window.innerHeight - 180) + 'px';
      document.body.appendChild(m);
      ctxMenu = m;
    }
    window.addEventListener('mousedown', e => { if (ctxMenu && !ctxMenu.contains(e.target)) closeCtx(); });
    cv.addEventListener('contextmenu', e => {
      e.preventDefault();
      const { px, py } = evPos(e);
      const hit = hitNote(beatOfX(px), midiOfY(py));
      if (hit) showArtMenu(e, hit);
      else delAt(e);
    });
    cv.addEventListener('wheel', e => {                         // 滚轮:纵向音高;Shift/横向 → 时间
      if (recording) return;
      const horiz = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (horiz) { const d = (e.deltaX || e.deltaY) / 45; viewStartTarget = clamp(viewStartTarget + d, 0, Math.max(0, timelineBeats() - viewLen * 0.5)); }
      else { viewLoTarget = clamp(viewLoTarget - e.deltaY / 40, PITCH_LO, maxViewLo()); }
      kick(); e.preventDefault();
    }, { passive: false });

    // ---- 节拍器(Web Audio,同 BPM;count-in 预备拍)---- //
    let metroOn = false, countinOn = false;
    let neuralInst = '';                              // '' = 关;'violin'/'guitar' = 神经音色
    const neuralMode = () => !!neuralInst;
    let behaviorSrc = '';                             // '' = 同 instrument;否则=行为源名称
    let trajGain = 1.0;                               // 轨迹强度(0.3–2.0)
    let vibSource = 'retrieval';                      // 'retrieval' / 'model'
    let vibRandom = 0.3;                              // 颤音随机性 0–1
    let _ac = null, _schedTimer = null, _nextClick = 0, _clickBeat = 0;
    function getAC() { if (!_ac) { try { _ac = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { return null; } } if (_ac.state === 'suspended') _ac.resume(); return _ac; }
    function tick(accent, when) {
      const ac = getAC(); if (!ac) return;
      const o = ac.createOscillator(), g = ac.createGain();
      o.frequency.value = accent ? 2000 : 1250;
      g.gain.setValueAtTime(0.0001, when); g.gain.exponentialRampToValueAtTime(accent ? 0.5 : 0.3, when + 0.001); g.gain.exponentialRampToValueAtTime(0.0001, when + 0.045);
      o.connect(g).connect(ac.destination); o.start(when); o.stop(when + 0.05);
    }
    function _sched() { const ac = getAC(); if (!ac) return; const bd = 60 / bpm; while (_nextClick < ac.currentTime + 0.12) { tick((((_clickBeat % BPB) + BPB) % BPB) === 0, _nextClick); _nextClick += bd; _clickBeat++; } _schedTimer = setTimeout(_sched, 25); }
    function startMetro(fromBeat) {
      if (api() && window.pywebview.api.click_on) { window.pywebview.api.click_on(bpm, BPB); return; }   // app:引擎采样级 click
      const ac = getAC(); if (!ac) return; stopMetro(); _clickBeat = fromBeat || 0; _nextClick = ac.currentTime + 0.06; _sched();   // 浏览器降级:WebAudio
    }
    function stopMetro() {
      if (api() && window.pywebview.api.click_off) window.pywebview.api.click_off();
      if (_schedTimer) clearTimeout(_schedTimer); _schedTimer = null;
    }

    // ---- 录制 ---- //
    function recStart() {
      if (!api()) return;
      notes = []; viewStart = 0; viewStartTarget = 0; recHeadBeat = 0;
      const bd = 60 / bpm;
      const arm = () => { recording = true; recT0 = performance.now(); window.pywebview.api.record_start(); pollRec(); kick(); };
      if (countinOn) { recording = 'lead'; startMetro(-BPB); kick(); setTimeout(() => { if (recording !== 'lead') return; if (!metroOn) stopMetro(); arm(); }, BPB * bd * 1000); }
      else { if (metroOn) startMetro(0); arm(); }
    }
    function pollRec() {
      if (recording !== true) return;
      window.pywebview.api.record_poll().then(res => {
        if (recording !== true) return;
        const bd = 60 / bpm;
        notes = (res.notes || []).map(n => ({ midi: n.midi, start: n.start / bd, len: Math.max(0.1, n.len / bd), vel: n.vel, art: n.art || 'pluck' }));
        setTimeout(pollRec, 55);
      }).catch(() => { if (recording === true) setTimeout(pollRec, 120); });
    }
    function recStop() {
      const wasLead = (recording === 'lead');
      recording = false; recHeadBeat = -1; stopMetro();
      if (!api() || wasLead) { draw(); return; }
      window.pywebview.api.record_stop().then(rec => {
        const bd = 60 / bpm;
        notes = (rec || []).map(n => ({ midi: n.midi, start: snap(n.start / bd), len: Math.max(SNAP, snap(n.len / bd)), vel: n.vel || 100, art: n.art || 'pluck' }));
        viewStartTarget = 0; centerOnNotes(); kick();
      });
    }

    // ---- 回放 ---- //
    function play() {
      stop(); if (!notes.length) return;
      playing = true; playBeat = 0; viewStart = 0; viewStartTarget = 0;
      const bd = 60 / bpm; t0 = performance.now();
      if (neuralMode() && api() && window.pywebview.api.play_notes_neural) {  // 神经音色(DDSP)
        window.pywebview.api.play_notes_neural(
          window.quillPiano.getNotes(), bpm, neuralInst,
          behaviorSrc || null, trajGain, vibSource, vibRandom);
      } else if (api() && window.pywebview.api.play_notes) {               // app:整段交给引擎采样级调度(音符+click 同一时钟)
        window.pywebview.api.play_notes(window.quillPiano.getNotes(), bpm, metroOn, BPB);
      } else {                                                             // 浏览器降级:setTimeout 逐音
        if (metroOn) startMetro(0);
        const seq = notes.slice().sort((a, b) => a.start - b.start);
        seq.forEach((n, i) => {
          const prev = i > 0 ? seq[i - 1].midi : -1;
          timers.push(setTimeout(() => api() && window.pywebview.api.note_on(n.midi, n.vel, n.art || 'pluck', prev), n.start * bd * 1000));
          timers.push(setTimeout(() => api() && window.pywebview.api.note_off(n.midi), (n.start + n.len) * bd * 1000));
        });
      }
      kick();
    }
    function stop() {
      playing = false; timers.forEach(clearTimeout); timers = [];
      stopMetro();
      if (api() && window.pywebview.api.stop_playback) window.pywebview.api.stop_playback();   // 取消引擎 plan + 收音
      else if (api()) notes.forEach(n => window.pywebview.api.note_off(n.midi));
      kick();
    }

    // ---- 工具条 + 对外接口 ---- //
    const byId = id => document.getElementById(id);
    function syncRecBtn() { const b = byId('prRec'); if (b) b.style.color = recording ? '#d9544e' : ''; }
    function toggleRec() { recording ? recStop() : recStart(); syncRecBtn(); }
    function stopAll() { if (recording) { recStop(); syncRecBtn(); } stop(); }
    if (byId('prRec')) byId('prRec').onclick = toggleRec;
    if (byId('prPlay')) byId('prPlay').onclick = play;
    if (byId('prStop')) byId('prStop').onclick = stopAll;
    if (byId('prClear')) byId('prClear').onclick = () => { stop(); notes = []; viewStartTarget = 0; centerOnNotes(); kick(); };
    function exportTake(fmt, btn) {                              // 卷帘导出:MIDI(互操作) / WAV(当前音色离线渲染)
      if (!api() || !window.pywebview.api.export_take) return;
      const orig = btn.textContent;
      btn.textContent = '…';
      window.pywebview.api.export_take(window.quillPiano.getNotes(), bpm, fmt).then(r => {
        btn.textContent = (r && r.ok) ? '✓ saved' : ((r && r.msg && r.msg !== 'cancelled') ? '✗' : orig);
        setTimeout(() => { btn.textContent = orig; }, 1600);
      }).catch(() => { btn.textContent = orig; });
    }
    if (byId('prMidi')) byId('prMidi').onclick = () => exportTake('midi', byId('prMidi'));
    if (byId('prWav')) byId('prWav').onclick = () => exportTake(neuralMode() ? 'wav_neural:' + neuralInst : 'wav', byId('prWav'));
    function syncNeuralBtn() {
      const b = byId('prNeural'); if (!b) return;
      b.textContent = neuralInst === 'violin' ? '🎻 Violin' : neuralInst === 'guitar' ? '🎸 Guitar' : '◌ Neural';
      b.style.color = neuralMode() ? '#7ec8c9' : '#cdc3a8';
      b.style.borderColor = neuralMode() ? 'rgba(126,200,201,.5)' : 'rgba(176,141,79,.3)';
      // 行为源控件随 Neural 模式显隐
      const bp = byId('prBehavior'); if (bp) bp.style.display = neuralMode() ? '' : 'none';
      const gp = byId('prGainWrap'); if (gp) gp.style.display = neuralMode() ? 'flex' : 'none';
    }
    if (byId('prNeural')) {
      byId('prNeural').onclick = () => {
        neuralInst = neuralInst === '' ? 'violin' : neuralInst === 'violin' ? 'guitar' : '';
        syncNeuralBtn();
      };
      syncNeuralBtn();
    }

    // ---- 行为源选择器 ---- //
    let _behaviorList = [];
    function loadBehaviorList() {
      if (!api() || !window.pywebview.api.list_behavior_sources) return;
      window.pywebview.api.list_behavior_sources().then(list => {
        _behaviorList = list || [];
        syncBehaviorBtn();
      }).catch(() => {});
    }
    function syncBehaviorBtn() {
      const b = byId('prBehavior'); if (!b) return;
      const label = behaviorSrc || '= timbre';
      b.textContent = '♪ ' + label;
      b.style.color = behaviorSrc ? '#e8b45a' : '#cdc3a8';
      b.style.borderColor = behaviorSrc ? 'rgba(232,180,90,.45)' : 'rgba(176,141,79,.3)';
    }
    if (byId('prBehavior')) {
      byId('prBehavior').onclick = () => {
        // 循环:= timbre → violin → flute → cello → guitar → 自定义库们 → = timbre
        const all = ['', 'violin', 'flute', 'cello', 'guitar'];
        _behaviorList.forEach(s => { if (!s.builtin && all.indexOf(s.name) < 0) all.push(s.name); });
        const idx = all.indexOf(behaviorSrc);
        behaviorSrc = all[(idx + 1) % all.length];
        syncBehaviorBtn();
      };
      syncBehaviorBtn();
      loadBehaviorList();
    }

    // ---- traj_gain 滑块 ---- //
    const gainSlider = byId('prGain');
    const gainVal = byId('prGainVal');
    function syncGain() { if (gainVal) gainVal.textContent = trajGain.toFixed(1); }
    if (gainSlider) {
      gainSlider.value = trajGain;
      gainSlider.oninput = () => { trajGain = parseFloat(gainSlider.value); syncGain(); };
      syncGain();
    }

    // ---- 导入自定义行为库 ---- //
    if (byId('prImport')) {
      byId('prImport').onclick = () => {
        if (!api() || !window.pywebview.api.import_style_wav) return;
        const btn = byId('prImport'), orig = btn.textContent;
        btn.textContent = '…';
        window.pywebview.api.import_style_wav().then(r => {
          if (r && r.ok) {
            btn.textContent = '✓ ' + r.name;
            loadBehaviorList();
            behaviorSrc = r.name;
            syncBehaviorBtn();
          } else {
            btn.textContent = (r && r.msg && r.msg !== 'cancelled') ? '✗' : orig;
          }
          setTimeout(() => { btn.textContent = orig; }, 2000);
        }).catch(() => { btn.textContent = orig; });
      };
    }
    function sendToAudiotool(btn) {                              // SEND TO AUDIOTOOL:登记 take(后台渲染),companion 页负责上传+插入
      if (!api() || !window.pywebview.api.stage_take) return;
      const orig = btn.textContent;
      btn.textContent = '…';
      window.pywebview.api.stage_take(window.quillPiano.getNotes(), bpm).then(r => {
        btn.textContent = (r && r.ok) ? '✓ staged' : '✗';
        if (r && !r.ok && r.msg) console.warn('stage_take:', r.msg);
        setTimeout(() => { btn.textContent = orig; }, 1800);
      }).catch(() => { btn.textContent = orig; });
    }
    if (byId('prSend')) byId('prSend').onclick = () => sendToAudiotool(byId('prSend'));

    window.quillPiano = {
      getNotes: () => notes.map(n => ({ midi: n.midi, start: +(n.start * 60 / bpm).toFixed(4), len: +(n.len * 60 / bpm).toFixed(4), vel: n.vel, art: n.art || 'pluck' })),
      setBPM: b => { bpm = +b || 120; draw(); },
      clear: () => { stop(); notes = []; viewStartTarget = 0; centerOnNotes(); kick(); },
      record: toggleRec, play: play, stop: stopAll, isRecording: () => recording,
      setMetro: v => { metroOn = !!v; if (!metroOn && !recording && !playing) stopMetro(); },
      setCountin: v => { countinOn = !!v; },
      setTimeSig: n => { BPB = Math.max(2, Math.min(12, +n || 4)); draw(); },
      setNeural: v => { neuralInst = (v === true ? 'violin' : v || ''); syncNeuralBtn(); },
      isNeural: () => neuralInst,
      setBehavior: v => { behaviorSrc = v || ''; syncBehaviorBtn(); },
      getBehavior: () => behaviorSrc,
      setTrajGain: v => { trajGain = Math.max(0.3, Math.min(2.0, +v || 1.0)); if (gainSlider) gainSlider.value = trajGain; syncGain(); },
      getTrajGain: () => trajGain,
      setVibSource: v => { vibSource = v === 'model' ? 'model' : 'retrieval'; },
      setVibRandom: v => { vibRandom = Math.max(0, Math.min(1, +v || 0.3)); },
      refreshBehaviors: loadBehaviorList
    };

    window.addEventListener('resize', fit);
    if (window.ResizeObserver) new ResizeObserver(fit).observe(cv);
    fit();
    setTimeout(fit, 250); setTimeout(fit, 800); setTimeout(fit, 1600);
    return true;
  }
  function sized() { const cv = document.getElementById('prCanvas'); if (!cv) return false; const r = cv.getBoundingClientRect(); return r.width > 4 && r.height > 4; }
  let tries = 0;
  const iv = setInterval(() => { if ((sized() && init()) || ++tries > 150) clearInterval(iv); }, 80);
})();
