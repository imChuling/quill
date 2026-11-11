/* Quill Piano Roll: web build of the desktop editor (ui/web/pianoroll.js). The backend is
   window.quillWebApi (webapi.js): a browser preview tone while recording and editing, the Quill
   render service for playback. Kept in step with the desktop file.
   Recording (on-screen, computer keyboard, external MIDI) echoes timing and velocity live; metronome click and count-in; BPM drives the grid.
   Editing: hover cursor hints, drag to move, drag the right edge to resize, right-click to label articulation or delete, double-click to delete.
   Playback goes through the real engine, the playhead follows.
   Viewport: time (horizontal) and pitch (vertical) both scroll; the wheel changes pitch, Shift or horizontal scroll changes time, eased throughout. */
(function () {
  function init() {
    const cv = document.getElementById('prCanvas');
    if (!cv) return false;
    const ctx = cv.getContext('2d');
    const SNAP = 0.25, KEYW = 42;
    const PITCH_LO = 21, PITCH_HI = 108;              // total scrollable range A0..C8
    const dpr = window.devicePixelRatio || 1;
    let bpm = 120;
    let VIEW_ROWS = 24;                               // number of visible pitch rows, adjusted by fit() to the canvas height
    let viewLen = 8;                                  // visible beats
    let BPB = 4;                                       // beats per bar (time signature 2/4, 3/4, 4/4, 6/4...)
    let viewStart = 0, viewStartTarget = 0;          // horizontal viewport start (beats): current/target
    let viewLo = 48, viewLoTarget = 48;              // vertical viewport bottom MIDI (float, eased): current/target
    let notes = [], drag = null, preview = -1;
    let playing = false, recording = false, recHeadBeat = -1, playBeat = 0;
    let t0 = 0, recT0 = 0, timers = [], renderRAF = null;

    const api = () => window.quillWebApi || null;
    // notify the main page that the session changed (the main page debounces and persists)
    const dirty = () => { if (window.quillSessionDirty) { try { window.quillSessionDirty(); } catch (e) {} } };
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const gridX = () => cv.width - KEYW * dpr;
    const rowH = () => cv.height / VIEW_ROWS;
    const topF = () => viewLo + VIEW_ROWS - 1;        // top visible MIDI (float)
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

    // centre the vertical viewport on the range of the notes
    function centerOnNotes() {
      if (!notes.length) { viewLoTarget = 48; return; }
      const lo = Math.min.apply(null, notes.map(n => n.midi)), hi = Math.max.apply(null, notes.map(n => n.midi));
      viewLoTarget = clamp(Math.round((lo + hi) / 2 - VIEW_ROWS / 2), PITCH_LO, maxViewLo());
    }
    // horizontally follow the play/record head: only set the target, the render loop eases toward it
    function followHead(beat) {
      const t = beat - viewLen * 0.6;
      if (t > viewStartTarget) viewStartTarget = Math.max(0, t);
    }

    // ---- Render loop: playhead, viewport easing, redraw ----
    function renderLoop() {
      if (playing) { playBeat = (performance.now() - t0) / 1000 / (60 / bpm); if (!isFinite(playBeat)) { /* waiting for the render */ } else if (playBeat > lastEnd() + 0.12) { if (loopOn && notes.length) { play(); return; } stop(); } else followHead(playBeat); }
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
      VIEW_ROWS = clamp(Math.round((r.height) / 9), 16, 30);   // row height ~9px
      viewLo = clamp(viewLo, PITCH_LO, maxViewLo()); viewLoTarget = clamp(viewLoTarget, PITCH_LO, maxViewLo());
      draw();
    }

    // keyboard column: white keys continuous, black keys short and inset, colours match the bottom keyboard
    function drawKeyboard() {
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
      for (let m = m0; m <= m1; m++) {                          // row background, black-key rows slightly darker
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
      for (let m = m0; m <= m1; m++) if (m % 12 === 0) {        // horizontal C reference lines
        const y = yOfMidi(m); ctx.strokeStyle = 'rgba(176,141,79,.12)'; ctx.lineWidth = dpr;
        ctx.beginPath(); ctx.moveTo(kw, y); ctx.lineTo(W, y); ctx.stroke();
      }
      notes.forEach(n => {
        if (n.start + n.len < viewStart || n.start > viewStart + viewLen || n.midi < viewLo - 1 || n.midi > topF() + 1) return;
        const x = xOfBeat(n.start), y = yOfMidi(n.midi), w = Math.max(2 * dpr, (n.len / viewLen) * gridX());
        ctx.fillStyle = velColor(n.vel); ctx.fillRect(x, y + dpr, w, rh - 2 * dpr);
        ctx.fillStyle = 'rgba(255,255,255,.18)'; ctx.fillRect(x, y + dpr, w, Math.max(dpr, rh * 0.13));
        // articulation marker: left colour bar, slide cyan, vibrato purple, legato cream
        if (n.art === 'slide') { ctx.fillStyle = '#7ec8c9'; ctx.fillRect(x, y + dpr, 2.5 * dpr, rh - 2 * dpr); }
        else if (n.art === 'vibrato') { ctx.fillStyle = '#c9a0e8'; ctx.fillRect(x, y + dpr, 2.5 * dpr, rh - 2 * dpr); }
        else if (n.art === 'legato') { ctx.fillStyle = '#f0e6cc'; ctx.fillRect(x, y + dpr, 2.5 * dpr, rh - 2 * dpr); }
        ctx.strokeStyle = 'rgba(40,30,12,.75)'; ctx.lineWidth = dpr; ctx.strokeRect(x + .5 * dpr, y + dpr, w - dpr, rh - 2 * dpr);
      });
      if (recording === true && recHeadBeat >= 0) { const x = xOfBeat(recHeadBeat); ctx.strokeStyle = '#d9544e'; ctx.lineWidth = 1.5 * dpr; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      if (playing && isFinite(playBeat)) { const x = xOfBeat(playBeat); ctx.strokeStyle = '#f0e6cc'; ctx.lineWidth = 1.5 * dpr; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      ctx.restore();
      drawKeyboard();
    }

    // ---- Editing ----
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
      else { const n = { midi, start: snap(beat), len: SNAP, vel: 100 }; notes.push(n); drag = { mode: 'resize', n }; if (api()) { api().note_on(midi, 100); preview = midi; } }
      draw(); e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
      if (!drag) return;
      const { px, py } = evPos(e);
      const beat = beatOfX(px), midi = clamp(midiOfY(py), PITCH_LO, PITCH_HI);
      if (drag.mode === 'resize') drag.n.len = Math.max(SNAP, snap(beat - drag.n.start));
      else { drag.n.start = Math.max(0, snap(beat - drag.gb)); if (midi !== drag.n.midi) { drag.n.midi = midi; if (api()) { api().note_off(preview); api().note_on(midi, drag.n.vel); preview = midi; } } }
      draw();
    });
    window.addEventListener('mouseup', () => { if (preview >= 0 && api()) api().note_off(preview); preview = -1; if (drag) dirty(); drag = null; });
    cv.addEventListener('mousemove', e => {                     // hover cursor hint
      if (drag || recording) return;
      const { px, py } = evPos(e);
      if (px < KEYW * dpr) { cv.style.cursor = 'default'; return; }
      const hit = hitNote(beatOfX(px), midiOfY(py));
      cv.style.cursor = hit ? (nearEnd(hit, px) ? 'ew-resize' : 'move') : 'crosshair';
    });
    function delAt(e) { const { px, py } = evPos(e); const hit = hitNote(beatOfX(px), midiOfY(py)); if (hit) { notes = notes.filter(n => n !== hit); draw(); dirty(); } return !!hit; }
    cv.addEventListener('dblclick', delAt);

    // ---- Context menu: articulation label / delete ----
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
          draw(); dirty();
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
    cv.addEventListener('wheel', e => {                         // wheel: vertical pitch; Shift or horizontal scroll: time
      if (recording) return;
      const horiz = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (horiz) { const d = (e.deltaX || e.deltaY) / 45; viewStartTarget = clamp(viewStartTarget + d, 0, Math.max(0, timelineBeats() - viewLen * 0.5)); }
      else { viewLoTarget = clamp(viewLoTarget - e.deltaY / 40, PITCH_LO, maxViewLo()); }
      kick(); e.preventDefault();
    }, { passive: false });

    // ---- Playback options and metronome ----
    let metroOn = false, countinOn = false, loopOn = false;   // loop: restart from the beginning at the end
    let neuralInst = '';                              // '' = off; otherwise a neural timbre name (from the backend list_instruments)
    let instruments = ['violin', 'trumpet', 'guitar', 'glass'];   // default order before the backend returns its list
    const neuralMode = () => !!neuralInst;
    let behaviorSrc = '';                             // '' = follow instrument; otherwise a behaviour source name
    let trajGain = 1.0;                               // trajectory strength 0.3–2.0
    let vibSource = 'retrieval';                      // 'retrieval' / 'model'
    let vibRandom = 0.3;                              // vibrato randomness 0–1
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
      if (api() && api().click_on) { api().click_on(bpm, BPB); return; }   // sample-accurate engine click
      const ac = getAC(); if (!ac) return; stopMetro(); _clickBeat = fromBeat || 0; _nextClick = ac.currentTime + 0.06; _sched();   // browser fallback: WebAudio
    }
    function stopMetro() {
      if (api() && api().click_off) api().click_off();
      if (_schedTimer) clearTimeout(_schedTimer); _schedTimer = null;
    }

    // ---- Recording ----
    function recStart() {
      if (!api()) return;
      notes = []; viewStart = 0; viewStartTarget = 0; recHeadBeat = 0;
      const bd = 60 / bpm;
      const arm = () => { recording = true; recT0 = performance.now(); api().record_start(); pollRec(); kick(); };
      if (countinOn) { recording = 'lead'; startMetro(-BPB); kick(); setTimeout(() => { if (recording !== 'lead') return; if (!metroOn) stopMetro(); arm(); }, BPB * bd * 1000); }
      else { if (metroOn) startMetro(0); arm(); }
    }
    function pollRec() {
      if (recording !== true) return;
      api().record_poll().then(res => {
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
      api().record_stop().then(rec => {
        const bd = 60 / bpm;
        notes = (rec || []).map(n => ({ midi: n.midi, start: snap(n.start / bd), len: Math.max(SNAP, snap(n.len / bd)), vel: n.vel || 100, art: n.art || 'pluck' }));
        viewStartTarget = 0; centerOnNotes(); kick(); dirty();
      });
    }

    // ---- Playback ----
    function play() {
      stop(); if (!notes.length) return;
      playing = true; playBeat = 0; viewStart = 0; viewStartTarget = 0;
      const bd = 60 / bpm; t0 = performance.now();
      if (neuralMode() && api() && api().play_notes_neural) {  // neural timbre (DDSP): rendered on the server, the playhead starts once ready
        t0 = Infinity;
        const p = api().play_notes_neural(window.quillPiano.getNotes(), bpm, neuralInst, behaviorSrc || null, trajGain, vibSource, vibRandom);
        if (p && p.then) p.then(r => { if (!playing) return; if (!r || !r.ok) { stop(); return; } t0 = performance.now(); playBeat = 0; kick(); }).catch(() => stop());
      } else if (api() && api().play_notes) {               // hand the whole thing to sample-accurate engine scheduling, notes and click share a clock
        api().play_notes(window.quillPiano.getNotes(), bpm, metroOn, BPB);
      } else {                                                             // browser fallback: setTimeout triggers each note
        if (metroOn) startMetro(0);
        const seq = notes.slice().sort((a, b) => a.start - b.start);
        seq.forEach((n, i) => {
          const prev = i > 0 ? seq[i - 1].midi : -1;
          timers.push(setTimeout(() => api() && api().note_on(n.midi, n.vel, n.art || 'pluck', prev), n.start * bd * 1000));
          timers.push(setTimeout(() => api() && api().note_off(n.midi), (n.start + n.len) * bd * 1000));
        });
      }
      kick();
    }
    function stop() {
      playing = false; timers.forEach(clearTimeout); timers = [];
      stopMetro();
      if (api() && api().stop_playback) api().stop_playback();   // cancel the engine plan + release notes
      else if (api()) notes.forEach(n => api().note_off(n.midi));
      kick();
    }

    // ---- Toolbar and external interface ----
    const byId = id => document.getElementById(id);
    function syncRecBtn() { const b = byId('prRec'); if (b) b.style.color = recording ? '#d9544e' : ''; }
    function toggleRec() { recording ? recStop() : recStart(); syncRecBtn(); }
    function stopAll() { if (recording) { recStop(); syncRecBtn(); } stop(); }
    if (byId('prRec')) byId('prRec').onclick = toggleRec;
    if (byId('prPlay')) byId('prPlay').onclick = play;
    if (byId('prStop')) byId('prStop').onclick = stopAll;
    if (byId('prClear')) byId('prClear').onclick = () => { stop(); notes = []; viewStartTarget = 0; centerOnNotes(); kick(); dirty(); };
    // called after the backend saves a timbre or finishes a bake
    window.quillRefreshInstruments = () => {
      if (api() && api().list_instruments)
        api().list_instruments().then(l => {
          if (l && l.length) instruments = l;
          if (neuralInst && neuralInst !== '' && instruments.indexOf(neuralInst) < 0) { neuralInst = ''; }
        }).catch(() => {});
    };
    window.quillPiano = {
      getNotes: () => notes.map(n => ({ midi: n.midi, start: +(n.start * 60 / bpm).toFixed(4), len: +(n.len * 60 / bpm).toFixed(4), vel: n.vel, art: n.art || 'pluck' })),
      setBPM: b => { bpm = +b || 120; draw(); },
      // convert notes in seconds to beats; used for session restore and loading example phrases
      setNotes: (list) => {
        stop(); const bd = 60 / bpm;
        notes = (list || []).filter(n => n && isFinite(+n.midi) && +n.midi >= PITCH_LO && +n.midi <= PITCH_HI && isFinite(+n.start) && +n.len > 0)
          .map(n => ({ midi: Math.round(+n.midi), start: Math.max(0, +n.start / bd), len: Math.max(SNAP, +n.len / bd), vel: clamp(Math.round(+n.vel || 100), 1, 127), art: n.art || 'pluck' }));
        viewStartTarget = 0; centerOnNotes(); kick();
      },
      getState: () => ({ bpm, notes: window.quillPiano.getNotes(), neural: neuralInst, behavior: behaviorSrc, traj_gain: trajGain, vib_source: vibSource, vib_random: vibRandom }),
      // session restore. Order: BPM first, then notes (beats depend on BPM), options last
      applyState: (st) => {
        if (!st) return;
        if (st.bpm) { bpm = clamp(+st.bpm, 40, 240); }
        if (Array.isArray(st.notes)) window.quillPiano.setNotes(st.notes);
        if (typeof st.neural === 'string') { neuralInst = (st.neural && instruments.indexOf(st.neural) >= 0) ? st.neural : (st.neural || ''); }
        if (typeof st.behavior === 'string') { behaviorSrc = st.behavior; }
        if (st.traj_gain != null) window.quillPiano.setTrajGain(st.traj_gain);
        if (st.vib_source) { vibSource = st.vib_source === 'model' ? 'model' : 'retrieval'; }
        if (st.vib_random != null) { vibRandom = clamp(+st.vib_random, 0, 1); }
        draw();
      },
      clear: () => { stop(); notes = []; viewStartTarget = 0; centerOnNotes(); kick(); },
      record: toggleRec, play: play, stop: stopAll, isRecording: () => recording,
      setMetro: v => { metroOn = !!v; if (!metroOn && !recording && !playing) stopMetro(); },
      setLoop: v => { loopOn = !!v; },
      setCountin: v => { countinOn = !!v; },
      setTimeSig: n => { BPB = Math.max(2, Math.min(12, +n || 4)); draw(); },
      setNeural: v => { neuralInst = (v === true ? 'violin' : v || ''); },
      setBehavior: v => { behaviorSrc = v || ''; },
      setTrajGain: v => { trajGain = Math.max(0.3, Math.min(2.0, +v || 1.0)); }
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
