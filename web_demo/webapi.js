/* window.quillWebApi: the piano roll's backend in the browser.
   Mirrors the subset of the desktop Api the roll calls:
     note_on / note_off          preview tone (Web Audio) + recording log + gesture label
     record_start / record_poll / record_stop
     play_notes_neural           POST /render on the Quill render service, then play the WAV
     stop_playback, list_instruments, list_behavior_sources
   Plus input: on-screen keys, computer keyboard (a w s e d f t g y h u j k o l), Web MIDI. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let ac = null, bus = null;
  function getAC() {
    if (!ac) {
      ac = new (window.AudioContext || window.webkitAudioContext)();
      bus = ac.createGain(); bus.gain.value = 0.6;
      const comp = ac.createDynamicsCompressor(); comp.threshold.value = -14; comp.ratio.value = 4;
      bus.connect(comp).connect(ac.destination);
    }
    if (ac.state === 'suspended') ac.resume();
    return ac;
  }

  // ---------------- preview tone (light additive pad, polyphonic) ----------------
  const voices = {}; const held = [];
  function previewOn(midi, vel) {
    const a = getAC(), now = a.currentTime, f = 440 * Math.pow(2, (midi - 69) / 12);
    const o1 = a.createOscillator(), o2 = a.createOscillator(); o1.type = 'triangle'; o2.type = 'sine';
    o1.frequency.value = f; o2.frequency.value = f * 2.001;
    const g2 = a.createGain(); g2.gain.value = 0.18;
    const filt = a.createBiquadFilter(); filt.type = 'lowpass'; filt.frequency.value = 900 + (vel / 127) * 2400; filt.Q.value = 0.4;
    const g = a.createGain(); g.gain.value = 0;
    o1.connect(filt); o2.connect(g2).connect(filt); filt.connect(g).connect(bus);
    const peak = 0.22 * (0.35 + 0.65 * vel / 127);
    g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(peak, now + 0.012); g.gain.linearRampToValueAtTime(peak * 0.72, now + 0.16);
    o1.start(); o2.start();
    voices[midi] = { o1, o2, g };
  }
  function previewOff(midi) {
    const v = voices[midi]; if (!v) return; delete voices[midi];
    const now = getAC().currentTime;
    v.g.gain.cancelScheduledValues(now); v.g.gain.setValueAtTime(v.g.gain.value, now); v.g.gain.linearRampToValueAtTime(0, now + 0.14);
    v.o1.stop(now + 0.16); v.o2.stop(now + 0.16);
  }
  function allOff() { Object.keys(voices).forEach((m) => previewOff(+m)); held.length = 0; }
  // keyup never arrives if the tab loses focus while a key is down, so release everything.
  window.addEventListener('blur', () => { Object.keys(voices).forEach((m) => api.note_off(+m)); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) Object.keys(voices).forEach((m) => api.note_off(+m)); });

  // ---------------- recording ----------------
  let rec = null;                                   // { t0, notes:[], open:{midi:{start,vel,art}} }
  const nowS = () => performance.now() / 1000;
  // Same rule as runtime/gesture.py: overlapping keys give legato (within 2 semitones) or slide.
  function gestureLabel(midi) {
    if (!held.length) return 'pluck';
    const prev = held[held.length - 1];
    return Math.abs(midi - prev) <= 2 ? 'legato' : 'slide';
  }

  const api = {
    note_on(midi, vel, art, prev) {
      midi = +midi; vel = vel == null ? 100 : +vel;
      if (voices[midi]) return Promise.resolve('pluck');
      const label = (art && art !== 'auto') ? art : gestureLabel(midi);
      previewOn(midi, vel);
      if (rec) rec.open[midi] = { start: nowS() - rec.t0, vel, art: label };
      if (held.indexOf(midi) < 0) held.push(midi);
      if (window.quillKeysVisual) window.quillKeysVisual(midi, true);
      if (window.quillArtVisual) window.quillArtVisual(label);
      return Promise.resolve(label);
    },
    note_off(midi) {
      midi = +midi; previewOff(midi);
      const i = held.indexOf(midi); if (i >= 0) held.splice(i, 1);
      if (rec && rec.open[midi]) { const o = rec.open[midi]; delete rec.open[midi]; rec.notes.push({ midi, start: +o.start.toFixed(4), len: +Math.max(0.05, nowS() - rec.t0 - o.start).toFixed(4), vel: o.vel, art: o.art }); }
      if (window.quillKeysVisual) window.quillKeysVisual(midi, false);
      return Promise.resolve(true);
    },
    record_start() { rec = { t0: nowS(), notes: [], open: {} }; if (window.quillRecState) window.quillRecState(true); return Promise.resolve(true); },
    record_poll() {
      if (!rec) return Promise.resolve({ notes: [], t: 0 });
      const t = nowS() - rec.t0;
      const open = Object.keys(rec.open).map((m) => ({ midi: +m, start: rec.open[m].start, len: Math.max(0.05, t - rec.open[m].start), vel: rec.open[m].vel, art: rec.open[m].art }));
      return Promise.resolve({ notes: rec.notes.concat(open), t });
    },
    record_stop() {
      if (!rec) return Promise.resolve([]);
      const t = nowS() - rec.t0;
      Object.keys(rec.open).forEach((m) => { const o = rec.open[m]; rec.notes.push({ midi: +m, start: +o.start.toFixed(4), len: +Math.max(0.05, t - o.start).toFixed(4), vel: o.vel, art: o.art }); });
      const out = rec.notes.slice(); rec = null; allOff();
      if (window.quillRecState) window.quillRecState(false, out.length);
      return Promise.resolve(out);
    },

    // ---------------- render + play ----------------
    _status: null, _lastTake: null, _lastKey: null, _busy: false,
    // _gen is bumped by stop_playback: a render that comes back after a Stop is kept but not played.
    _gen: 0,
    MAX_S: 30,
    onStatus(fn) { this._status = fn; },
    status(msg, err) { if (this._status) this._status(msg, err); },
    play_notes_neural(notes, bpm, instrument, behavior, trajGain, vibSource, vibRandom) {
      const audio = $('result-audio');
      // A second click while rendering is ignored.
      if (this._busy) return Promise.resolve({ ok: false, msg: 'busy' });
      const end = notes.reduce((m, n) => Math.max(m, n.start + n.len), 0);
      if (end > this.MAX_S) { this.status(`That is ${end.toFixed(1)} s. Keep it under ${this.MAX_S} s for the demo server.`, true); return Promise.resolve({ ok: false, msg: 'too long' }); }
      if (window.quillAutoLabel) notes = window.quillAutoLabel(notes);
      // The performance panel (app.js) is the source of truth for the renderer's knobs; the roll's own
      // trajGain/vibSource/vibRandom are used only when the panel is absent.
      const perf = window.quillPerf || {};
      const sp = Object.assign({ artint: Math.max(0, Math.min(1, ((+trajGain || 1) - 0.3) / 1.7)), vibmodel: vibSource === 'model' ? 1 : 0, randomness: +vibRandom || 0.3 }, perf);
      const body = { notes, bpm: +bpm || 100, title: 'web demo', timbre: instrument || 'violin', behavior: behavior || '', synth_params: sp };
      const nVib = notes.filter((n) => n.art === 'vibrato').length, nSl = notes.filter((n) => n.art === 'slide').length;
      const key = JSON.stringify(body);
      // Same notes and settings (loop, repeated Play): replay without a round trip.
      if (key === this._lastKey && this._lastTake && audio.src) {
        audio.currentTime = 0;
        return audio.play().then(() => ({ ok: true, take_id: this._lastTake.take_id, cached: true })).catch((e) => { this._lastKey = null; this._lastTake = null; this.status('That render is gone from the server. Press Play to render again.', true); return { ok: false, msg: e.message }; });
      }
      this._busy = true; if (window.quillBusy) window.quillBusy(true);
      this.status('Rendering.');
      const t0 = performance.now(), gen = this._gen;
      return fetch('render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(async (r) => { if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`); return r.json(); })
        .then((j) => {
          const url = `takes/${j.take_id}/audio.wav`;
          this._lastTake = { url, take_id: j.take_id, timbre: body.timbre, behavior: body.behavior }; this._lastKey = key;
          audio.src = url + `?t=${Date.now()}`; audio.currentTime = 0;
          const dl = $('dl-wav'); if (dl) { dl.href = url; dl.download = `quill_${body.timbre}_x_${body.behavior || body.timbre}.wav`; dl.style.display = ''; }
          this.status(`${body.timbre} timbre · ${body.behavior || body.timbre} library · ${notes.length} notes${nVib ? `, ${nVib} vibrato` : ''}${nSl ? `, ${nSl} slide` : ''} · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
          if (window.quillRenderDone) window.quillRenderDone();
          // Stop was pressed while rendering: the take is ready for the next Play, nothing plays now.
          if (gen !== this._gen) return { ok: false, msg: 'stopped' };
          return audio.play().then(() => ({ ok: true, take_id: j.take_id }));
        })
        .catch((e) => { const offline = /Failed to fetch|NetworkError|Load failed/.test(e.message); this.status(offline ? 'Render service is not running. The matrix above still plays.' : `Render failed: ${e.message}`, true); return { ok: false, msg: e.message }; })
        .finally(() => { this._busy = false; if (window.quillBusy) window.quillBusy(false); });
    },
    stop_playback() { this._gen++; const a = $('result-audio'); if (a) { a.pause(); } allOff(); return Promise.resolve(true); },
    _initAudio() { const a = $('result-audio'); if (!a || a._quillBound) return; a._quillBound = true; a.addEventListener('error', () => { this._lastKey = null; this._lastTake = null; this.status('That render is gone from the server. Press Play to render again.', true); }); },

    // Lists come from /status, filled by app.js.
    _instruments: [], _behaviors: [],
    list_instruments() { return Promise.resolve(this._instruments.slice()); },
    list_behavior_sources() { return Promise.resolve(this._behaviors.map((n) => ({ name: n }))); },
  };
  window.quillWebApi = api; api._initAudio();

  // ---------------- input: on-screen keys, computer keyboard, Web MIDI ----------------
  let octave = 0;                                   // computer-keyboard octave shift
  const KEYMAP = { a: 60, w: 61, s: 62, e: 63, d: 64, f: 65, t: 66, g: 67, y: 68, h: 69, u: 70, j: 71, k: 72, o: 73, l: 74 };
  const typing = (e) => { const t = e.target && e.target.tagName; return t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || (e.target && e.target.isContentEditable); };
  window.addEventListener('keydown', (e) => {
    if (typing(e) || e.metaKey || e.ctrlKey) return;
    if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); if (e.repeat) return; const p = window.quillPiano; if (!p) return; if (window.__quillPlaying) p.stop(); else p.play(); return; }
    if (e.key === 'Escape') { const p = window.quillPiano; if (p) p.stop(); return; }
    if (e.key === 'r' || e.key === 'R') { e.preventDefault(); if (e.repeat) return; const p = window.quillPiano; if (p) p.record(); return; }
    if (e.key === 'z') { octave = Math.max(-2, octave - 1); return; }
    if (e.key === 'x') { octave = Math.min(2, octave + 1); return; }
    const m = KEYMAP[e.key.toLowerCase()]; if (m == null) return;
    e.preventDefault(); if (e.repeat) return; api.note_on(m + 12 * octave, 100, 'auto', -1);
  });
  window.addEventListener('keyup', (e) => { if (typing(e)) return; const m = KEYMAP[e.key.toLowerCase()]; if (m == null) return; e.preventDefault(); api.note_off(m + 12 * octave); });

  function buildKeys() {
    const host = $('keys'); if (!host) return;
    host.innerHTML = '';
    const LO = 48, HI = 72;                         // C3..C5
    const whites = [], blacks = [];
    for (let m = LO; m <= HI; m++) ([1, 3, 6, 8, 10].includes(m % 12) ? blacks : whites).push(m);
    const ww = 100 / whites.length;
    whites.forEach((m, i) => { const k = document.createElement('button'); k.className = 'wk'; k.dataset.midi = m; k.style.left = (i * ww) + '%'; k.style.width = ww + '%'; k.setAttribute('aria-label', 'key ' + m); if (m % 12 === 0) k.textContent = 'C' + (m / 12 - 1); host.appendChild(k); });
    blacks.forEach((m) => { const idx = whites.filter((w) => w < m).length; const k = document.createElement('button'); k.className = 'bk'; k.dataset.midi = m; k.style.left = (idx * ww - ww * 0.3) + '%'; k.style.width = (ww * 0.6) + '%'; k.setAttribute('aria-label', 'key ' + m); host.appendChild(k); });
    const down = (e) => { const b = e.target.closest('button'); if (!b) return; e.preventDefault(); b.setPointerCapture && b.setPointerCapture(e.pointerId); api.note_on(+b.dataset.midi, 100, 'auto', -1); };
    const up = (e) => { const b = e.target.closest('button'); if (!b) return; api.note_off(+b.dataset.midi); };
    host.addEventListener('pointerdown', down); host.addEventListener('pointerup', up); host.addEventListener('pointercancel', up); host.addEventListener('pointerleave', (e) => { if (e.target.closest && e.target.closest('button')) up(e); }, true);
    window.quillKeysVisual = (midi, on) => { const k = host.querySelector(`[data-midi="${midi}"]`); if (k) k.classList.toggle('on', !!on); };
  }
  buildKeys();

  // The roll's editor (shared with the desktop) listens to mouse events; forward touches as mouse events so
  // phones can draw, drag and resize notes. Two fingers are left to the browser (pinch/scroll).
  (function bridgeTouch() {
    const cv = $('prCanvas'); if (!cv) return;
    cv.style.touchAction = 'none';
    const fire = (type, t, target) => { const ev = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: t.clientX, clientY: t.clientY, button: 0, buttons: type === 'mouseup' ? 0 : 1 }); (target || window).dispatchEvent(ev); };
    let tracking = false, lastT = null, t0 = 0, moved = false;
    cv.addEventListener('touchstart', (e) => { if (e.touches.length !== 1) return; tracking = true; moved = false; lastT = e.touches[0]; t0 = performance.now(); fire('mousedown', lastT, cv); e.preventDefault(); }, { passive: false });
    cv.addEventListener('touchmove', (e) => { if (!tracking || e.touches.length !== 1) return; lastT = e.touches[0]; moved = true; fire('mousemove', lastT); e.preventDefault(); }, { passive: false });
    const end = (e) => { if (!tracking) return; tracking = false; fire('mouseup', lastT); if (!moved && performance.now() - t0 > 450) { const ce = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: lastT.clientX, clientY: lastT.clientY }); cv.dispatchEvent(ce); } e.preventDefault(); };
    cv.addEventListener('touchend', end, { passive: false }); cv.addEventListener('touchcancel', end, { passive: false });
  })();

  function setupMidi() {
    const sel = $('midi-in'), note = $('midi-note'); if (!sel) return;
    if (!navigator.requestMIDIAccess) { sel.disabled = true; sel.innerHTML = '<option>No Web MIDI in this browser</option>'; if (note) note.textContent = 'MIDI controllers need Chrome or Edge. The keys and a w s e d work anywhere.'; return; }
    let access = null, current = null;
    const attach = (id) => { if (current) current.onmidimessage = null; current = null; if (!access || !id) return; current = access.inputs.get(id); if (!current) return; current.onmidimessage = (m) => { const [st, d1, d2] = m.data; const cmd = st & 0xf0; if (cmd === 0x90 && d2 > 0) api.note_on(d1, d2, 'auto', -1); else if (cmd === 0x80 || (cmd === 0x90 && d2 === 0)) api.note_off(d1); }; };
    const refresh = () => { const ins = [...access.inputs.values()]; sel.innerHTML = ''; const o0 = document.createElement('option'); o0.value = ''; o0.textContent = ins.length ? 'MIDI input: none' : 'No MIDI device found'; sel.appendChild(o0); ins.forEach((i) => { const o = document.createElement('option'); o.value = i.id; o.textContent = i.name; sel.appendChild(o); }); if (ins.length && !current) { sel.value = ins[0].id; attach(ins[0].id); } };
    navigator.requestMIDIAccess({ sysex: false }).then((a) => { access = a; refresh(); a.onstatechange = refresh; sel.onchange = () => attach(sel.value); if (note) note.textContent = 'Rec, play, Stop.'; })
      .catch(() => { sel.disabled = true; sel.innerHTML = '<option>MIDI access denied</option>'; });
  }
  setupMidi();
})();
