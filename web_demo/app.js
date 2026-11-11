/* Quill web demo: a prerendered timbre × behavior matrix, plus the visitor's own melody rendered
   by the render service (/status, /render, /takes/{id}/audio.wav). No frameworks, no external APIs. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ICON = { violin: '🎻', trumpet: '🎺', guitar: '🎸', glass: '🥂' };
  const NICE = { violin: 'violin', trumpet: 'trumpet', guitar: 'guitar', glass: 'wine glass', distilled: 'MIDI-DDSP distilled' };

  // ---------------- matrix ----------------
  const audio = new Audio();
  let manifest = null, melody = 'lyrical', playingCell = null;

  function buildMatrix() {
    const tbl = $('matrix'), head = tbl.tHead.rows[0], body = tbl.tBodies[0];
    while (head.cells.length > 1) head.deleteCell(1);
    body.innerHTML = '';
    manifest.behaviors.forEach((b) => {
      const th = document.createElement('th'); th.innerHTML = `<span class="ico">${ICON[b] || ''}</span>${NICE[b] || b}`; head.appendChild(th);
    });
    manifest.timbres.forEach((t) => {
      const tr = document.createElement('tr');
      const th = document.createElement('th'); th.className = 'row'; th.innerHTML = `<span class="ico">${ICON[t] || ''}</span>${NICE[t] || t}`; tr.appendChild(th);
      manifest.behaviors.forEach((b) => {
        const td = document.createElement('td');
        const btn = document.createElement('button');
        btn.className = 'cell' + (t === b ? ' diag' : '');
        btn.dataset.timbre = t; btn.dataset.behavior = b;
        btn.setAttribute('aria-label', `${NICE[t]} timbre played with ${NICE[b]} behavior`);
        btn.innerHTML = `<span class="lbl">${t === b ? 'as recorded' : `${b} → ${t}`}</span><span class="bar"></span>`;
        btn.addEventListener('click', () => playCell(btn));
        td.appendChild(btn); tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    // phone layout: one row per timbre, libraries as inline buttons (same handler)
    const list = $('matrix-list'); list.innerHTML = '';
    manifest.timbres.forEach((t) => {
      const row = document.createElement('div'); row.className = 'mrow';
      const h = document.createElement('h4'); h.textContent = `${ICON[t] || ''} ${NICE[t] || t}`; row.appendChild(h);
      manifest.behaviors.forEach((b) => { const btn = document.createElement('button'); btn.className = 'cell' + (t === b ? ' diag' : ''); btn.dataset.timbre = t; btn.dataset.behavior = b; btn.innerHTML = `<span class="lbl">${t === b ? 'as recorded' : `${b} →`}</span><span class="bar"></span>`; btn.addEventListener('click', () => playCell(btn)); row.appendChild(btn); });
      list.appendChild(row);
    });
  }

  // The visible cell for a pair: the table on desktop, the list on phones.
  function cellFor(t, b) {
    const all = [...document.querySelectorAll(`.cell[data-timbre="${t}"][data-behavior="${b}"]`)];
    return all.find((el) => el.offsetParent !== null) || all[0] || null;
  }
  function clipFor(t, b) {
    return manifest.clips.find((c) => c.melody === melody && c.timbre === t && c.behavior === b);
  }

  let lastPair = null, heard = 0;
  // After a native cell, light up the other players in the same row.
  function suggestNext(t, b) {
    document.querySelectorAll('.cell.suggest').forEach((c) => c.classList.remove('suggest'));
    if (t === b) document.querySelectorAll(`.cell[data-timbre="${t}"]`).forEach((c) => { if (c.dataset.behavior !== b && !c.classList.contains('heard')) c.classList.add('suggest'); });
  }
  function playCell(btn) {
    const clip = clipFor(btn.dataset.timbre, btn.dataset.behavior);
    if (!clip) return;
    lastPair = { timbre: btn.dataset.timbre, behavior: btn.dataset.behavior };
    btn.classList.add('heard'); heard++;
    suggestNext(lastPair.timbre, lastPair.behavior);
    markStep(lastPair.timbre === lastPair.behavior ? 1 : 2);
    guide(lastPair.timbre === lastPair.behavior ? 'native' : 'swapped');
    if (playingCell === btn && !audio.paused) { audio.pause(); return; }
    if (playingCell) playingCell.classList.remove('playing');
    playingCell = btn; btn.classList.add('playing');
    audio.src = clip.file; audio.currentTime = 0;
    audio.play().catch(() => {});
    $('nowplaying').innerHTML = `<b>${NICE[clip.timbre]}</b> timbre · <b>${NICE[clip.behavior]}</b> style library · ${melody} · ${clip.dur_s}s`;
  }
  audio.addEventListener('ended', () => { if (playingCell) playingCell.classList.remove('playing'); $('nowplaying').textContent = heard >= 2 ? 'Now your own melody, below.' : 'Same row, another library.'; });
  audio.addEventListener('timeupdate', () => { if (playingCell && audio.duration) { const bar = playingCell.querySelector('.bar'); if (bar) bar.style.width = (100 * audio.currentTime / audio.duration) + '%'; } });
  audio.addEventListener('pause', () => { if (playingCell && audio.ended === false) playingCell.classList.remove('playing'); });

  // Scoped to data-melody: the vibrato toggle shares the .seg-btn look.
  document.querySelectorAll('.seg-btn[data-melody]').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.seg-btn[data-melody]').forEach((x) => { x.classList.toggle('is-on', x === b); x.setAttribute('aria-checked', x === b ? 'true' : 'false'); });
    melody = b.dataset.melody;
    if (playingCell) { audio.pause(); playingCell.classList.remove('playing'); playingCell = null; }
    $('nowplaying').textContent = 'Pick a cell.';
    loadMelodyIntoRoll(melody);
  }));

  // ---------------- flow: steps, CTAs, carry the chosen pair into the roll ----------------
  function markStep(n) { document.querySelectorAll('.steps .step').forEach((el) => { const k = +el.dataset.step; el.classList.toggle('done', k < n || (k === n && n < 3)); el.classList.toggle('now', k === n); }); }
  function focusCard(id) { document.querySelectorAll('.card.focus').forEach((c) => c.classList.remove('focus')); const c = $(id); c.classList.add('focus'); c.scrollIntoView({ behavior: 'smooth', block: 'start' }); setTimeout(() => c.classList.remove('focus'), 1800); }
  function applyPair(pair) {
    if (!pair) return;
    const st = $('sel-timbre'), sb = $('sel-behavior');
    if ([...st.options].some((o) => o.value === pair.timbre)) st.value = pair.timbre;
    if ([...sb.options].some((o) => o.value === pair.behavior)) sb.value = pair.behavior;
    if (pr()) { pr().setNeural(st.value); pr().setBehavior(sb.value); }
    renderChips();
  }
  $('cta-listen').addEventListener('click', () => { focusCard('matrix-card'); const cell = cellFor('glass', 'glass') || document.querySelector('.cell'); if (cell) setTimeout(() => playCell(cell), 350); });
  $('cta-play').addEventListener('click', () => { markStep(3); focusCard('render-card'); if (pr() && !pr().getNotes().length) loadMelodyIntoRoll(melody); guide('roll'); });
  document.querySelectorAll('.steps .step').forEach((el) => el.addEventListener('click', (e) => { e.preventDefault(); const k = +el.dataset.step; markStep(k); focusCard(k === 3 ? 'render-card' : 'matrix-card'); guide(k === 3 ? 'roll' : (k === 2 && lastPair ? 'native' : 'start')); }));
  // ---------------- guide bar: one next action at a time ----------------
  let gState = 'start';
  const guideDefs = {
    start:   () => ({ step: '01', msg: 'Start by listening to one instrument as it was recorded.', act: '▶ Listen', run: () => $('cta-listen').click() }),
    native:  () => { const t = lastPair.timbre, alt = manifest.behaviors.find((b) => b !== t); return { step: '02', msg: `That is ${NICE[t]} with its own library. Same instrument, another player:`, act: `▶ ${NICE[alt]} playing ${NICE[t]}`, run: () => { const c = cellFor(t, alt); if (c) playCell(c); } }; },
    swapped: () => ({ step: '03', msg: `You heard ${NICE[lastPair.behavior]} play the ${NICE[lastPair.timbre]}. Now put your own melody through that pair.`, act: 'Open the piano roll ↓', run: () => { applyPair(lastPair); markStep(3); focusCard('render-card'); if (pr() && !pr().getNotes().length) loadMelodyIntoRoll(melody); guide('roll'); } }),
    roll:    () => ({ step: '03', msg: `Press Play to render ${NICE[$('sel-timbre').value] || 'the timbre'} × ${NICE[$('sel-behavior').value] || 'its own library'}. Or record your own first.`, act: '▶ Play', run: () => { if (pr()) pr().play(); } }),
    rendering: () => ({ step: '03', msg: 'Rendering on the server. The first one loads the models, a few seconds.', act: '…', run: () => {} }),
    rendered: () => { const cur = $('sel-behavior').value; const alt = (api._behaviors || []).find((b) => b !== cur && b !== 'distilled'); return { step: '03', msg: `Rendered ${NICE[$('sel-timbre').value]} × ${NICE[cur] || cur}. Try another player on the same melody, or record your own.`, act: alt ? `▶ ${NICE[alt]}` : 'Download WAV', run: () => { if (alt) { $('sel-behavior').value = alt; $('sel-behavior').dispatchEvent(new Event('change')); if (pr()) pr().play(); } else $('dl-wav').click(); } }; },
    recording: () => ({ step: '03', msg: 'Recording. Play the keys, then stop.', act: '■ Stop', run: () => { if (pr()) pr().stop(); } }),
  };
  function guide(state) {
    if (!guideDefs[state] || (state !== 'start' && !manifest)) return;
    gState = state; const d = guideDefs[state]();
    $('g-step').textContent = d.step; $('g-msg').textContent = d.msg; const a = $('g-act'); a.textContent = d.act; a.disabled = d.act === '…'; a.onclick = d.run;
    a.classList.toggle('quiet', state === 'rendering');
  }
  window.quillGuide = guide;

  // One-click "try another player" after a render.
  function renderChips() {
    const host = $('alt-chips'); if (!host) return; host.innerHTML = '';
    const cur = $('sel-behavior').value, t = $('sel-timbre').value;
    if (!api || !api._behaviors.length || !(pr() && pr().getNotes().length)) return;
    const lab = document.createElement('span'); lab.className = 'chips-lab'; lab.textContent = `${NICE[t] || t} timbre, other libraries:`; host.appendChild(lab);
    api._behaviors.forEach((b) => { const c = document.createElement('button'); c.className = 'chip' + (b === cur ? ' cur' : ''); c.textContent = (ICON[b] || '') + ' ' + (NICE[b] || b); c.disabled = b === cur; c.addEventListener('click', () => { $('sel-behavior').value = b; if (pr()) { pr().setBehavior(b); pr().play(); } renderChips(); }); host.appendChild(c); });
  }
  window.quillRenderDone = () => { renderChips(); guide('rendered'); };
  window.quillRecState = (on, n) => { guide(on ? 'recording' : 'roll'); const hint = document.querySelector('#render-card .hint'); const rb = $('prRec'); if (rb) rb.classList.toggle('on', !!on); if (!hint) return; hint.classList.toggle('live', !!on); hint.textContent = on ? 'Recording. Play, then Stop.' : (n ? `${n} ${n === 1 ? 'note' : 'notes'}. Tidy ${n === 1 ? 'it' : 'them'} if you like, then Play.` : 'Record from the keys, from a w s e d f t g y h u j k, or from a MIDI controller. Draw and drag notes. Right-click one to mark vibrato or slide. Play renders it.'); };

  function ensureStartingNotes() { if (manifest && pr() && !pr().getNotes().length) loadMelodyIntoRoll(melody); }
  fetch('manifest.json').then((r) => r.json()).then((m) => { manifest = m; buildMatrix(); ensureStartingNotes(); guide('start'); }).catch(() => {
    $('nowplaying').textContent = 'Could not load the prerendered clips.';
  });

  // ---------------- piano roll glue (editor lives in pianoroll.js, backend in webapi.js) ----------------
  const MAX_S = 30;
  const api = window.quillWebApi;
  const pr = () => window.quillPiano;
  function clampBpm() { const v = Math.max(40, Math.min(200, +$('bpm').value || 100)); $('bpm').value = v; return v; }
  // Expressive labels for plain notes only; notes marked via right-click are kept.
  function autoLabel(notes) {
    if (!$('auto-art').checked) return notes;
    const out = notes.map((n) => Object.assign({}, n)).sort((a, b) => a.start - b.start);
    for (let i = 0; i < out.length; i++) {
      const n = out[i]; if (n.art && n.art !== 'pluck') continue;
      const prev = i ? out[i - 1] : null;
      const tied = prev && Math.abs((prev.start + prev.len) - n.start) < 0.03;
      if (n.len >= 0.5) n.art = 'vibrato';
      else if (tied && Math.abs(n.midi - prev.midi) >= 3) n.art = 'slide';
    }
    return out;
  }
  window.quillAutoLabel = autoLabel;
  function updateInfo() {
    const n = pr() ? pr().getNotes() : []; const dur = n.length ? Math.max(...n.map((x) => x.start + x.len)) : 0;
    $('roll-info').textContent = n.length ? `${n.length} notes · ${dur.toFixed(1)} s` : '';
    if (dur > MAX_S) setStatus(`Keep it under ${MAX_S} s for the demo server.`, true);
    else if (/Keep it under/.test($('render-status').textContent)) setStatus('');
  }
  window.quillSessionDirty = updateInfo;            // called by the roll on every edit
  function loadMelodyIntoRoll(name) {
    if (!manifest || !manifest.melodies[name] || !pr()) return;
    pr().setBPM(clampBpm()); pr().setNotes(manifest.melodies[name].notes); updateInfo();
  }
  $('roll-example').addEventListener('click', () => loadMelodyIntoRoll(melody));
  $('bpm').addEventListener('change', () => { if (pr()) pr().setBPM(clampBpm()); updateInfo(); });
  $('metro').addEventListener('change', (e) => { if (pr()) pr().setMetro(e.target.checked); });
  $('countin').addEventListener('change', (e) => { if (pr()) pr().setCountin(e.target.checked); });
  $('sel-timbre').addEventListener('change', (e) => { if (pr()) pr().setNeural(e.target.value); renderChips(); saveSession(); });
  $('sel-behavior').addEventListener('change', (e) => { if (pr()) pr().setBehavior(e.target.value); renderChips(); saveSession(); });
  // The roll initialises asynchronously: it waits for the canvas to have a size.
  let _wait = 0;
  const whenRoll = () => { if (pr()) { pr().setBPM(clampBpm()); pr().setNeural($('sel-timbre').value || 'violin'); pr().setBehavior($('sel-behavior').value || ''); updateInfo(); if (restoreSession()) setStatus('Session restored.'); ensureStartingNotes(); } else if (_wait++ < 100) setTimeout(whenRoll, 100); };
  whenRoll();

  // ---------------- performance panel (desktop knobs that reach the renderer) ----------------
  const PERF_DEF = { artint: 0.41, timing: 0.5, continuity: 1.0, brightness: 0.5, noise: 0.6, reverb: 0.0, vibmodel: 0, randomness: 0.3 };
  const perf = Object.assign({}, PERF_DEF); window.quillPerf = perf;
  const fmt = {
    artint: (v) => (0.3 + v * 1.7).toFixed(1) + '×', timing: (v) => v.toFixed(2), continuity: (v) => v.toFixed(2),
    brightness: (v) => ((v - 0.5) * 12 >= 0 ? '+' : '') + ((v - 0.5) * 12).toFixed(1) + ' dB', noise: (v) => Math.round(v * 100) + '%',
    reverb: (v) => (v < 0.02 ? 'dry' : Math.round(v * 100) + '%'), randomness: (v) => v.toFixed(2),
  };
  function paintPerf() {
    Object.keys(fmt).forEach((k) => { const i = $('p-' + k), o = $('v-' + k); if (i) i.value = perf[k]; if (o) o.textContent = fmt[k](perf[k]); });
    document.querySelectorAll('[data-vib]').forEach((b) => { const on = (b.dataset.vib === 'model') === (perf.vibmodel >= 0.5); b.classList.toggle('is-on', on); b.setAttribute('aria-checked', on ? 'true' : 'false'); });
    $('rand-wrap').style.display = perf.vibmodel >= 0.5 ? '' : 'none';
  }
  Object.keys(fmt).forEach((k) => { const i = $('p-' + k); if (!i) return; i.addEventListener('input', () => { perf[k] = +i.value; $('v-' + k).textContent = fmt[k](perf[k]); }); i.addEventListener('change', saveSession); });
  document.querySelectorAll('[data-vib]').forEach((b) => b.addEventListener('click', () => { perf.vibmodel = b.dataset.vib === 'model' ? 1 : 0; paintPerf(); saveSession(); }));
  $('perf-reset').addEventListener('click', () => { Object.assign(perf, PERF_DEF); paintPerf(); saveSession(); });
  let perfOpen = false;
  function setPerfOpen(v) { perfOpen = !!v; $('perf-body').hidden = !perfOpen; $('perf-toggle').setAttribute('aria-expanded', perfOpen ? 'true' : 'false'); }
  $('perf-toggle').addEventListener('click', () => { setPerfOpen(!perfOpen); saveSession(); });
  paintPerf();

  // Transport extras.
  $('loop').addEventListener('change', (e) => { if (pr()) pr().setLoop(e.target.checked); saveSession(); });
  const TSIGS = [4, 3, 2, 6]; let tsig = 4;
  $('tsig').addEventListener('click', () => { tsig = TSIGS[(TSIGS.indexOf(tsig) + 1) % TSIGS.length]; $('tsig').textContent = tsig + '/4'; if (pr()) pr().setTimeSig(tsig); saveSession(); });

  // Articulation chips: the last played or recorded note.
  window.quillArtVisual = (label) => { document.querySelectorAll('#art-chips span').forEach((c) => c.classList.toggle('on', c.dataset.a === label)); };

  // Transport button states: rendering, playing, idle. The roll itself does not know what the server is doing.
  const playBtn = $('prPlay');
  window.quillBusy = (on) => { playBtn.classList.toggle('busy', !!on); playBtn.textContent = on ? '…' : (window.__quillPlaying ? '▶' : '▶'); if (on) guide('rendering'); };
  const ra = $('result-audio');
  ra.addEventListener('play', () => { window.__quillPlaying = true; playBtn.textContent = '▶'; playBtn.classList.add('live'); });
  const idle = () => { window.__quillPlaying = false; if (!playBtn.classList.contains('busy')) playBtn.textContent = '▶'; playBtn.classList.remove('live'); };
  ra.addEventListener('pause', idle); ra.addEventListener('ended', idle);
  $('prStop').addEventListener('click', idle);

  // Save .mid: type-0 SMF from the roll, seconds converted to ticks at the current tempo.
  function writeMidi(notes, bpm) {
    const PPQ = 480, tick = (sec) => Math.round(sec / (60 / bpm) * PPQ);
    const vlq = (n) => { const b = [n & 0x7f]; n >>= 7; while (n > 0) { b.unshift((n & 0x7f) | 0x80); n >>= 7; } return b; };
    const ev = [];
    notes.forEach((n) => { ev.push({ t: tick(n.start), d: [0x90, n.midi, n.vel || 100] }); ev.push({ t: tick(n.start + n.len), d: [0x80, n.midi, 0] }); });
    ev.sort((a, b) => a.t - b.t || (a.d[0] === 0x80 ? -1 : 1));
    const tempo = Math.round(60000000 / bpm);
    let trk = [0, 0xff, 0x51, 3, (tempo >> 16) & 255, (tempo >> 8) & 255, tempo & 255];
    let last = 0; ev.forEach((e) => { trk = trk.concat(vlq(e.t - last), e.d); last = e.t; });
    trk = trk.concat([0, 0xff, 0x2f, 0]);
    const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
    const bytes = [0x4d, 0x54, 0x68, 0x64, ...u32(6), 0, 0, 0, 1, (PPQ >> 8) & 255, PPQ & 255, 0x4d, 0x54, 0x72, 0x6b, ...u32(trk.length), ...trk];
    return new Blob([new Uint8Array(bytes)], { type: 'audio/midi' });
  }
  $('dl-mid').addEventListener('click', () => {
    const notes = pr() ? pr().getNotes() : []; if (!notes.length) { setStatus('Nothing to save yet.', true); return; }
    const url = URL.createObjectURL(writeMidi(notes, clampBpm())); const a = document.createElement('a'); a.href = url; a.download = `quill_${$('sel-timbre').value}_x_${$('sel-behavior').value || 'self'}.mid`; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
  });

  // Session (localStorage): the roll, tempo, selection and panel, the web counterpart of the desktop's assets/session.json.
  const SKEY = 'quill.web.session.v1';
  function saveSession() {
    try {
      const st = { notes: pr() ? pr().getNotes() : [], bpm: clampBpm(), timbre: $('sel-timbre').value, behavior: $('sel-behavior').value, perf, perfOpen, autoArt: $('auto-art').checked, metro: $('metro').checked, countin: $('countin').checked, loop: $('loop').checked, tsig };
      localStorage.setItem(SKEY, JSON.stringify(st));
    } catch (e) { /* private mode etc. */ }
  }
  function restoreSession() {
    let st = null; try { st = JSON.parse(localStorage.getItem(SKEY) || 'null'); } catch (e) { st = null; }
    if (!st) return false;
    if (st.perf) { Object.keys(PERF_DEF).forEach((k) => { if (typeof st.perf[k] === 'number') perf[k] = st.perf[k]; }); paintPerf(); }
    if (typeof st.perfOpen === 'boolean') setPerfOpen(st.perfOpen);
    if (typeof st.autoArt === 'boolean') $('auto-art').checked = st.autoArt;
    if (typeof st.metro === 'boolean') { $('metro').checked = st.metro; if (pr()) pr().setMetro(st.metro); }
    if (typeof st.countin === 'boolean') { $('countin').checked = st.countin; if (pr()) pr().setCountin(st.countin); }
    if (typeof st.loop === 'boolean') { $('loop').checked = st.loop; if (pr()) pr().setLoop(st.loop); }
    if (TSIGS.includes(st.tsig)) { tsig = st.tsig; $('tsig').textContent = tsig + '/4'; if (pr()) pr().setTimeSig(tsig); }
    if (st.bpm) { $('bpm').value = st.bpm; if (pr()) pr().setBPM(clampBpm()); }
    if (pr() && Array.isArray(st.notes) && st.notes.length) { pr().setNotes(st.notes); updateInfo(); }
    // The selection is applied once /status has filled the selects, or right now if they are already filled
    // (status may win the race).
    window.__sessionSel = { timbre: st.timbre, behavior: st.behavior };
    const applySel = (id, v) => { const el = $(id); if (v && [...el.options].some((o) => o.value === v)) { el.value = v; return true; } return false; };
    if (applySel('sel-timbre', st.timbre) && pr()) pr().setNeural(st.timbre);
    if (applySel('sel-behavior', st.behavior) && pr()) pr().setBehavior(st.behavior);
    return true;
  }
  const _oldDirty = window.quillSessionDirty; window.quillSessionDirty = () => { _oldDirty(); saveSession(); };
  $('auto-art').addEventListener('change', saveSession); $('bpm').addEventListener('change', saveSession);
  $('metro').addEventListener('change', saveSession); $('countin').addEventListener('change', saveSession);

  // ---------------- MIDI file (Standard MIDI File, tracks merged, first tempo map) ----------------
  function parseMidi(buf) {
    const d = new DataView(buf); let p = 0;
    const u32 = () => { const v = d.getUint32(p); p += 4; return v; }, u16 = () => { const v = d.getUint16(p); p += 2; return v; }, u8 = () => d.getUint8(p++);
    const str4 = () => { const s = String.fromCharCode(u8(), u8(), u8(), u8()); return s; };
    const vlq = () => { let v = 0, b; do { b = u8(); v = (v << 7) | (b & 0x7f); } while (b & 0x80); return v; };
    if (str4() !== 'MThd') throw new Error('not a MIDI file');
    const hlen = u32(); u16(); const ntr = u16(); const div = u16(); p += hlen - 6;
    if (div & 0x8000) throw new Error('SMPTE time not supported');
    const tempos = [{ tick: 0, us: 500000 }]; const evs = [];
    for (let t = 0; t < ntr; t++) {
      if (str4() !== 'MTrk') throw new Error('bad track');
      const len = u32(), end = p + len; let tick = 0, status = 0;
      while (p < end) {
        tick += vlq(); let b = u8();
        if (b < 0x80) { p--; b = status; } else status = b;
        const type = b & 0xf0;
        if (b === 0xff) { const mt = u8(), l = vlq(); if (mt === 0x51) { tempos.push({ tick, us: (d.getUint8(p) << 16) | (d.getUint8(p + 1) << 8) | d.getUint8(p + 2) }); } p += l; }
        else if (b === 0xf0 || b === 0xf7) { p += vlq(); }
        else if (type === 0x90 || type === 0x80) { const n = u8(), v = u8(); evs.push({ tick, on: type === 0x90 && v > 0, n, v }); }
        else if (type === 0xc0 || type === 0xd0) { p += 1; }
        else { p += 2; }
      }
      p = end;
    }
    tempos.sort((a, b) => a.tick - b.tick);
    const toSec = (tick) => { let sec = 0, last = tempos[0]; for (let i = 1; i < tempos.length && tempos[i].tick <= tick; i++) { sec += (tempos[i].tick - last.tick) / div * last.us / 1e6; last = tempos[i]; } return sec + (tick - last.tick) / div * last.us / 1e6; };
    evs.sort((a, b) => a.tick - b.tick);
    const open = {}, notes = [];
    evs.forEach((e) => {
      if (e.on) { open[e.n] = e; }
      else if (open[e.n]) { const s = toSec(open[e.n].tick), l = Math.max(0.05, toSec(e.tick) - s); notes.push({ midi: e.n, start: +s.toFixed(4), len: +l.toFixed(4), vel: open[e.n].v, art: 'pluck' }); delete open[e.n]; }
    });
    if (!notes.length) throw new Error('no notes found');
    const t0 = Math.min(...notes.map((n) => n.start));
    notes.forEach((n) => { n.start = +(n.start - t0).toFixed(4); });
    return notes.sort((a, b) => a.start - b.start);
  }
  $('midi-file').addEventListener('change', (ev) => {
    const f = ev.target.files && ev.target.files[0]; if (!f) return;
    f.arrayBuffer().then((buf) => { const notes = parseMidi(buf); if (pr()) { pr().setBPM(clampBpm()); pr().setNotes(notes); } updateInfo(); setStatus(`Loaded ${f.name} · ${notes.length} notes`); })
      .catch((e) => setStatus(`Could not read MIDI: ${e.message}`, true));
    ev.target.value = '';
  });

  // ---------------- render service ----------------
  let serviceOk = false;
  function setStatus(msg, err) { const el = $('render-status'); el.textContent = msg || ''; el.classList.toggle('err', !!err); }
  if (api) api.onStatus(setStatus);
  function fillSelect(sel, items, icons) {
    sel.innerHTML = ''; items.forEach((it) => { const o = document.createElement('option'); o.value = it; o.textContent = (icons && ICON[it] ? ICON[it] + ' ' : '') + (NICE[it] || it); sel.appendChild(o); });
  }
  fetch('status').then((r) => r.json()).then((s) => {
    serviceOk = !!s.ok; const badge = $('svc-badge');
    badge.textContent = serviceOk ? 'render service ready' : 'render service offline'; badge.className = 'badge ' + (serviceOk ? 'ok' : 'err');
    // The guitar timbre is hidden on the web (plucked-envelope quality); the guitar behavior stays.
    // Behaviors include 'distilled' when its checkpoint is deployed.
    const timbres = (s.timbres || []).filter((t) => t !== 'guitar');
    const behaviors = (s.behaviors || []).slice();
    if (api) { api._instruments = timbres; api._behaviors = behaviors; }
    fillSelect($('sel-timbre'), timbres, true); fillSelect($('sel-behavior'), behaviors, true);
    if (timbres.includes('glass')) $('sel-timbre').value = 'glass';
    if (behaviors.includes('violin')) $('sel-behavior').value = 'violin';
    const ss = window.__sessionSel || {};
    if (ss.timbre && timbres.includes(ss.timbre)) $('sel-timbre').value = ss.timbre;
    if (ss.behavior && behaviors.includes(ss.behavior)) $('sel-behavior').value = ss.behavior;
    window.__sessionSel = null;
    if (pr()) { pr().setNeural($('sel-timbre').value); pr().setBehavior($('sel-behavior').value); }
    $('version').textContent = s.version ? `render service v${s.version}` : '';
  }).catch(() => { const badge = $('svc-badge'); badge.textContent = 'render service offline'; badge.className = 'badge err'; setStatus('The matrix above still plays; rendering needs the service.', true); });

  window.quillDemo = { parseMidi, autoLabel };   // 调试用
})();
