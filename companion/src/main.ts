/**
 * Quill Nexus Companion —— 产品级界面(桌面版设计语言)+ region 级 INK 闭环。
 *
 * 数据流:
 *   [INK]:所选工程读 noteRegion(按 region 分组)→ 渲染后端 /render
 *   (可多音色 A/B)→ 写回 region.positionTicks 对齐源 region、
 *   musicDurationTicks 锁定小节跨度、每个音色独立轨(DAW 可 solo 对比)。
 *
 * 铁律:OAuth token 只在本页(浏览器);不向渲染后端提交任何路径;
 * 后端不在线时优雅降级;写回是显式动作(必须亲手选工程)。
 */
import { audiotool } from "@audiotool/nexus"

// 官方 tick 基准(SDK dist/utils/ticks:Beat=3840,与 tempo 无关;未从包根导出故本地实现)
const TICKS_PER_BEAT = 3840
const ticksToSec = (ticks: number, bpm: number) => (ticks / TICKS_PER_BEAT) * (60 / bpm)
const secToTicks = (sec: number, bpm: number) => Math.round((sec / (60 / bpm)) * TICKS_PER_BEAT)

// 部署参数化:本地开发零配置;托管时 Vite env 注入(redirect 须在
// developer.audiotool.com/applications 登记)。
const CLIENT_ID = import.meta.env.VITE_AT_CLIENT_ID ?? ""
const REDIRECT = import.meta.env.VITE_REDIRECT ?? "http://127.0.0.1:5173/"
const QUILL_API = import.meta.env.VITE_QUILL_API ?? "http://127.0.0.1:8723"

const $ = (id: string) => document.getElementById(id)!
const logEl = $("log"), loginBtn = $("loginBtn") as HTMLButtonElement
const controls = $("controls"), takesEl = $("takes"), statusEl = $("quillStatus")
const regionListEl = $("regionList"), timbreCardsEl = $("timbreCards")

interface CapturedAsset { id: string; name: string; f0_hz?: number; note?: string; partials?: number; n_notes?: number; ts: number }
// 未登录也要记住 capture:退到 guest 命名空间(按浏览器);登录后按账号隔离。
// tone 与 behavior 各一个桶,同一模式:capture 落盘,loadAssets 时与服务端合并。
const currentUser = () => ((window as any).__quillUser as string | undefined) || "guest"
function loadStore(bucket: "captures" | "behaviors", user: string): CapturedAsset[] {
  try { return JSON.parse(localStorage.getItem(`quill_${bucket}_${user}`) || "[]") } catch { return [] }
}
function saveStore(bucket: "captures" | "behaviors", user: string, items: CapturedAsset[]) {
  localStorage.setItem(`quill_${bucket}_${user}`, JSON.stringify(items.slice(-20)))
}
function storeAsset(bucket: "captures" | "behaviors", item: CapturedAsset) {
  const user = currentUser()
  saveStore(bucket, user, [...loadStore(bucket, user).filter(c => c.id !== item.id), item])
}
const inkBtn = $("inkBtn") as HTMLButtonElement, inkStatus = $("inkStatus")
const capModal = $("capModal"), capWaveCv = $("capWave") as HTMLCanvasElement
const srcStrip = $("srcStrip") as HTMLCanvasElement

function log(msg: string, cls = "") {
  const line = document.createElement("div")
  if (cls) line.className = cls
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`
  logEl.appendChild(line)
  while (logEl.childElementCount > 150) logEl.firstElementChild!.remove()   // 防 DOM 无限增长
  logEl.scrollTop = logEl.scrollHeight
}

const sent = new Set<string>(JSON.parse(localStorage.getItem("quill_sent") ?? "[]"))
const markSent = (id: string) => {
  sent.add(id)
  localStorage.setItem("quill_sent", JSON.stringify([...sent]))
}

// take_id → 源 region 的 positionTicks。INK 渲染时记下,Send 写回时用它
// 对齐插入 —— 两步流(先试听后写回)不丢对齐信息。落 localStorage 防刷新。
const alignStore: Record<string, number> = (() => {
  try { return JSON.parse(localStorage.getItem("quill_align") ?? "{}") } catch { return {} }
})()
function rememberAlign(tid: string, ticks: number) {
  alignStore[tid] = ticks
  const keys = Object.keys(alignStore)
  for (const k of keys.slice(0, Math.max(0, keys.length - 60))) delete alignStore[k]
  localStorage.setItem("quill_align", JSON.stringify(alignStore))
}

// ---- 实体字段访问助手(NexusObject 两种形状防御)---- //
function fieldVal(f: any): any {
  if (f == null) return undefined
  if (typeof f === "object" && "value" in f) return f.value
  return f
}
function regionTicks(nr: any): { pos: number; dur: number } {
  const r = nr.fields?.region
  const inner = r?.fields ?? r
  return { pos: Number(fieldVal(inner?.positionTicks) ?? 0),
           dur: Number(fieldVal(inner?.durationTicks) ?? 0) }
}
const locKey = (loc: any) => JSON.stringify(fieldVal(loc) ?? loc)

// ---- 旋钮 ---- //
// 铁律:只保留真的作用于渲染的旋钮。前四个随请求送到后端(白名单见
// render_service.SYNTH_KEYS);vibrato 是本地判定阈值,不进请求体。
const SYNTH_KNOBS = ["brightness", "noise", "master", "reverb", "artint"] as const
const knobVals: Record<string, number> = {}

// VIBRATO 旋钮 → 颤音时长阈值(秒)。0 → 0.20s(几乎都揉),1 → 1.20s(几乎不揉)
const vibratoMinDur = () => 0.20 + (knobVals.vibrato ?? 0.35) * 1.0
const synthParamsForRequest = () =>
  Object.fromEntries(SYNTH_KNOBS.filter(k => k in knobVals).map(k => [k, knobVals[k]]))

function initKnobs() {
  document.querySelectorAll<HTMLElement>(".knob[data-knob]").forEach(knob => {
    const key = knob.dataset.knob
    if (!key) return
    knobVals[key] = parseFloat(knob.dataset.val || "0.5")
    updateKnob(knob, key)
    knob.addEventListener("pointerdown", (e: PointerEvent) => {
      e.preventDefault()
      knob.setPointerCapture(e.pointerId)
      const y0 = e.clientY, v0 = knobVals[key]
      const mv = (ev: PointerEvent) => {
        knobVals[key] = Math.max(0, Math.min(1, v0 - (ev.clientY - y0) / 160))
        updateKnob(knob, key)
      }
      // pointercancel 也要解绑,否则指针被系统取消后旋钮会"粘住"跟随鼠标
      const up = () => {
        knob.removeEventListener("pointermove", mv)
        knob.removeEventListener("pointerup", up)
        knob.removeEventListener("pointercancel", up)
        try { knob.releasePointerCapture(e.pointerId) } catch { /* 已释放 */ }
      }
      knob.addEventListener("pointermove", mv)
      knob.addEventListener("pointerup", up)
      knob.addEventListener("pointercancel", up)
    })
  })
}
function updateKnob(knob: HTMLElement, key: string) {
  const v = knobVals[key]
  const needle = knob.querySelector<HTMLElement>(".needle")
  if (needle) needle.style.transform = `translateX(-50%) rotate(${-135 + v * 270}deg)`
  const valEl = document.getElementById(`kv_${key}`)
  if (valEl) {
    valEl.textContent = key === "vibrato"
      ? `${vibratoMinDur().toFixed(2)}s`
      : Math.round(v * 100).toString()
  }
  if (key === "brightness") drawFilterCurve()
  if (key === "master") drawEnvCurve()
  if (key === "artint" || key === "vibrato") drawLfoCurve()
  if (key === "reverb") updateVuWet()
}
initKnobs()

function drawFilterCurve() {
  const path = document.getElementById("filterPath")
  const fill = document.getElementById("filterFill")
  if (!path) return
  const brt = knobVals.brightness ?? 0.5
  const tilt = (brt - 0.5) * 2.0
  const pts: string[] = []
  for (let x = 0; x <= 200; x += 2) {
    const f = x / 200
    const db = tilt * 18 * (f - 0.15)
    const y = 34 - db * 1.4
    pts.push(`${x},${Math.max(4, Math.min(64, y))}`)
  }
  path.setAttribute("d", "M" + pts.join("L"))
  if (fill) fill.setAttribute("d", "M0,68L" + pts.join("L") + "L200,68Z")
}

function drawEnvCurve() {
  const path = document.getElementById("envPath")
  const fill = document.getElementById("envFill")
  if (!path) return
  const m = knobVals.master ?? 0.62
  const peak = 8 + (1 - m) * 20
  const sus = 14 + (1 - m) * 30
  const d = `M0,64 L12,${peak} Q24,${peak - 2} 40,${sus} L140,${sus + 2} Q160,${sus + 4} 180,58 L200,64`
  path.setAttribute("d", d)
  if (fill) fill.setAttribute("d", d + " L200,68 L0,68 Z")
}

function drawLfoCurve() {
  const path = document.getElementById("lfoPath")
  if (!path) return
  const depth = knobVals.artint ?? 0.6
  const amp = depth * 28
  const cycles = 2.5 + depth * 3
  const pts: string[] = []
  for (let x = 0; x <= 200; x++) {
    const y = 34 - Math.sin(x / 200 * Math.PI * 2 * cycles) * amp
    pts.push(`${x},${Math.max(2, Math.min(66, y)).toFixed(1)}`)
  }
  path.setAttribute("d", "M" + pts.join("L"))
}

function initVoiceGrid() {
  const grid = document.getElementById("voiceGrid")
  if (!grid) return
  for (let i = 0; i < 16; i++) {
    const cell = document.createElement("div")
    cell.className = "voice-cell " + (i === 0 ? "on" : "off")
    grid.appendChild(cell)
  }
}

function updateVoiceRange(f0: number, note: string) {
  const bar = document.getElementById("vrBar")
  const line = document.getElementById("vrNote")
  const label = document.getElementById("voiceNote")
  if (!bar || !line || !label) return
  const midi = 69 + 12 * Math.log2(f0 / 440)
  const x = ((midi - 24) / 72) * 200
  const cx = Math.max(4, Math.min(196, x))
  line.setAttribute("x1", String(cx))
  line.setAttribute("x2", String(cx))
  bar.setAttribute("x", String(Math.max(0, cx - 20)))
  bar.setAttribute("width", "40")
  label.textContent = `${note} ${Math.round(f0)}Hz`
}

function updateVuWet() {
  const wet = document.getElementById("vuWetFill")
  const dry = document.getElementById("vuDryFill")
  if (!wet || !dry) return
  const rev = knobVals.reverb ?? 0
  const w = Math.round(rev * 78)
  wet.setAttribute("width", String(w))
  dry.setAttribute("width", String(78 - w))
}

function drawHarmonics(partials?: number) {
  const g = document.getElementById("hBars")
  if (!g) return
  g.innerHTML = ""
  const n = partials || 8
  const barW = 200 / (n * 2 + 1)
  for (let i = 0; i < n; i++) {
    const amp = 1 / (i + 1)
    const h = amp * 58
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
    rect.setAttribute("class", "hbar")
    rect.setAttribute("x", String((i * 2 + 1) * barW))
    rect.setAttribute("y", String(66 - h))
    rect.setAttribute("width", String(barW))
    rect.setAttribute("height", String(h))
    rect.setAttribute("rx", "1.5")
    g.appendChild(rect)
  }
}

// ---- 画布响应式:缓冲区 = CSS 尺寸 × DPR,窗口变化用最后一份数据重绘 ----
// 画布上的绘制都以 cv.width/height 为坐标系,换缓冲区无须改绘制代码。
const lastDraw = new Map<HTMLCanvasElement, () => void>()
function fitCanvas(cv: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const w = Math.max(1, Math.round(cv.clientWidth * dpr))
  const h = Math.max(1, Math.round(cv.clientHeight * dpr))
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h }
}
const canvasRO = new ResizeObserver(entries => {
  for (const e of entries) {
    const cv = e.target as HTMLCanvasElement
    fitCanvas(cv)
    lastDraw.get(cv)?.()
  }
})
canvasRO.observe($("contour"))
canvasRO.observe(srcStrip)
canvasRO.observe(capWaveCv)
// RO 的兜底(部分内嵌 WebView 不触发 RO):window resize 去抖后统一重绘
let rzTimer: number | undefined
window.addEventListener("resize", () => {
  clearTimeout(rzTimer)
  rzTimer = window.setTimeout(() => {
    for (const [cv, redraw] of lastDraw) { fitCanvas(cv); redraw() }
  }, 120)
})
initVoiceGrid()
drawFilterCurve()
drawEnvCurve()
drawLfoCurve()
drawHarmonics()

type RNote = { midi: number; start: number; len: number; vel: number; art: string }
type RegionInfo = { label: string; startTicks: number; notes: RNote[] }

// ---- 右轨:选中 region 的音高轮廓(step 曲线,金)---- //
function drawContour(notes: RNote[]) {
  const cv = $("contour") as HTMLCanvasElement
  fitCanvas(cv)
  lastDraw.set(cv, () => drawContour(notes))
  const ctx = cv.getContext("2d")!
  ctx.clearRect(0, 0, cv.width, cv.height)
  if (!notes.length) return
  const t1 = Math.max(...notes.map(n => n.start + n.len))
  const lo = Math.min(...notes.map(n => n.midi)) - 1
  const hi = Math.max(...notes.map(n => n.midi)) + 1
  const s = cv.width / Math.max(1, cv.clientWidth)   // DPR 比例:线宽/边距随缓冲区缩放
  const X = (t: number) => 10 * s + (t / t1) * (cv.width - 20 * s)
  const Y = (m: number) => cv.height - 12 * s - ((m - lo) / (hi - lo)) * (cv.height - 24 * s)
  // 演奏法语义色 = 蓝青(与桌面版 ARTICULATION 区一致)
  ctx.strokeStyle = "#8aadae"; ctx.lineWidth = 2.5 * s; ctx.lineJoin = "round"
  ctx.shadowColor = "rgba(138,173,174,.5)"; ctx.shadowBlur = 4 * s
  ctx.beginPath()
  for (const n of notes) {
    ctx.lineTo(X(n.start), Y(n.midi)); ctx.lineTo(X(n.start + n.len), Y(n.midi))
  }
  ctx.stroke()
}

// ---- 波形绘制(capture 浮窗 + 左轨 SOURCE 条)---- //
function drawWave(cv: HTMLCanvasElement, ch: Float32Array, color = "#c9a86a") {
  fitCanvas(cv)
  lastDraw.set(cv, () => drawWave(cv, ch, color))
  const ctx = cv.getContext("2d")!
  ctx.clearRect(0, 0, cv.width, cv.height)
  const mid = cv.height / 2
  ctx.strokeStyle = color; ctx.lineWidth = cv.width / Math.max(1, cv.clientWidth)
  ctx.beginPath()
  const step = Math.max(1, Math.floor(ch.length / cv.width))
  for (let x = 0; x < cv.width; x++) {
    let mn = 1, mx = -1
    for (let i = x * step; i < (x + 1) * step && i < ch.length; i++) {
      const v = ch[i]; if (v < mn) mn = v; if (v > mx) mx = v
    }
    ctx.moveTo(x, mid + mn * mid * 0.92); ctx.lineTo(x, mid + mx * mid * 0.92 + 0.5)
  }
  ctx.stroke()
}

// ---- 音色色卡(桌面版 SAVED TONES 同款视觉)---- //
const TONE_STYLE: Record<string, string> = {
  violin: "linear-gradient(135deg,#caa86b,#8a6a30)",
  trumpet: "linear-gradient(135deg,#d9a05a,#8a4f22)",
  guitar: "linear-gradient(135deg,#b08d4f,#5f4520)",
  glass: "linear-gradient(135deg,#8fa3b8,#41546b)",
}
let onTimbreChange = () => {}
function hueGrad(seed: string) {          // captured tone 的稳定生成色
  let h = 0
  for (const c of seed) h = (h * 31 + c.charCodeAt(0)) % 360
  return `linear-gradient(135deg,hsl(${h},32%,52%),hsl(${(h + 24) % 360},38%,26%))`
}
type ToneDef = { value: string | null; label: string; grad?: string; stale?: boolean }
function renderToneCards(tones: ToneDef[], keepOn?: Set<string>) {
  timbreCardsEl.innerHTML = ""
  for (const t of tones) {
    const card = document.createElement("div")
    const key = t.value ?? ""
    const on = !t.stale && (keepOn ? keepOn.has(key) : (t.value === "violin" || t.value === null))
    card.className = "tone" + (on ? " on" : "") + (t.stale ? " stale" : "")
    card.dataset.timbre = key
    card.style.background = t.grad ?? TONE_STYLE[key] ?? hueGrad(key)
    // 名字来自 /assets(可由 /capture 的 ?name= 任意写入)→ 必须走 textContent,
    // 不能拼进 innerHTML,否则是存储型 XSS。
    const check = document.createElement("span")
    check.className = "tcheck"; check.textContent = "✓"
    const nm = document.createElement("span")
    nm.className = "tname"; nm.textContent = t.label
    card.append(check, nm)
    // 过期资产(服务端已淘汰)不可选:选中会让整次 INK 400,提示重录。
    card.onclick = t.stale
      ? () => log(`"${t.label.replace(" ⏳", "")}" expired on server — re-capture to use it`, "err")
      : () => { card.classList.toggle("on"); onTimbreChange() }
    timbreCardsEl.appendChild(card)
  }
}
const behChipsEl = $("behChips")
function renderBehChips(behs: { id: string; name: string; n_notes: number; stale?: boolean }[]) {
  const cur = behChipsEl.querySelector<HTMLElement>(".bchip.on")?.dataset.beh ?? ""
  behChipsEl.innerHTML = ""
  const mk = (val: string, label: string, stale = false) => {
    const c = document.createElement("span")
    c.className = "bchip" + (!stale && (val === cur || (!val && !cur)) ? " on" : "")
      + (stale ? " stale" : "")
    c.dataset.beh = val
    c.textContent = label
    // 过期行为库不可选:选中会让 INK 400,提示重录(与 tone 卡同一规则)。
    c.onclick = stale
      ? () => log(`"${label.replace(" ⏳", "")}" expired on server — re-capture to use it`, "err")
      : () => {
          behChipsEl.querySelectorAll(".bchip").forEach(x => x.classList.remove("on"))
          c.classList.add("on")
          $("nowBeh").textContent = val ? label.split(" · ")[0] : "own"
        }
    behChipsEl.appendChild(c)
  }
  mk("", "own")
  for (const b of behs) mk(`beh:${b.id}`, `${b.name} · ${b.n_notes}n`, b.stale)
}
const chosenBehavior = () =>
  behChipsEl.querySelector<HTMLElement>(".bchip.on")?.dataset.beh || ""
const chosenTimbres = () =>
  [...timbreCardsEl.querySelectorAll<HTMLElement>(".tone.on")]
    .map(c => c.dataset.timbre || null)

async function main() {
  const at = await audiotool({ clientId: CLIENT_ID, redirectUrl: REDIRECT,
                               scope: "project:write project:read sample:write sample:read" })

  // 渲染后端能力:builtin timbres + captured assets 合并成卡
  let builtin: string[] = []
  async function loadAssets(selectNew?: string) {
    const keep = new Set([...timbreCardsEl.querySelectorAll<HTMLElement>(".tone.on")]
      .map(c => c.dataset.timbre ?? ""))
    if (selectNew) keep.add(selectNew)
    try {
      const a = await fetch(`${QUILL_API}/assets`).then(r => r.json())
      const cards: ToneDef[] = builtin.map(t => ({ value: t, label: t }))
      const serverToneIds = new Set<string>()
      for (const t of (a.tones ?? [])) {
        cards.push({ value: `tone:${t.id}`, label: t.name, grad: hueGrad(t.id) })
        serverToneIds.add(t.id)
      }
      for (const c of loadStore("captures", currentUser())) {
        if (!serverToneIds.has(c.id))
          cards.push({ value: `tone:${c.id}`, label: `${c.name} ⏳`,
                       grad: hueGrad(c.id), stale: true })
      }
      if (!cards.length) cards.push({ value: null, label: "current timbre" })
      renderToneCards(cards, keep.size ? keep : undefined)
      const serverBehs = (a.behaviors ?? []) as { id: string; name: string; n_notes: number }[]
      const serverBehIds = new Set(serverBehs.map(b => b.id))
      const staleBehs = loadStore("behaviors", currentUser())
        .filter(c => !serverBehIds.has(c.id))
        .map(c => ({ id: c.id, name: `${c.name} ⏳`, n_notes: c.n_notes ?? 0, stale: true }))
      renderBehChips([...serverBehs, ...staleBehs])
      onTimbreChange()
    } catch { /* 桌面后端无 /assets:忽略 */ }
  }
  fetch(`${QUILL_API}/status`).then(r => r.json()).then(s => {
    builtin = s.timbres ?? []
    renderToneCards(builtin.length ? builtin.map(t => ({ value: t, label: t }))
                                   : [{ value: null, label: "current timbre" }])
    $("footEngine").textContent = `RENDER · ${s.engine ?? "local"} · ${builtin.length || 1} TIMBRES`
    loadAssets()
  }).catch(() => { $("footEngine").textContent = "RENDER · OFFLINE" })

  // ---- CAPTURE 浮窗:暂存 → 波形预览 → SAVE → /capture → 指标回填 ---- //
  const capStatus = $("capStatus"), capName = $("capName") as HTMLInputElement
  const dropZone = $("dropZone"), recBtn = $("recBtn") as HTMLButtonElement
  const capSave = $("capSave") as HTMLButtonElement
  const capPlay = $("capPlay") as HTMLButtonElement
  let pendingBlob: Blob | null = null
  let pendingWave: Float32Array | null = null
  let pendingDur = 0

  $("capOpen").onclick = () => capModal.classList.add("open")
  capModal.onclick = e => { if (e.target === capModal) capModal.classList.remove("open") }

  const fmtT = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`
  async function setPending(blob: Blob) {
    pendingBlob = blob
    try {
      const ac = new AudioContext()
      try {   // 解码失败也要 close,否则 AudioContext 泄漏(浏览器有实例上限)
        const buf = await ac.decodeAudioData(await blob.arrayBuffer())
        pendingWave = buf.getChannelData(0).slice(0)
        pendingDur = buf.duration
      } finally {
        ac.close()
      }
      drawWave(capWaveCv, pendingWave)
      $("capTimer").textContent = fmtT(pendingDur)
      $("statTake").textContent = `${pendingDur.toFixed(1)} s`
      capStatus.textContent = "ready — name it and SAVE"
    } catch {
      pendingWave = null
      capStatus.textContent = "loaded (waveform preview unavailable) — SAVE to analyze"
    }
    capSave.disabled = false
    capPlay.disabled = false
  }
  capPlay.onclick = () => {
    if (!pendingBlob) return
    const url = URL.createObjectURL(pendingBlob)   // 播完即回收,防 blob 泄漏
    const a = new Audio(url)
    a.onended = a.onerror = () => URL.revokeObjectURL(url)
    a.play()
  }

  capSave.onclick = async () => {
    if (!pendingBlob) return
    const name = capName.value.trim() || "capture"
    capSave.disabled = true
    capStatus.textContent = `analyzing "${name}" — extracting tone + behavior…`
    try {
      const r = await fetch(`${QUILL_API}/capture?name=${encodeURIComponent(name)}`,
        { method: "POST", body: pendingBlob }).then(x => x.json())
      if (!r.ok) throw new Error(r.detail || r.warnings?.join("; ") || "capture failed")
      $("statPitch").textContent = r.tone ? `${r.tone.note} ${Math.round(r.tone.f0_hz)}Hz` : "—"
      $("statPartials").textContent = r.tone ? String(r.tone.partials) : "—"
      if (r.tone?.partials) drawHarmonics(r.tone.partials)
      if (r.tone?.f0_hz) updateVoiceRange(r.tone.f0_hz, r.tone.note)
      $("statNotes").textContent = r.behavior ? `${r.behavior.n_notes} notes` : "—"
      const parts = []
      if (r.tone) parts.push("tone ✓")
      if (r.behavior) parts.push(`behavior ✓ (${r.behavior.n_notes} notes)`)
      capStatus.textContent = `"${name}" saved — ${parts.join(" · ")}`
      if (r.tone)
        storeAsset("captures", { id: r.tone.id, name, f0_hz: r.tone.f0_hz,
                                 note: r.tone.note, partials: r.tone.partials, ts: Date.now() })
      if (r.behavior)
        storeAsset("behaviors", { id: r.behavior.id, name,
                                  n_notes: r.behavior.n_notes, ts: Date.now() })
      log(`CAPTURE ✓ "${name}" (${r.duration_s}s) → ${parts.join(", ")}`
          + (r.warnings?.length ? ` [${r.warnings.join("; ")}]` : ""), "ok")
      // 音色语义色 = 粉(与桌面版 TIMBRE 区一致)
      if (pendingWave) drawWave(srcStrip, pendingWave, "#d18a92")
      ;($("srcMeta").children[0] as HTMLElement).textContent =
        r.tone ? `f₀ ${Math.round(r.tone.f0_hz)} Hz · ${r.tone.note}` : name
      ;($("srcMeta").children[1] as HTMLElement).textContent = `${r.duration_s}s`
      await loadAssets(r.tone ? `tone:${r.tone.id}` : undefined)
      // 流程润色:保存成功即完成使命,1.2s 后自动收窗,视线回到主界面的新音色卡
      setTimeout(() => capModal.classList.remove("open"), 1200)
    } catch (e) {
      capStatus.textContent = "capture failed — see log"
      log(`CAPTURE failed: ${e instanceof Error ? e.message : String(e)}`, "err")
    }
    capSave.disabled = false
  }

  const capFile = $("capFile") as HTMLInputElement
  dropZone.onclick = () => capFile.click()
  capFile.onchange = () => { if (capFile.files?.[0]) setPending(capFile.files[0]) }
  dropZone.ondragover = e => { e.preventDefault(); dropZone.classList.add("over") }
  dropZone.ondragleave = () => dropZone.classList.remove("over")
  dropZone.ondrop = e => {
    e.preventDefault(); dropZone.classList.remove("over")
    const f = e.dataTransfer?.files?.[0]
    if (f) setPending(f)
  }
  let rec: MediaRecorder | null = null
  let recChunks: Blob[] = []
  let recT0 = 0
  let recTick: number | undefined
  recBtn.onclick = async () => {
    if (rec) { rec.stop(); return }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      recChunks = []
      rec = new MediaRecorder(stream)
      rec.ondataavailable = e => recChunks.push(e.data)
      rec.onstop = () => {
        stream.getTracks().forEach(t => t.stop())
        clearInterval(recTick)
        recBtn.classList.remove("rec"); recBtn.textContent = "● REC"
        const blob = new Blob(recChunks, { type: rec?.mimeType || "audio/webm" })
        rec = null
        if (blob.size > 2000) setPending(blob)
      }
      rec.start()
      recT0 = Date.now()
      recTick = window.setInterval(() =>
        { $("capTimer").textContent = fmtT((Date.now() - recT0) / 1000) }, 100)
      recBtn.classList.add("rec"); recBtn.textContent = "■ STOP"
      capStatus.textContent = "recording — hold one pitch or play a phrase, click ■ to finish"
      setTimeout(() => rec?.state === "recording" && rec.stop(), 90000)
    } catch (e) {
      log(`mic error: ${e instanceof Error ? e.message : String(e)}`, "err")
    }
  }

  if (at.status === "unauthenticated") {
    if (at.error) log(`auth error: ${at.error}`, "err")
    log("Not logged in — click Login to authorize with Audiotool")
    loginBtn.disabled = false
    loginBtn.onclick = () => at.login()
    return
  }
  ;(window as any).at = at
  loginBtn.textContent = "Logged in ✓"
  loginBtn.classList.remove("gbtn")
  loginBtn.disabled = true
  const outBtn = document.createElement("button")
  outBtn.textContent = "Logout"
  outBtn.style.marginLeft = "8px"
  outBtn.onclick = () => { at.logout(); location.reload() }
  controls.appendChild(outBtn)
  $("footAuth").textContent = `AUDIOTOOL · ${at.userName}`
  ;(window as any).__quillUser = at.userName

  // ---- Project 选择器 ---- //
  const picker = document.createElement("select")
  picker.id = "projectPicker"
  controls.appendChild(picker)
  // 铁律:写回是显式动作 → 占位项强制亲手选;只列本人工程,最近更新在前
  const ph = document.createElement("option")
  ph.value = ""; ph.textContent = "— Select target project —"
  ph.disabled = true; ph.selected = true
  picker.appendChild(ph)
  const projs = await at.projects.listProjects({
    filter: `project.creator_name == "${at.userName}"`,   // CEL 用 ==
    orderBy: "project.update_time desc", pageSize: 30,
  })
  if (projs instanceof Error || !projs?.projects) throw (projs instanceof Error ? projs : new Error("listProjects bad shape"))
  for (const p of projs.projects) {
    const o = document.createElement("option")
    o.value = p.name; o.textContent = p.displayName || p.name
    picker.appendChild(o)
  }
  log(`Logged in as ${at.userName} — ${projs.projects.length} projects`)

  let regions: RegionInfo[] = []
  let selected = -1
  let projBpm = 120
  // 重入守卫:快速换工程/写回后重读会产生并发加载,旧结果落地会把
  // 同一批 region push 两遍 → 列表重复、演奏法计数翻倍。只认最新一次。
  // (必须声明在首次 loadRegions() 调用之前,否则启动自动加载踩 TDZ。)
  let loadSeq = 0

  function syncInk() {
    inkBtn.disabled = !(picker.value && selected >= 0 && chosenTimbres().length)
    const first = timbreCardsEl.querySelector<HTMLElement>(".tone.on")
    if (first) {
      $("toneCardName").textContent = first.querySelector(".tname")?.textContent ?? "—"
      ;($("toneCard") as HTMLElement).style.background =
        (first as HTMLElement).style.background
    }
  }
  onTimbreChange = syncInk

  const savedProj = localStorage.getItem("quill_project")
  if (savedProj && [...picker.options].some(o => o.value === savedProj)) picker.value = savedProj
  picker.onchange = () => { localStorage.setItem("quill_project", picker.value); loadRegions() }
  if (picker.value) loadRegions()

  function renderRegionCards() {
    regionListEl.innerHTML = ""
    if (!regions.length) {
      regionListEl.innerHTML = `<span class="empty">no note regions in this project</span>`
      drawContour([]); selected = -1; syncInk(); return
    }
    regions.forEach((r, i) => {
      const card = document.createElement("div")
      card.className = "rcard"
      const [name, ...meta] = r.label.split(" · ")
      const nmEl = document.createElement("span")
      nmEl.className = "rname"; nmEl.textContent = name
      const mtEl = document.createElement("span")
      mtEl.className = "rmeta"; mtEl.textContent = meta.join(" · ").toUpperCase()
      card.append(nmEl, mtEl)
      card.onclick = () => {
        selected = i
        regionListEl.querySelectorAll(".rcard").forEach((c, j) => c.classList.toggle("on", j === i))
        drawContour(r.notes); syncInk()
      }
      regionListEl.appendChild(card)
    })
    selected = 0
    regionListEl.querySelector(".rcard")?.classList.add("on")
    drawContour(regions[0].notes)
    syncInk()
  }

  async function loadRegions() {
    const seq = ++loadSeq
    regionListEl.innerHTML = `<span class="empty">reading regions…</span>`
    regions = []
    try {
      const doc = await at.open(picker.value)
      await doc.start()
      let noteEnts: any[], regionEnts: any[]
      try {   // 读取中途抛错也必须 stop,否则同步会话泄漏
        const q = (doc as any).queryEntities
        const cfg = q.ofTypes("config").get()[0]
        projBpm = cfg ? Number(fieldVal(cfg.fields.tempoBpm)) || 120 : 120
        $("tempoRead").textContent = `${projBpm.toFixed(1)} BPM`
        noteEnts = q.ofTypes("note").get()
        regionEnts = q.ofTypes("noteRegion").get()
      } finally {
        await doc.stop()
      }
      if (seq !== loadSeq) return   // 已有更新的加载在跑,本次结果作废

      const byColl = new Map<string, any[]>()
      for (const n of noteEnts) {
        const k = locKey(n.fields.collection)
        if (!byColl.has(k)) byColl.set(k, [])
        byColl.get(k)!.push(n)
      }

      const build = (raw: any[], startTicks: number, label: string): RegionInfo | null => {
        if (!raw.length) return null
        const ns = raw.map((n: any) => ({
          pitch: Number(fieldVal(n.fields.pitch)),
          pos: Number(fieldVal(n.fields.positionTicks)),
          dur: Number(fieldVal(n.fields.durationTicks)),
          vel: Number(fieldVal(n.fields.velocity)),
          slide: Boolean(fieldVal(n.fields.doesSlide)),
        })).sort((a, b) => a.pos - b.pos)
        const t0 = ns[0].pos
        let prevEnd = -1
        const notes = ns.map(n => {
          const start = ticksToSec(n.pos - t0, projBpm)
          const len = Math.max(0.03, ticksToSec(n.dur, projBpm))
          // Audiotool 的音符只带 doesSlide,没有颤音位。但引擎的三类演奏法里
          // vibrato 是被标签条件化处理的那一类(去漂移/钳制/AM),不产生它
          // 等于整条标签链路在 DAW 路径上是死的。演奏惯例:够长的持续音
          // 才揉弦 —— 以时长阈值定标,阈值由 VIBRATO 旋钮控制,并在日志里
          // 报出三类计数,使这个判定可见、可审。
          const legato = start < prevEnd - 1e-3
          const art = n.slide ? "slide"
                    : (len >= vibratoMinDur() ? "vibrato" : (legato ? "legato" : "pluck"))
          prevEnd = Math.max(prevEnd, start + len)
          return { midi: n.pitch, start: +start.toFixed(4), len: +len.toFixed(4),
                   vel: Math.round(30 + 97 * Math.min(1, n.vel)), art }
        })
        return { label, startTicks, notes }
      }

      if (regionEnts.length) {
        regionEnts.sort((a: any, b: any) => regionTicks(a).pos - regionTicks(b).pos)
        regionEnts.forEach((re: any, i: number) => {
          const { pos, dur } = regionTicks(re)
          const raw = byColl.get(locKey(re.fields.collection)) ?? []
          if (!raw.length) return
          // note.positionTicks 相对/绝对启发:全部落在 region 区间内=绝对
          const minPos = Math.min(...raw.map((n: any) => Number(fieldVal(n.fields.positionTicks))))
          const absolute = dur > 0 && minPos >= pos - 1 && minPos < pos + dur
          const startTicks = absolute ? minPos : pos + minPos
          const bar = Math.floor(startTicks / (TICKS_PER_BEAT * 4)) + 1
          const info = build(raw, startTicks, `Region ${i + 1} · bar ${bar} · ${raw.length} notes`)
          if (info) regions.push(info)
        })
      }
      if (!regions.length && noteEnts.length) {   // 兜底:无 region 实体 → 虚拟 region
        const minPos = Math.min(...noteEnts.map((n: any) => Number(fieldVal(n.fields.positionTicks))))
        const info = build(noteEnts, minPos, `Whole project · all · ${noteEnts.length} notes`)
        if (info) regions.push(info)
      }
      renderRegionCards()
      const artCounts: Record<string, number> = {}
      for (const r of regions) for (const n of r.notes) artCounts[n.art] = (artCounts[n.art] ?? 0) + 1
      const artSummary = Object.entries(artCounts).map(([k, v]) => `${k}:${v}`).join(" ")
      log(`Found ${regions.length} region(s) @ ${projBpm} bpm [${artSummary}]`)
    } catch (e) {
      if (seq !== loadSeq) return   // 过期加载的报错不覆盖最新状态
      console.error(e)
      regionListEl.innerHTML = `<span class="empty">failed to read this project</span>`
      log(`Failed to read regions: ${e instanceof Error ? e.message : String(e)}`, "err")
    }
  }

  // ---- 上传 + 对齐插入(Send 与 INK 共用)---- //
  async function uploadAndInsert(take: any, opts?: { positionTicks?: number }) {
    const resp = await fetch(`${QUILL_API}/takes/${take.take_id}/audio.wav`)
    if (!resp.ok) throw new Error(`audio fetch ${resp.status}`)
    const wav = await resp.blob()
    log(`[${take.take_id}] Uploading "${take.title}" (${(wav.size / 1024).toFixed(0)} KB)…`)
    const upload = await at.samples.upload({
      file: wav, displayName: take.title,
      description: `Quill take ${take.take_id} · ${take.provenance?.quill_version ?? ""}`,
      kind: "one-shot", visibility: "unlisted",
      tags: ["quill", "take"], bpm: take.bpm || 0,
    })
    if (upload instanceof Error) throw upload
    const up = await upload.uploaded
    if (up instanceof Error) throw up
    // 官方 api.md:insert 不必等转码,只需自带 durationSeconds
    upload.ready.then(s => log(`[${take.take_id}] Transcode done ${s instanceof Error ? "(err)" : "✓"}`))
    const doc = await at.open(picker.value)
    await doc.start()
    try {   // modify 抛错也必须 stop,否则同步会话泄漏
      await doc.modify((t: any) => {
        t.insertSample(
          { name: upload.name, durationSeconds: take.duration_s,
            bpm: take.bpm || undefined, displayName: take.title },
          opts?.positionTicks != null
            ? { region: { positionTicks: opts.positionTicks },
                // 音乐时值 = 渲染秒数按工程 bpm 折算 → 与源 region 同跨度,变速仍对齐
                sample: { musicDurationTicks: secToTicks(take.duration_s, projBpm) } }
            : {})
      })
    } finally {
      await doc.stop()
    }
    markSent(take.take_id)
  }

  async function send(take: any, btn: HTMLButtonElement) {
    if (!picker.value) { log("Select a target project first, then Send", "err"); return }
    btn.disabled = true
    btn.textContent = "sending…"
    try {
      // INK 渲染的 take 带对齐信息 → 按源 region 位置插入;其余追加插入
      const alignTicks = alignStore[take.take_id]
      await uploadAndInsert(take,
        alignTicks != null ? { positionTicks: alignTicks } : undefined)
      const where = alignTicks != null
        ? `aligned at bar ${Math.floor(alignTicks / (TICKS_PER_BEAT * 4)) + 1} (own track, solo to compare)`
        : "appended"
      log(`[${take.take_id}] INSERT ✓ — ${where} in "${picker.selectedOptions[0]?.textContent}"`, "ok")
      btn.textContent = "Send again"
      btn.disabled = false
    } catch (e) {
      console.error(e)
      log(`[${take.take_id}] Failed: ${e instanceof Error ? e.message : String(e)}`, "err")
      btn.disabled = false
      btn.textContent = "Send"
    }
  }

  // ---- INK ---- //
  // 新渲染的 take_id:poll 建行时据此高亮 + 滚动到 TAKES,回答"去哪试听"
  const freshTakes = new Set<string>()
  inkBtn.onclick = async () => {
    const region = regions[selected]
    if (!picker.value || !region) return
    const variants = chosenTimbres()
    inkBtn.disabled = true
    const setStatus = (s: string) => { inkStatus.textContent = s }
    try {
      log(`INK ${region.label} × [${variants.map(v => v ?? "current").join(", ")}]`)
      for (let vi = 0; vi < variants.length; vi++) {
        const timbre = variants[vi]
        const tlabel = timbre?.startsWith("tone:")
          ? (timbreCardsEl.querySelector<HTMLElement>(`[data-timbre="${timbre}"] .tname`)?.textContent ?? "capture")
          : (timbre ?? "take")
        const beh = chosenBehavior()
        const title = `Quill · ${tlabel}${beh ? " × " + (behChipsEl.querySelector<HTMLElement>(".bchip.on")?.textContent?.split(" · ")[0] ?? "") : ""} · ${region.label.split(" · ")[0]}`
        setStatus(`rendering ${tlabel} (${vi + 1}/${variants.length})…`)
        const body: any = { notes: region.notes, bpm: projBpm, title,
                            synth_params: synthParamsForRequest() }
        if (timbre) body.timbre = timbre
        if (beh && !timbre?.startsWith("tone:")) body.behavior = beh
        const rr = await fetch(`${QUILL_API}/render`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }).then(r => r.json())
        if (!rr.ok) throw new Error(rr.msg || rr.detail || rr.code || "render error")
        freshTakes.add(rr.take_id)
        let take: any = null
        for (let i = 0; i < 100; i++) {
          const t = await fetch(`${QUILL_API}/takes/${rr.take_id}`).then(r => r.json())
          if (t.audio?.state === "ready") { take = t; break }
          if (t.audio?.state === "failed") throw new Error("render failed: " + (t.audio.error ?? ""))
          await new Promise(r => setTimeout(r, 500))
        }
        if (!take) throw new Error("render timeout")
        // 两步流:INK 只渲染,写回是 Send 的显式动作。对齐信息现在记下,
        // Send 时按源 region 的 tick 位置插入。
        rememberAlign(rr.take_id, region.startTicks)
        const bar = Math.floor(region.startTicks / (TICKS_PER_BEAT * 4)) + 1
        log(`INK ✓ ${title} — rendered (${take.duration_s}s) · Send places it at bar ${bar}`, "ok")
      }
      setStatus(`done — ${variants.length} take(s) in TAKES below · audition, then Send to place into "${picker.selectedOptions[0]?.textContent}"`)
    } catch (e) {
      console.error(e)
      log(`INK failed: ${e instanceof Error ? e.message : String(e)}`, "err")
      setStatus("failed — see log")
    }
    inkBtn.disabled = false
    syncInk()
  }

  // ---- takes 轮询 ---- //
  const rows = new Map<string, HTMLDivElement>()
  async function poll() {
    try {
      const r = await fetch(`${QUILL_API}/takes`, { signal: AbortSignal.timeout(1500) })
      const { takes } = await r.json()
      statusEl.textContent = "QUILL ONLINE"
      statusEl.className = "mono ok"
      for (const t of takes) {
        let row = rows.get(t.take_id)
        if (!row) {
          takesEl.querySelector(".empty")?.remove()
          row = document.createElement("div")
          row.className = "take"
          row.innerHTML = `<span class="tt"></span><span class="ts"></span>`
          const b = document.createElement("button")
          b.textContent = "Send"
          b.onclick = () => send(t, b)
          row.appendChild(b)
          takesEl.prepend(row)
          rows.set(t.take_id, row)
          if (freshTakes.has(t.take_id)) {
            freshTakes.delete(t.take_id)
            row.classList.add("fresh")
            row.scrollIntoView({ behavior: "smooth", block: "nearest" })
            row.addEventListener("animationend", () => row.classList.remove("fresh"))
          }
        }
        ;(row.querySelector(".tt") as HTMLElement).textContent =
          `${t.title} · ${t.duration_s}s`
        const st = row.querySelector(".ts") as HTMLElement
        st.textContent = (sent.has(t.take_id) ? "SENT" : t.audio.state.toUpperCase())
        if (t.audio.state === "ready" && !row.querySelector("audio")) {
          const a = document.createElement("audio")
          a.controls = true; a.preload = "none"
          a.src = `${QUILL_API}/takes/${t.take_id}/audio.wav`
          row.insertBefore(a, row.querySelector("button"))
        }
        const b = row.querySelector("button")!
        if (b.textContent !== "sending…") {
          b.disabled = t.audio.state !== "ready"
          b.textContent = sent.has(t.take_id) ? "Send again" : "Send"
        }
      }
    } catch {
      statusEl.textContent = "RENDER OFFLINE"
      statusEl.className = "mono err"
    }
  }
  poll()
  setInterval(() => { if (!document.hidden) poll() }, 3000)   // 后台标签页不空转
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll() })
}

main().catch(e => log(`init failed: ${e?.message ?? e}`, "err"))
// 静默失败是最贵的 bug:未捕获的 Promise 拒绝一律进可见日志
window.addEventListener("unhandledrejection", e =>
  log(`unhandled: ${e.reason?.message ?? e.reason}`, "err"))
