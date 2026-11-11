"""Stage 2 unit tests: residual / noise / formant redirection / snapshot serialisation."""

import numpy as np
import pytest

from analysis.harmonic import harmonics
from analysis.residual import attack_residual, extract_residual, noise_envelope
from snapshot import load_snapshot, make_snapshot, save_snapshot
from synth.additive import redirect_harmonics
from synth.noise import istft_np, mel_filterbank, stft_np, synth_noise

SR = 44100


def _band_energy(y, sr, lo, hi):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float(Y[(f >= lo) & (f < hi)].sum())


# --------------------------------------------------------------------------- #
# numpy STFT/ISTFT COLA perfect reconstruction
# --------------------------------------------------------------------------- #
def test_stft_istft_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(44100).astype(np.float64)
    X = stft_np(x, 1024, 256)
    xr = istft_np(X, 1024, 256, length=len(x))
    # the inner region (avoiding incomplete frames at the ends) should reconstruct perfectly
    err = np.max(np.abs(xr[2048:-2048] - x[2048:-2048]))
    assert err < 1e-6, f"STFT/ISTFT round-trip error {err:.2e}"


# --------------------------------------------------------------------------- #
# noise shaping round trip: band envelope of the synthesised noise ≈ target N_env
# --------------------------------------------------------------------------- #
def test_synth_noise_envelope_roundtrip():
    n_mels = 40
    fb, centers = mel_filterbank(SR, 1024, n_mels)
    # target: smooth envelope with a mid-frequency bump
    bump = np.exp(-(((np.arange(n_mels) - 18) / 6.0) ** 2))
    N_target = (0.2 + bump)[:, None].astype(np.float32)  # [40, 1]

    y = synth_noise(N_target, SR, length=SR, seed=1)
    N_meas, _ = noise_envelope(y, sr=SR, n_mels=n_mels)
    N_meas_med = np.median(N_meas, axis=1)

    active = N_target[:, 0] > 0.3
    err_db = 20 * np.log10((N_meas_med[active] + 1e-9) / (N_target[active, 0] + 1e-9))
    assert np.mean(np.abs(err_db)) < 3.0, f"mean noise band level error {np.mean(np.abs(err_db)):.2f} dB"
    # shape correlation
    corr = np.corrcoef(N_meas_med[active], N_target[active, 0])[0, 1]
    assert corr > 0.8, f"envelope shape correlation {corr:.3f}"


# --------------------------------------------------------------------------- #
# residual: cancels pure harmonics; keeps broadband noise
# --------------------------------------------------------------------------- #
def test_residual_nulls_pure_harmonics():
    f0 = 200.0
    t = np.arange(int(SR * 1.0)) / SR
    amps = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1]
    y = sum(a * np.cos(2 * np.pi * (k + 1) * f0 * t) for k, a in enumerate(amps)).astype(np.float32)
    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))
    res = extract_residual(y, H, info["f0_per_frame"], sr=SR, n_fft=info["n_fft"], hop=512)
    red_db = 20 * np.log10(np.sqrt(np.mean(res**2)) / (np.sqrt(np.mean(y**2)) + 1e-12) + 1e-12)
    assert red_db < -14.0, f"pure harmonic residual only attenuated by {red_db:.1f} dB (should be < -14dB)"


def test_residual_preserves_high_band_noise():
    f0 = 200.0
    t = np.arange(int(SR * 1.0)) / SR
    harm = sum(
        a * np.cos(2 * np.pi * (k + 1) * f0 * t) for k, a in enumerate([1.0, 0.6, 0.4, 0.25])
    )  # harmonics <=800Hz
    # high-frequency noise (5-8kHz, far from the harmonics), made as high-band noise with synth_noise
    n_mels = 40
    _, centers = mel_filterbank(SR, 1024, n_mels)
    Nt = np.zeros((n_mels, 1), dtype=np.float32)
    Nt[(centers >= 5000) & (centers <= 8000), 0] = 0.3
    noise = synth_noise(Nt, SR, length=len(t), seed=3)
    y = (harm + noise).astype(np.float32)

    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=4)
    res = extract_residual(y, H, info["f0_per_frame"], sr=SR, n_fft=info["n_fft"], hop=512)
    # high-band energy should be mostly preserved (residual vs original noise)
    e_noise = _band_energy(noise, SR, 5000, 8000)
    e_res = _band_energy(res, SR, 5000, 8000)
    ratio = e_res / (e_noise + 1e-12)
    assert 0.5 < ratio < 2.0, f"high-band noise preservation ratio {ratio:.2f} (should be ~1)"
    # low-band harmonic energy should be removed substantially
    e_harm_in = _band_energy(harm, SR, 100, 900)
    e_harm_res = _band_energy(res, SR, 100, 900)
    assert e_harm_res / (e_harm_in + 1e-12) < 0.1, "low-band harmonics not removed effectively"


def test_attack_residual_length():
    res = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    atk = attack_residual(res, sr=SR, ms=60.0)
    assert atk.shape == (2646,), f"attack length {atk.shape} != 2646"


# --------------------------------------------------------------------------- #
# FR-3 formant preservation vs carry-over by index (chipmunk)
# --------------------------------------------------------------------------- #
def test_formant_preservation_vs_index():
    K = 30
    f0_ref = 200.0
    k = np.arange(1, K + 1)
    # fixed formant @ 1000Hz (amplitude determined only by absolute frequency)
    H = np.exp(-(((k * f0_ref - 1000.0) / 300.0) ** 2))[:, None].astype(np.float32)

    f0_tgt = 300.0  # shifted up
    H_form, _ = redirect_harmonics(H, f0_ref, f0_tgt, mode="formant")
    H_idx, _ = redirect_harmonics(H, f0_ref, f0_tgt, mode="index")

    peak_form_hz = (np.argmax(H_form[:, 0]) + 1) * f0_tgt
    peak_idx_hz = (np.argmax(H_idx[:, 0]) + 1) * f0_tgt

    # formant preservation: the peak stays at ~1000Hz (± half the target f0)
    assert abs(peak_form_hz - 1000.0) < f0_tgt, (
        f"formant mode peak moved to {peak_form_hz:.0f}Hz (should be ~1000)"
    )
    # carry-over by index: the peak moves up with f0 to ~1500Hz
    assert peak_idx_hz > 1300.0, f"index mode peak should move up, got {peak_idx_hz:.0f}Hz"


def test_formant_identity_at_same_pitch():
    K = 20
    f0 = 220.0
    H = np.random.default_rng(0).random((K, 3)).astype(np.float32)
    H_same, _ = redirect_harmonics(H, f0, f0, mode="formant")
    assert np.allclose(H_same, H, atol=1e-4), "redirection at the same f0 should be the identity"


def test_redirect_quasi_harmonic():
    """Quasi-harmonic: returns the original partial_freqs, amplitudes resampled by absolute frequency."""
    pf = np.array([660.0, 1335.0, 2025.0, 2740.0], dtype=np.float32)
    H = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.float32)[:, None]
    H_t, pf_t = redirect_harmonics(H, 660.0, 990.0, partial_freqs=pf, mode="formant")
    assert pf_t is not None and np.allclose(pf_t, pf)
    assert H_t.shape == H.shape


# --------------------------------------------------------------------------- #
# TimbreSnapshot serialisation round trip + tolerant loading
# --------------------------------------------------------------------------- #
def test_snapshot_roundtrip(tmp_path):
    H = np.random.default_rng(0).random((60, 50)).astype(np.float32)
    N = np.random.default_rng(1).random((40, 80)).astype(np.float32)
    atk = np.random.default_rng(2).standard_normal(2646).astype(np.float32)
    pf = np.linspace(660, 40000, 60).astype(np.float32)
    snap = make_snapshot(
        440.0,
        H,
        N,
        atk,
        partial_freqs=pf,
        vibrato={"rate_hz": 5.5, "depth_cents": 20.0},
        name="testglass",
        sr=SR,
    )
    p = save_snapshot(tmp_path / "s.npz", snap)
    out = load_snapshot(p)

    assert np.allclose(out["H_env"], H)
    assert np.allclose(out["N_env"], N)
    assert np.allclose(out["attack_res"], atk)
    assert np.allclose(out["partial_freqs"], pf)
    assert out["vibrato"]["rate_hz"] == pytest.approx(5.5)
    assert out["meta"]["name"] == "testglass"
    assert abs(out["f0_ref"] - 440.0) < 1e-6


def test_snapshot_none_partials(tmp_path):
    snap = make_snapshot(
        220.0,
        np.zeros((60, 10), np.float32),
        np.zeros((40, 10), np.float32),
        np.zeros(2646, np.float32),
        partial_freqs=None,
        vibrato=None,
        name="voice",
    )
    out = load_snapshot(save_snapshot(tmp_path / "v.npz", snap))
    assert out["partial_freqs"] is None
    assert out["vibrato"] is None


def test_detect_vibrato():
    """A known 5.5Hz / ~30cents vibrato should be detected; a steady f0 should give None."""
    from analysis.timbre import detect_vibrato

    dt = 0.01
    t = np.arange(0, 1.5, dt)
    f0_med = 220.0
    # 5.5Hz vibrato with 30 cents depth
    f0_vib = f0_med * 2 ** ((30.0 / 1200.0) * np.sin(2 * np.pi * 5.5 * t))
    v = detect_vibrato(f0_vib, t)
    assert v is not None
    assert abs(v["rate_hz"] - 5.5) < 0.7, f"rate={v['rate_hz']}"
    assert 20 < v["depth_cents"] < 45, f"depth={v['depth_cents']}"

    f0_steady = np.full_like(t, f0_med)
    assert detect_vibrato(f0_steady, t) is None


def test_snapshot_tolerant_loader(tmp_path):
    """An old npz with missing fields should load tolerantly (attack/N_env get defaults)."""
    p = tmp_path / "partial.npz"
    np.savez(p, H_env=np.ones((60, 5), np.float32), f0_ref=np.float64(330.0))
    out = load_snapshot(p)
    assert out["attack_res"].shape == (2646,)  # filled with defaults
    assert out["N_env"].shape[0] == 40
    assert out["meta"]["format_version"] == 1  # default meta
    assert out["partial_freqs"] is None
