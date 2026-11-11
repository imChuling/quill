"""synth/voice.py 单测 —— ADSR 状态机、glide 插值、共振峰保持、渲染正确性。"""

import numpy as np
import pytest

from synth.voice import (
    _ATK,
    _DEC,
    _DONE,
    _REL,
    _SILENCE,
    _SUS,
    Voice,
    set_humanize,
    set_liveness,
)
from tests.helpers import ADSR_FAST as _ADSR
from tests.helpers import RT_CFG as _RT
from tests.helpers import SR, make_test_instrument
from tests.helpers import VEL_CFG as _VEL

K = 8


def _inst(f0_ref=220.0, k=K):
    return make_test_instrument(K=k, f0_ref=f0_ref)


@pytest.fixture(autouse=True)
def _disable_humanize():
    set_humanize(0.0, 0.0)
    set_liveness(0.0, 0.0, 0.0)
    yield
    set_humanize(6.0, 0.05)
    set_liveness(0.85, 0.0, 0.8)


# ===== ADSR 状态机 =====


class TestADSR:
    def _make(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        return v

    def test_note_on_starts_attack(self):
        v = self._make()
        assert v.stage == _ATK
        assert v.env == 0.0

    def test_attack_reaches_one(self):
        v = self._make()
        n = int(0.01 * SR) + 1
        e0, e1 = v._advance_adsr(n)
        assert e1 >= 1.0
        assert v.stage == _DEC

    def test_decay_reaches_sustain(self):
        v = self._make()
        v._advance_adsr(int(0.01 * SR) + 1)
        n = int(0.02 * SR) + 1
        v._advance_adsr(n)
        assert v.env <= _ADSR["sustain"]
        assert v.stage == _SUS

    def test_sustain_holds(self):
        v = self._make()
        v._advance_adsr(int(0.01 * SR) + 1)
        v._advance_adsr(int(0.02 * SR) + 1)
        e0, e1 = v._advance_adsr(1000)
        assert e1 == pytest.approx(_ADSR["sustain"])
        assert v.stage == _SUS

    def test_release_to_done(self):
        v = self._make()
        v._advance_adsr(int(0.01 * SR) + 1)
        v._advance_adsr(int(0.02 * SR) + 1)
        v.note_off()
        assert v.stage == _REL
        n = int(0.03 * SR) + 100
        v._advance_adsr(n)
        assert v.stage == _DONE
        assert v.active is False
        assert v.env == 0.0

    def test_steal_fast_release(self):
        v = self._make()
        v._advance_adsr(int(0.01 * SR) + 1)
        v.steal(3.0)
        assert v.stage == _REL
        n = int(0.003 * SR) + 100
        v._advance_adsr(n)
        assert v.stage == _DONE

    def test_silence_threshold(self):
        assert pytest.approx(10 ** (-120 / 20.0)) == _SILENCE


# ===== glide 插值 =====


class TestGlide:
    def test_identity_when_no_glide(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        f = v._glide_f0(0)
        assert f == pytest.approx(v.f0_target)

    def test_linear_glide(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        f_from = v.f0_target
        f_to = 880.0
        dur = 1000
        v.glide_to(f_to, dur)
        mid = v._glide_f0(dur // 2)
        geo_mean = np.sqrt(f_from * f_to)
        assert mid == pytest.approx(geo_mean, rel=0.01)

    def test_glide_endpoints(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v.glide_to(880.0, 1000)
        assert v._glide_f0(0) == pytest.approx(v._glide_from)
        assert v._glide_f0(1000) == pytest.approx(880.0)
        assert v._glide_f0(2000) == pytest.approx(880.0)

    def test_glide_with_curve(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        curve = np.array([0.0, 0.0, 1.0, 1.0])
        v.glide_to(880.0, 1000, curve=curve)
        early = v._glide_f0(100)
        late = v._glide_f0(900)
        assert early < late


# ===== 共振峰保持 =====


class TestFormantPreservation:
    def test_gain_k_shape(self):
        v = Voice(K, SR)
        inst = _inst()
        v.note_on(69, 100, inst, _ADSR, _VEL, _RT, age=0)
        assert v.gain_k.shape == (K,)
        assert np.all(np.isfinite(v.gain_k))

    def test_higher_pitch_shifts_gain(self):
        v = Voice(K, SR)
        inst = _inst(f0_ref=220.0)
        v.note_on(69, 100, inst, _ADSR, _VEL, _RT, age=0)
        g_a4 = v._compute_gain_k(440.0).copy()
        g_a5 = v._compute_gain_k(880.0).copy()
        assert not np.allclose(g_a4, g_a5, atol=1e-6)

    def test_rolloff_beyond_support(self):
        v = Voice(K, SR)
        inst = _inst(f0_ref=220.0)
        v.note_on(69, 100, inst, _ADSR, _VEL, _RT, age=0)
        g = v._compute_gain_k(220.0)
        assert g[-1] <= g[0]

    def test_pruning_below_minus60(self):
        v = Voice(K, SR)
        inst = _inst(f0_ref=220.0)
        v.note_on(69, 100, inst, _ADSR, _VEL, _RT, age=0)
        g = v._compute_gain_k(220.0)
        gmax = np.max(np.abs(g))
        small = np.abs(g) < 1e-3 * gmax
        assert np.all(g[small] == 0.0)


# ===== note_on / note_off =====


class TestNoteLifecycle:
    def test_note_on_activates(self):
        v = Voice(K, SR)
        assert v.active is False
        v.note_on(60, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        assert v.active is True
        assert v.note == 60

    def test_f0_target_from_midi(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        assert v.f0_target == pytest.approx(440.0)
        v2 = Voice(K, SR)
        v2.note_on(60, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        assert v2.f0_target == pytest.approx(261.626, rel=0.001)

    def test_schroeder_phase_not_zero(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        assert not np.all(v.phase == 0.0)

    def test_note_off_enters_release(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v._advance_adsr(int(0.05 * SR))
        v.note_off()
        assert v.stage == _REL

    def test_velocity_mapping(self):
        v1 = Voice(K, SR)
        v1.note_on(69, 127, _inst(), _ADSR, _VEL, _RT, age=0)
        v2 = Voice(K, SR)
        v2.note_on(69, 40, _inst(), _ADSR, _VEL, _RT, age=0)
        assert v1.vel_loud > v2.vel_loud


# ===== pitch bend =====


class TestPitchBend:
    def test_set_bend(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v.set_bend(2.0)
        assert v._bend_target == pytest.approx(2.0 ** (2.0 / 12.0))

    def test_init_bend_immediate(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v.init_bend(2.0)
        expected = 2.0 ** (2.0 / 12.0)
        assert v.bend == pytest.approx(expected)
        assert v.bend_prev == pytest.approx(expected)


# ===== vibrato =====


class TestVibrato:
    def test_set_vibrato_resets_age(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v._vib_age = 5000
        v.set_vibrato(0.0)
        v.set_vibrato(30.0)
        assert v._vib_age == 0

    def test_zero_depth_no_reset(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v._vib_age = 5000
        v.set_vibrato(0.0)
        assert v._vib_age == 5000


# ===== render_into 集成 =====


class TestRenderInto:
    def test_produces_nonzero_output(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.zeros(512, dtype=np.float64)
        v.render_into(block, 512)
        assert np.max(np.abs(block)) > 0

    def test_inactive_voice_silent(self):
        v = Voice(K, SR)
        block = np.zeros(512, dtype=np.float64)
        v.render_into(block, 512)
        assert np.all(block == 0.0)

    def test_after_release_becomes_silent(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.zeros(512, dtype=np.float64)
        for _ in range(20):
            v.render_into(block, 512)
        v.note_off()
        for _ in range(200):
            block[:] = 0.0
            v.render_into(block, 512)
        assert v.stage == _DONE
        assert v.active is False

    def test_output_frequency_matches_note(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        n = SR
        block = np.zeros(n, dtype=np.float64)
        for off in range(0, n, 512):
            chunk = min(512, n - off)
            v.render_into(block[off : off + chunk], chunk)
        spec = np.abs(np.fft.rfft(block))
        freqs = np.fft.rfftfreq(n, 1.0 / SR)
        peak_hz = freqs[np.argmax(spec[1:]) + 1]
        assert peak_hz == pytest.approx(440.0, abs=5.0)

    def test_accumulates_into_block(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.ones(512, dtype=np.float64) * 0.5
        v.render_into(block, 512)
        assert not np.allclose(block, 0.5)

    def test_multiple_blocks_phase_continuous(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        chunks = []
        for _ in range(4):
            b = np.zeros(512, dtype=np.float64)
            v.render_into(b, 512)
            chunks.append(b)
        joined = np.concatenate(chunks)
        diffs = np.abs(np.diff(joined))
        assert np.max(diffs) < 1.0


# ===== glide_to 演奏法 =====


class TestGlideTo:
    def test_legato_resets_attack(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v.attack_pos = 999
        v.glide_to(880.0, 4410, kind="legato", vel=90)
        assert v.attack_pos == 0

    def test_slide_sets_glide_strength(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v.glide_to(880.0, 4410, kind="slide", vel=90)
        assert v._glide_stren > 0.0

    def test_gain_crossfade_set(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v.glide_to(880.0, 4410)
        assert v._gain_from is not None
        assert v._gain_to is not None


# ===== D2: ADSR 采样级精确时刻 =====


class TestADSRSamplePrecision:
    """D2: 四段转换时刻与电平精确到采样级。"""

    def test_attack_exact_sample_count(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        atk_samples = int(_ADSR["attack_ms"] / 1000.0 * SR)
        v._advance_adsr(atk_samples - 1)
        assert v.stage == _ATK
        assert v.env < 1.0
        v._advance_adsr(1)
        assert v.stage == _DEC
        assert v.env >= 1.0

    def test_decay_exact_endpoint(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        atk_n = int(_ADSR["attack_ms"] / 1000.0 * SR) + 1
        v._advance_adsr(atk_n)
        dec_n = int(_ADSR["decay_ms"] / 1000.0 * SR)
        v._advance_adsr(dec_n)
        assert v.stage == _SUS
        assert v.env == pytest.approx(_ADSR["sustain"], abs=0.01)

    def test_release_reaches_silence(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v._advance_adsr(int(0.01 * SR) + 1)
        v._advance_adsr(int(0.02 * SR) + 1)
        v.note_off()
        rel_n = int(_ADSR["release_ms"] / 1000.0 * SR) + 100
        v._advance_adsr(rel_n)
        assert v.stage == _DONE
        assert v.env == 0.0
        assert v.active is False

    def test_adsr_envelope_monotonic_attack(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        prev = 0.0
        for _ in range(10):
            _, e = v._advance_adsr(int(0.001 * SR))
            if v.stage == _ATK:
                assert e >= prev
                prev = e


class TestStealTiming:
    """D2: 偷音在 steal_ms 内衰减到 0。"""

    def test_steal_3ms(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v._advance_adsr(int(0.05 * SR))
        v.steal(3.0)
        assert v.stage == _REL
        n = int(0.003 * SR) + 10
        v._advance_adsr(n)
        assert v.stage == _DONE
        assert v.active is False

    def test_steal_5ms(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v._advance_adsr(int(0.05 * SR))
        v.steal(5.0)
        n = int(0.005 * SR) + 10
        v._advance_adsr(n)
        assert v.stage == _DONE


# ===== D2: glide 精确 log 域 =====


class TestGlidePrecision:
    """D2: glide log 域曲线端点与中点精确。"""

    def test_glide_geometric_midpoint(self):
        v = Voice(K, SR)
        v.note_on(60, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        f_from = v.f0_target
        f_to = f_from * 2.0
        dur = 10000
        v.glide_to(f_to, dur)
        mid = v._glide_f0(dur // 2)
        geo_mean = np.sqrt(f_from * f_to)
        assert mid == pytest.approx(geo_mean, rel=1e-6)

    def test_glide_quarter_point(self):
        v = Voice(K, SR)
        v.note_on(60, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        f_from = v.f0_target
        f_to = f_from * 4.0
        dur = 10000
        v.glide_to(f_to, dur)
        q = v._glide_f0(dur // 4)
        expected = f_from * (f_to / f_from) ** 0.25
        assert q == pytest.approx(expected, rel=1e-6)

    def test_glide_clamped_past_end(self):
        v = Voice(K, SR)
        v.note_on(60, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        v.glide_to(880.0, 1000)
        assert v._glide_f0(2000) == pytest.approx(880.0)
        assert v._glide_f0(1000) == pytest.approx(880.0)


# ===== D2: legato/slide 相位连续性 =====


class TestLegatoPhaseContinuity:
    """D2: legato/slide 不重置相位。"""

    def test_legato_preserves_phase(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.zeros(2048, dtype=np.float64)
        v.render_into(block, 2048)
        phase_before = v.phase.copy()
        v.glide_to(880.0, 4410, kind="legato", vel=90)
        assert np.allclose(v.phase, phase_before)

    def test_slide_preserves_phase(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.zeros(2048, dtype=np.float64)
        v.render_into(block, 2048)
        phase_before = v.phase.copy()
        v.glide_to(880.0, 4410, kind="slide", vel=90)
        assert np.allclose(v.phase, phase_before)

    def test_legato_no_adsr_reset(self):
        v = Voice(K, SR)
        v.note_on(69, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.zeros(2048, dtype=np.float64)
        for _ in range(10):
            v.render_into(block, 2048)
        assert v.stage in (_DEC, _SUS)
        env_before = v.env
        v.glide_to(880.0, 4410, kind="legato", vel=90)
        assert v.stage in (_DEC, _SUS)
        assert v.env == pytest.approx(env_before)


# ===== D2: gain_k 沿 glide 无跳变 =====


class TestGainCrossfade:
    """D2: gain_k 沿 glide 渐变无跳变。"""

    def test_gain_transitions_smoothly(self):
        v = Voice(K, SR)
        v.note_on(60, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.zeros(512, dtype=np.float64)
        for _ in range(5):
            v.render_into(block, 512)
        gain_before = v.gain_k.copy()
        v.glide_to(880.0, 4410, kind="legato", vel=90)
        gains = [gain_before.copy()]
        for _ in range(20):
            block[:] = 0.0
            v.render_into(block, 512)
            gains.append(v.gain_k.copy())
        for i in range(1, len(gains)):
            diff = np.max(np.abs(gains[i] - gains[i - 1]))
            assert diff < 0.5, f"gain jump at step {i}: max diff={diff}"

    def test_gain_reaches_target(self):
        v = Voice(K, SR)
        v.note_on(60, 100, _inst(), _ADSR, _VEL, _RT, age=0)
        block = np.zeros(512, dtype=np.float64)
        for _ in range(5):
            v.render_into(block, 512)
        v.glide_to(880.0, 4410, kind="legato", vel=90)
        target = v._gain_to.copy()
        for _ in range(50):
            block[:] = 0.0
            v.render_into(block, 512)
        assert np.allclose(v.gain_k, target, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
