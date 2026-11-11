# Model Cards

## DDSP Timbre Models

All DDSP models: ~500K params, harmonic-plus-noise decoder, GRU backbone, 16 kHz @ 250 fps.
Training loss: multi-scale spectral (MSS).

### ddsp_urmp_vn_12k.pt

- **Source**: URMP violin, single instrument
- **Training data**: ~45 s mono 16 kHz (URMP dataset; license not formally stated, requires registration and citation)
- **Steps**: 12,000
- **License**: Model weights CC BY-NC 4.0; training data license pending clarification with URMP authors
- **SHA-256**: `d63c58185c753f096033c3d941725acbdebd1822455297b69c2fb7516dfd9f3d`

### ddsp_urmp_tpt_12k.pt

- **Source**: URMP trumpet, single instrument
- **Training data**: ~40 s mono 16 kHz (URMP dataset; license not formally stated, requires registration and citation)
- **Steps**: 12,000
- **License**: Model weights CC BY-NC 4.0; training data license pending clarification with URMP authors
- **SHA-256**: `ed653891fc0ed21151ef12f4495d2b9b01ef8239be08af2d667b42e3e853b831`

### ddsp_glass.pt

- **Source**: Wine glass, self-recorded (twelve takes across three glasses + one fishbowl)
- **Training data**: ~82 s mono 16 kHz (self-recorded, CC BY-NC 4.0)
- **Steps**: 12,000
- **License**: CC BY-NC 4.0
- **SHA-256**: `b718be519aee9e87271ad7ebbe8810b76179f55a8555d6189c02b88cc396151f`

### ddsp_guitarset.pt

- **Source**: GuitarSet acoustic guitar, single instrument
- **Training data**: ~60 s mono 16 kHz (GuitarSet, CC BY-NC 4.0)
- **Steps**: 12,000
- **License**: Model weights CC BY-NC 4.0; training data CC BY-NC 4.0
- **SHA-256**: `bdc153946fec299a07943950fa0e014c5663e481e0c8d2e5fedac6f0c5fb012e`

## Behavior Distillation Model

### behavior_distilled.pt

- **Architecture**: ResidualBehaviorNet (MLP hidden=256, ~200K params)
- **Teacher**: MIDI-DDSP expression + synthesis generators
- **Training data**: 7,582 note-level trajectory pairs from MIDI-DDSP teacher exports (URMP violin, trumpet, GuitarSet guitar)
- **Best epoch**: 21 (seed 42)
- **License**: Model weights CC BY-NC 4.0; teacher model Apache 2.0 (MIDI-DDSP)
- **SHA-256**: `238df55dadd812f4cea34dfefb9000d64d8eeaeba9855e79cabaf4cea8be1a8c`

## Vibrato Model

### vibrato_model.pt

- **Architecture**: MLP regressor 12 → 64 → 32 → 8 (outputs: rate, depth, onset_ratio, envelope[5])
- **Training data**: Vibrato-labeled notes from URMP violin, URMP cello, and GuitarSet guitar trajectories (plain notes included with depth=0 target)
- **Steps**: 2,000
- **License**: CC BY-NC 4.0
- **SHA-256**: `79b6f544502e8b8d893ae9d18ba5211cf5d866c04ce4b64934ad918304635405`

## Behavior Libraries (npz)

**Distribution policy (decided 2026-10-27).** The two URMP-derived libraries (`urmp_vn_trajectories.npz`, `urmp_tpt_trajectories.npz`) are **not redistributed**: URMP ships no standard license and requires every user to register and cite the dataset, and its terms do not address derived data. These files stay in this repository only to reproduce the ISMIR LBD 2026 numbers and are excluded from Zenodo / GitHub Release packages. To rebuild them, register for URMP, download the separated stems with their `Notes_*.txt` annotations, and run `tools/extract_trajectories_urmp.py --inst vn|tpt --urmp <URMP root> -o <out.npz>`. This is the script that built the shipped files; `tests/test_urmp_extraction_rules.py` checks that its rules reproduce 100% of the stored labels and context features. `glass_style.npz` (self-recorded, CC BY-NC 4.0) is the library we publish. See `docs/behavior_library_format.md` for the file format.

### urmp_vn_trajectories.npz

- **Source**: URMP violin performance trajectories
- **Distribution**: not redistributed (repo-internal, paper reproduction only); see policy above
- **Format**: Note-level 250 fps pitch/energy/brightness/noise trajectories + articulation labels
- **License**: Derived trajectories (no audio); URMP source license not formally stated (requires registration and citation)
- **SHA-256**: `0f081f432874392fe8e3ca8e4558fd2be0eb6fb916e9bd04654e031a43a157fe`
- **SHA-256 (pre-format-v1.0 file, LBD submission)**: `fe77bfaed5a0f6c9fb7794df44a412828c17d303347b42fce3b3b9a5790dc25c`

### urmp_tpt_trajectories.npz

- **Source**: URMP trumpet performance trajectories
- **Distribution**: not redistributed (repo-internal, paper reproduction only); see policy above
- **License**: Derived trajectories (no audio); URMP source license not formally stated (requires registration and citation)
- **SHA-256**: `d80a59f5ee0c01e1394d52a5f31549d358531cc9c4b6f20a66e2d74cbc6a042c`
- **SHA-256 (pre-format-v1.0 file, LBD submission)**: `fe76134694393f5e764a4dd671f78dd3cd025dea4e212d0acbdeb903e36f7ff2`

### guitarset_trajectories.npz

- **Source**: GuitarSet acoustic guitar performance trajectories
- **License**: Derived trajectories (no audio); source CC BY-NC 4.0
- **SHA-256**: `ac41f33d253d86bbbf6406f30cb186203b0f59bd5bb52a67631444a24283275b`
- **SHA-256 (pre-format-v1.0 file, LBD submission)**: `7c04c47fa559c7701f0f7934b7a0a61c8029807bf105f75c226f8d299ad854c2`

### glass_style.npz

- **Source**: Self-recorded wine glass performance
- **Distribution**: published (Zenodo / GitHub Release, see README "Behavior libraries")
- **License**: CC BY-NC 4.0
- **SHA-256**: `198675c22031590bf2115e2324bc8bd49b4901a579ffa9505f687003be9f1bcf`
- **SHA-256 (pre-format-v1.0 file, LBD submission)**: `d92eefbdaa704f4577a4b2ec584aa905d0830d773b6250cd38b90f6c127c2077`
