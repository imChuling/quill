"""Modal 部署:Quill 云端渲染服务(take_schema_v1)。

跑: modal deploy deploy/modal_app.py
出: https://<workspace>--quill-render-serve.modal.run

镜像只带渲染路径需要的东西:neural/ + analysis/ + ui/ 源码与四个音色的
checkpoint(共 ~25MB)。CPU 渲染(DDSP 500K 参数,短 region 秒级),不要 GPU。
容器无状态:TakeStore 落 /tmp,幂等缓存仅在 warm 容器内有效——对
"render→轮询→下载"这个窗口足够。
"""
import modal

QUILL = "/root/quill"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.4.*", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy", "scipy", "soundfile", "fastapi[standard]", "pretty_midi", "librosa", "pyyaml")
    .env({"QUILL_WEB_DIR": f"{QUILL}/webdist",
          "QUILL_ASSETS_DIR": "/assets_store",
          "QUILL_VOLUME_NAME": "quill-assets"})
    .add_local_dir("../neural", f"{QUILL}/neural")
    .add_local_dir("../analysis", f"{QUILL}/analysis")
    .add_local_dir("../ui", f"{QUILL}/ui")
    .add_local_dir("../runtime", f"{QUILL}/runtime")
    .add_local_dir("../synth", f"{QUILL}/synth")
    .add_local_file("../snapshot.py", f"{QUILL}/snapshot.py")
    .add_local_file("../quill_config.py", f"{QUILL}/quill_config.py")
    .add_local_file("../config.yaml", f"{QUILL}/config.yaml")
    .add_local_file("../tools/extract_style.py", f"{QUILL}/tools/extract_style.py")
    .add_local_file("../checkpoints/ddsp_urmp_vn_12k.pt",
                    f"{QUILL}/checkpoints/ddsp_urmp_vn_12k.pt")
    .add_local_file("../checkpoints/ddsp_urmp_tpt_12k.pt",
                    f"{QUILL}/checkpoints/ddsp_urmp_tpt_12k.pt")
    .add_local_file("../checkpoints/ddsp_guitarset.pt",
                    f"{QUILL}/checkpoints/ddsp_guitarset.pt")
    .add_local_file("../checkpoints/ddsp_glass.pt",
                    f"{QUILL}/checkpoints/ddsp_glass.pt")
    .add_local_file("../checkpoints/urmp_vn_trajectories.npz",
                    f"{QUILL}/checkpoints/urmp_vn_trajectories.npz")
    .add_local_file("../checkpoints/urmp_tpt_trajectories.npz",
                    f"{QUILL}/checkpoints/urmp_tpt_trajectories.npz")
    .add_local_file("../checkpoints/guitarset_trajectories.npz",
                    f"{QUILL}/checkpoints/guitarset_trajectories.npz")
    .add_local_file("../checkpoints/glass_style.npz",
                    f"{QUILL}/checkpoints/glass_style.npz")
    # Plan D 蒸馏行为学生(MIDI-DDSP 教师 → Quill 行为残差),网页可切换对比
    .add_local_file("../checkpoints/behavior_distilled.pt",
                    f"{QUILL}/checkpoints/behavior_distilled.pt")
    # companion 静态页(同源托管):deploy 前先构建 ——
    #   cd companion && VITE_QUILL_API="" \
    #     VITE_REDIRECT=https://chuling-li-cs--quill-render-serve.modal.run/ \
    #     npx vite build
    .add_local_dir("../companion/dist", f"{QUILL}/webdist")
)

app = modal.App("quill-render")
assets_vol = modal.Volume.from_name("quill-assets", create_if_missing=True)


@app.function(
    image=image,
    cpu=2.0,
    memory=1024,
    min_containers=0,          # demo 当天可临时调 1 消除冷启动
    scaledown_window=300,
    timeout=180,
    volumes={"/assets_store": assets_vol},
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def serve():
    import sys
    sys.path.insert(0, QUILL)
    from ui.render_service import app as fastapi_app
    return fastapi_app
