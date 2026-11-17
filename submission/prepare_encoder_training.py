"""打包 Kaggle Encoder 训练代码；只创建本地目录，不上传、不登录。"""

import argparse
import base64
import json
from pathlib import Path


def build_standalone(root, destination):
    """把运行所需源码嵌入单文件，避免 Kaggle script Kernel 丢失兄弟目录。"""
    files = [
        "algorithms/__init__.py",
        "algorithms/baseline/__init__.py",
        "algorithms/baseline/preprocess.py",
        "algorithms/transformer/__init__.py",
        "algorithms/transformer/dataset.py",
        "algorithms/transformer/losses.py",
        "algorithms/transformer/model.py",
        "configs/transformer.json",
    ]
    bundle = {}
    for relative in files:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"打包文件不存在：{path}")
        bundle[relative] = base64.b64encode(path.read_bytes()).decode("ascii")
    bundle_literal = json.dumps(bundle, ensure_ascii=True, separators=(",", ":"))
    trainer = (root / "submission/kaggle/train_encoder.py").read_text(encoding="utf-8")
    trainer_literal = base64.b64encode(trainer.encode("utf-8")).decode("ascii")
    wrapper = f'''"""自包含 Kaggle Encoder 训练脚本；由 prepare_encoder_training.py 生成。"""
import base64
import runpy
import tempfile
from pathlib import Path
import sys

_BUNDLE = {bundle_literal}
_TRAINER = "{trainer_literal}"
_root = Path("/kaggle/working/_casmi_encoder_bundle") if Path("/kaggle/working").exists() else Path(tempfile.gettempdir()) / "casmi_encoder_bundle"
_root.mkdir(parents=True, exist_ok=True)
for _relative, _encoded in _BUNDLE.items():
    _target = _root / _relative
    _target.parent.mkdir(parents=True, exist_ok=True)
    _target.write_bytes(base64.b64decode(_encoded))
_trainer = _root / "train_encoder.py"
_trainer.write_bytes(base64.b64decode(_TRAINER))
sys.path.insert(0, str(_root))
runpy.run_path(str(_trainer), run_name="__main__")
'''
    destination.write_text(wrapper, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="新的构建目录")
    parser.add_argument("--username", default="YOUR_USERNAME")
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("输出目录已存在，请使用新的构建目录。")
    notebook = out / "notebook"
    notebook.mkdir(parents=True)
    build_standalone(root, notebook / "train_encoder.py")
    metadata = dict(id=args.username + "/casmi-encoder-training",
                    title="CASMI Encoder Training", code_file="train_encoder.py",
                    language="python", kernel_type="script", is_private=True,
                    enable_gpu=True, enable_internet=False, machine_shape=args.machine_shape,
                    dataset_sources=[], competition_sources=["enveda-CASMI26-molecule-id-mass-spectra"],
                    kernel_sources=[], model_sources=[])
    (notebook / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared Kaggle Encoder training package: {notebook}")
    print("Upload with: kaggle kernels push -p", notebook)


if __name__ == "__main__":
    main()
