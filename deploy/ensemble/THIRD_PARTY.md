# Model and dependency notices

Public upstream model cards checked on 2026-10-18. Files are fetched at the revisions and SHA-256 values in `model-manifest.json`; model weights are not redistributed in this source repository.

| Component | Upstream | Declared license |
|---|---|---|
| X-CLIP base patch32 weights | https://huggingface.co/microsoft/xclip-base-patch32 | MIT |
| X-CLIP upstream implementation | https://github.com/microsoft/VideoX | MIT |
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | Apache-2.0 |
| YOLO11n weights | https://huggingface.co/Ultralytics/YOLO11 | AGPL-3.0 |
| Ultralytics runtime and bundled ByteTrack integration | https://github.com/ultralytics/ultralytics/tree/v8.3.221 | AGPL-3.0 |
| Transformers | https://github.com/huggingface/transformers | Apache-2.0 |
| PyTorch / torchvision | https://github.com/pytorch/pytorch / https://github.com/pytorch/vision | BSD-style |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause |

The AGPL is not a noncommercial license, but its source availability and other copyleft conditions matter, including for network use and combined applications. This experiment does not purchase an Ultralytics enterprise license or relicense its weights. Preserve upstream notices and comply with the applicable licenses when distributing or operating a combined solution. A closed proprietary deployment needs a separate licensing decision.

The initially considered `MCG-NJU/videomae-base-finetuned-kinetics` checkpoint declares **CC-BY-NC-4.0**, so it is not downloaded or used here. The name `VideoMAEImageProcessor` in `action_expert.py` is the generic Transformers preprocessing class used by X-CLIP's processor; it does not load VideoMAE weights.

This folder is an original cash-video analysis pipeline assembled from pretrained components. No upstream component is a ready-made ForteBank violation detector. The FAQ permits pretrained commercially usable open-source components; this statement is an interpretation of the supplied FAQ, not an organizer's individual approval.
