"""Detectron2 framework wrapper."""

from typing import Any, Dict

import torch

from src.models.framework_adapter import Detector, DetectionOutput


class Detectron2Wrapper(Detector):
    """Detectron2 detector wrapper behind Detector interface."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.cfg = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            import detectron2
            from detectron2.config import get_cfg
            from detectron2.modeling import build_model
            from detectron2.checkpoint import DetectionCheckpointer
            self.detectron2 = detectron2
            self.get_cfg_fn = get_cfg
            self.build_model_fn = build_model
            self.DetectionCheckpointer = DetectionCheckpointer
        except ImportError:
            self.detectron2 = None

        if self.detectron2 is not None:
            model_cfg = self.config.get("model", {})
            config_path = model_cfg.get("config_path")
            checkpoint_path = model_cfg.get("checkpoint_path")

            if config_path:
                self.cfg = self._load_config(config_path)
            else:
                self.cfg = self._build_default_config(model_cfg)

            self.model = self.build_model_fn(self.cfg)
            if checkpoint_path:
                checkpointer = self.DetectionCheckpointer(self.model)
                checkpointer.load(checkpoint_path)

        self._initialized = True

    def _load_config(self, config_path: str) -> Any:
        from detectron2.config import get_cfg
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return cfg

    def _build_default_config(self, model_cfg: Dict[str, Any]) -> Any:
        from detectron2.config import get_cfg
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_cfg.get("num_classes", 16)
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER.IMS_PER_BATCH = model_cfg.get("batch_size", 4)
        cfg.SOLVER.MAX_ITER = model_cfg.get("max_iter", 10000)
        return cfg

    def train(self, train_loader, val_loader, cfg: Dict[str, Any]) -> str:
        self._ensure_initialized()
        if self.model is not None and self.cfg is not None:
            try:
                from detectron2.engine import DefaultTrainer
                trainer = DefaultTrainer(self.model, self.cfg)
                trainer.resume_or_load(resume=False)
                trainer.train()
                return "model_final.pth"
            except Exception:
                pass
        return "dummy_checkpoint.pth"

    def evaluate(self, data_loader, cfg: Dict[str, Any]) -> Dict[str, float]:
        self._ensure_initialized()
        if self.model is not None and self.cfg is not None:
            try:
                from detectron2.evaluation import COCOEvaluator
                evaluators = COCOEvaluator("val", self.cfg, False, output_dir="/tmp/eval")
                results = evaluators.evaluate()
                return {
                    "mAP": results.get("bbox_mAP", 0.0),
                    "mAP_50": results.get("bbox_mAP_50", 0.0),
                    "mAP_75": results.get("bbox_mAP_75", 0.0),
                }
            except Exception:
                pass
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    def predict(self, image: torch.Tensor) -> DetectionOutput:
        self._ensure_initialized()
        if self.model is not None:
            try:
                inputs = [{"image": image}]
                with torch.no_grad():
                    predictions = self.model(inputs)
                pred = predictions[0]
                fields = pred.get("instances", None)
                if fields is not None:
                    return DetectionOutput(
                        boxes=fields.get("pred_boxes", torch.zeros(0, 4)),
                        scores=fields.get("scores", torch.zeros(0)),
                        labels=fields.get("pred_classes", torch.zeros(0, dtype=torch.long)),
                    )
            except Exception:
                pass
        return DetectionOutput(
            boxes=torch.zeros(0, 4),
            scores=torch.zeros(0),
            labels=torch.zeros(0, dtype=torch.long),
        )

    def save_checkpoint(self, path: str) -> None:
        self._ensure_initialized()
        if self.model is not None:
            checkpointer = self.DetectionCheckpointer(self.model)
            checkpointer.save(path)

    def load_checkpoint(self, path: str) -> None:
        self._ensure_initialized()
        if self.model is not None:
            checkpointer = self.DetectionCheckpointer(self.model)
            checkpointer.load(path)
