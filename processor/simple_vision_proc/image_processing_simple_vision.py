# image_processing_simple_vision.py

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from transformers.image_processing_base import (
    ImageProcessingMixin,
)
from transformers.utils.generic import TensorType


class SimpleVisionImageProcessor(
    ImageProcessingMixin,
):
    """
    Custom ImageProcessor
    (ImageProcessingMixin 기반)
    - images → pixel_values (B, C, H, W)
    - save_pretrained() 호출 시
      preprocessor_config.json 자동 생성
    """

    # HuggingFace 관례:
    # 모델에 전달할 입력 key 이름을 지정
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: int = 224,
        do_resize: bool = True,
        do_normalize: bool = True,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        **kwargs: Any,
    ):
        # ── 반드시 super().__init__() 호출 ──
        # ImageProcessingMixin 내부 초기화 +
        # kwargs 처리를 위해 필수임.
        super().__init__(**kwargs)

        # ── flat attributes로 저장 ──
        # self.xxx 형태 + JSON 직렬화 가능 타입
        #   부모의 기본 to_dict()가
        #   알아서 직렬화해줌
        self.size = int(size)
        self.do_resize = bool(do_resize)
        self.do_normalize = bool(do_normalize)
        self.image_mean = (
            image_mean
            if image_mean is not None
            else [0.485, 0.456, 0.406]
        )
        self.image_std = (
            image_std
            if image_std is not None
            else [0.229, 0.224, 0.225]
        )

    # ────────────────────────────
    # 내부 헬퍼 메서드
    # ────────────────────────────

    def _ensure_pil(
        self,
        img: Image.Image | np.ndarray,
    ) -> Image.Image:
        """입력을 PIL Image로 변환."""
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 2:  # grayscale→3ch
                arr = np.stack(
                    [arr, arr, arr], axis=-1,
                )
            if arr.dtype != np.uint8:
                arr = (
                    np.clip(arr, 0, 255)
                    .astype(np.uint8)
                )
            return Image.fromarray(arr)
        raise TypeError(
            f"Unsupported image type:"
            f" {type(img)}"
        )

    def _resize(
        self, img: Image.Image,
    ) -> Image.Image:
        """self.size x self.size 로 리사이즈."""
        if not self.do_resize:
            return img
        return img.resize(
            (self.size, self.size),
            resample=Image.BILINEAR,
        )

    def _to_chw_float01(
        self, img: Image.Image,
    ) -> np.ndarray:
        """PIL Image → (C,H,W) float32 [0,1]."""
        x = (  # HWC, [0,1]
            np.asarray(img, dtype=np.float32)
            / 255.0
        )
        return np.transpose(x, (2, 0, 1))  # CHW

    def _normalize(
        self, x: np.ndarray,
    ) -> np.ndarray:
        """ImageNet mean/std 기반 정규화."""
        if not self.do_normalize:
            return x
        mean = np.asarray(
            self.image_mean, dtype=np.float32,
        )[:, None, None]
        std = np.asarray(
            self.image_std, dtype=np.float32,
        )[:, None, None]
        return (x - mean) / std

    # ────────────────────────────
    # __call__: 실제 전처리 진입점
    # ────────────────────────────

    def __call__(
        self,
        images: (
            Image.Image
            | np.ndarray
            | list[Image.Image | np.ndarray]
        ),
        return_tensors: (
            str | TensorType | None
        ) = "pt",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Parameters
        ----------
        images :
            단일 이미지 또는 이미지 리스트
        return_tensors :
            "pt" (PyTorch) 또는 "np" (NumPy)

        Returns
        -------
        dict with "pixel_values":
            shape (B, C, H, W)
        """
        if not isinstance(images, list):
            images = [images]

        batch = []
        for im in images:
            im = (
                self._ensure_pil(im)
                .convert("RGB")
            )
            im = self._resize(im)
            x = self._to_chw_float01(im)
            x = self._normalize(x)
            batch.append(x)

        pixel_values = (
            np.stack(batch, axis=0)
            .astype(np.float32)
        )

        if return_tensors in (
            "pt", TensorType.PYTORCH,
        ):
            pixel_values = (
                torch.from_numpy(pixel_values)
            )
        elif (
            return_tensors
            in ("np", TensorType.NUMPY)
            or return_tensors is None
        ):
            pass
        else:
            raise ValueError(
                "Unsupported return_tensors:"
                f" {return_tensors}"
            )

        return {"pixel_values": pixel_values}


# ── AutoImageProcessor 등록 ──
# 이 한 줄을 추가하면,
# config.json의 auto_map에
# "AutoImageProcessor" 매핑이 자동 포함됨.
SimpleVisionImageProcessor \
    .register_for_auto_class(
        "AutoImageProcessor",
    )