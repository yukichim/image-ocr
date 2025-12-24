"""
OCRエンジンモジュール

複数のOCRエンジン（Tesseract, OnnxOCR, PaddleOCR）を統一インターフェースで提供:
- Tesseract: LSTM モードによる高精度認識
- OnnxOCR: PP-OCRv5 ONNX版（日本語対応・軽量）
- PaddleOCR: 高精度・方向検出付き

後方互換性のため、既存のAPIは維持しています。
"""

# 新しいエンジンシステムからの再エクスポート（後方互換性）
from .engines.base import (
    BoundingBox, OCRWord, OCRLine, OCRBlock, OCRResult
)
from .engines.tesseract_engine import (
    TesseractEngine, TesseractConfig, OCRMode, PageSegMode
)
from .engines import (
    EngineType, get_available_engines, create_engine
)

import numpy as np
from dataclasses import dataclass
from typing import Optional


# 後方互換性のためのエイリアス
@dataclass
class OCRConfig:
    """OCR設定（後方互換性のため維持）"""
    language: str = "jpn+eng"
    oem: OCRMode = OCRMode.LSTM
    psm: PageSegMode = PageSegMode.FULLY_AUTO
    confidence_threshold: float = 60.0
    custom_config: str = ""
    multi_scale_enabled: bool = False
    nms_iou_threshold: float = 0.5


class OCREngine:
    """
    OCRエンジンクラス（後方互換性ラッパー）
    
    新しいコードでは engines.create_engine() を使用してください。
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        
        # TesseractConfigに変換
        tesseract_config = TesseractConfig(
            language=self.config.language,
            oem=self.config.oem,
            psm=self.config.psm,
            confidence_threshold=self.config.confidence_threshold,
            custom_config=self.config.custom_config,
        )
        
        # 内部エンジンとしてTesseractを使用
        self._engine = TesseractEngine(config=tesseract_config)
    
    def recognize(self, image: np.ndarray, multi_scale_images: list = None) -> OCRResult:
        """
        画像からテキストを認識
        
        Args:
            image: 前処理済み画像
            multi_scale_images: マルチスケール画像リスト（現在は未使用）
            
        Returns:
            OCRResult: 認識結果
        """
        return self._engine.recognize(image)


def recognize_image(image: np.ndarray, 
                   config: Optional[OCRConfig] = None) -> OCRResult:
    """
    OCR認識のショートカット関数（後方互換性）
    
    Args:
        image: 前処理済み画像
        config: OCR設定（省略時はデフォルト）
        
    Returns:
        OCRResult: 認識結果
    """
    engine = OCREngine(config)
    return engine.recognize(image)


# 新しいAPIのエクスポート
__all__ = [
    # 後方互換性
    'OCRConfig',
    'OCREngine',
    'OCRMode',
    'PageSegMode',
    'BoundingBox',
    'OCRWord',
    'OCRLine',
    'OCRBlock',
    'OCRResult',
    'recognize_image',
    # 新しいAPI
    'EngineType',
    'get_available_engines',
    'create_engine',
    'TesseractEngine',
    'TesseractConfig',
]
