"""
OnnxOCRエンジン

PP-OCRv5ベースのONNX OCRエンジン:
- 日本語・中国語・英語対応
- 方向分類（use_angle_cls）対応
- PaddlePaddle不要の軽量実装
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from .base import (
    BaseOCREngine, BoundingBox, OCRWord, OCRResult
)


# OnnxOCRのパスを追加
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "OnnxOCR"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))


@dataclass
class OnnxOCRConfig:
    """OnnxOCR設定"""
    use_angle_cls: bool = True   # 方向分類を使用
    use_gpu: bool = False        # GPU使用（CPU版なのでFalse）
    drop_score: float = 0.5      # 信頼度閾値


class OnnxOCREngine(BaseOCREngine):
    """OnnxOCR（PP-OCRv5 ONNX）エンジン"""
    
    def __init__(self, config: Optional[OnnxOCRConfig] = None, **kwargs):
        self.config = config or OnnxOCRConfig()
        
        # kwargsからの設定上書き
        if 'use_angle_cls' in kwargs:
            self.config.use_angle_cls = kwargs['use_angle_cls']
        if 'use_gpu' in kwargs:
            self.config.use_gpu = kwargs['use_gpu']
        if 'drop_score' in kwargs:
            self.config.drop_score = kwargs['drop_score']
        
        self._model = None
        self._available = self._initialize()
    
    @property
    def name(self) -> str:
        return "OnnxOCR (PP-OCRv5)"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _initialize(self) -> bool:
        """OnnxOCRモデルを初期化"""
        try:
            # onnxruntimeの確認
            import onnxruntime
            
            # OnnxOCRのインポート
            from onnxocr.onnx_paddleocr import ONNXPaddleOcr
            
            self._model = ONNXPaddleOcr(
                use_angle_cls=self.config.use_angle_cls,
                use_gpu=self.config.use_gpu,
                drop_score=self.config.drop_score
            )
            return True
            
        except ImportError as e:
            print(f"OnnxOCR初期化エラー: {e}")
            return False
        except Exception as e:
            print(f"OnnxOCRモデル初期化エラー: {e}")
            return False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        画像からテキストを認識
        
        Args:
            image: 入力画像（BGR）
            
        Returns:
            OCRResult: 認識結果
        """
        start_time = time.time()
        
        if not self._available or self._model is None:
            return OCRResult(
                full_text="",
                blocks=[],
                words=[],
                average_confidence=0.0,
                language="jpn+chi+eng",
                engine_name=self.name,
                processing_time=0.0,
                warnings=["OnnxOCRが利用できません。onnxruntimeをインストールしてください。"]
            )
        
        try:
            # BGR画像を確保（OnnxOCRはBGR入力）
            if len(image.shape) == 2:
                # グレースケールをBGRに変換
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # OCR実行
            result = self._model.ocr(image)
            
            # 結果をOCRResult形式に変換
            words = []
            full_text_lines = []
            confidences = []
            warnings = []
            
            if result and result[0]:
                for idx, line in enumerate(result[0]):
                    if len(line) >= 2:
                        # line[0]: bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        # line[1]: (text, confidence)
                        points = line[0]
                        text_data = line[1]
                        
                        text = text_data[0] if isinstance(text_data, (list, tuple)) else str(text_data)
                        confidence = float(text_data[1]) if isinstance(text_data, (list, tuple)) and len(text_data) > 1 else 0.0
                        
                        # 信頼度を0-100スケールに変換（OnnxOCRは0-1）
                        confidence_pct = confidence * 100
                        
                        bbox = BoundingBox.from_points(points)
                        
                        word = OCRWord(
                            text=text,
                            confidence=confidence_pct,
                            bbox=bbox,
                            block_num=0,
                            line_num=idx,
                            word_num=0
                        )
                        words.append(word)
                        full_text_lines.append(text)
                        
                        if confidence_pct > 0:
                            confidences.append(confidence_pct)
                        
                        # 低信頼度の警告
                        if 0 < confidence_pct < 60:
                            warnings.append(
                                f"低信頼度: '{text[:20]}...' (confidence: {confidence_pct:.1f}%)"
                            )
            
            # ブロック構造を構築
            blocks = self._build_blocks_from_words(words)
            
            # 平均信頼度
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                full_text="\n".join(full_text_lines),
                blocks=blocks,
                words=words,
                average_confidence=avg_confidence,
                language="jpn+chi+eng",
                engine_name=self.name,
                processing_time=processing_time,
                warnings=warnings[:20]
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return OCRResult(
                full_text="",
                blocks=[],
                words=[],
                average_confidence=0.0,
                language="jpn+chi+eng",
                engine_name=self.name,
                processing_time=processing_time,
                warnings=[f"OnnxOCR認識エラー: {str(e)}"]
            )


# 公開API
__all__ = [
    'OnnxOCREngine',
    'OnnxOCRConfig',
]
