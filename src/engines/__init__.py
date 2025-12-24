"""
OCRエンジンモジュール

複数のOCRエンジン（Tesseract, OnnxOCR, PaddleOCR）を統一インターフェースで提供
"""

from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseOCREngine


class EngineType(Enum):
    """OCRエンジンタイプ"""
    TESSERACT = "tesseract"      # Tesseract OCR（デフォルト）
    ONNXOCR = "onnxocr"          # OnnxOCR（PP-OCRv5 ONNX、日本語対応）
    PADDLEOCR = "paddleocr"      # PaddleOCR（方向検出あり）
    PADDLE_VL = "paddleocr_vl"   # PaddleOCR-VL (PP-Structure, レイアウト解析)
    
    @classmethod
    def from_string(cls, value: str) -> 'EngineType':
        """文字列からEngineTypeを取得"""
        value_lower = value.lower()
        for engine_type in cls:
            if engine_type.value == value_lower:
                return engine_type
        raise ValueError(f"Unknown engine type: {value}")
    
    @property
    def display_name(self) -> str:
        """表示名を取得"""
        names = {
            EngineType.TESSERACT: "Tesseract",
            EngineType.ONNXOCR: "OnnxOCR (PP-OCRv5)",
            EngineType.PADDLEOCR: "PaddleOCR",
            EngineType.PADDLE_VL: "PaddleOCR-VL (Layout)",
        }
        return names.get(self, self.value)
    
    @property
    def description(self) -> str:
        """説明を取得"""
        descriptions = {
            EngineType.TESSERACT: "Tesseract LSTM OCR - 軽量・高速",
            EngineType.ONNXOCR: "PP-OCRv5 ONNX版 - 日本語対応・方向分類あり",
            EngineType.PADDLEOCR: "PaddleOCR - 高精度・方向検出あり",
            EngineType.PADDLE_VL: "PaddleOCR-VL - レイアウト解析・表認識対応",
        }
        return descriptions.get(self, "")


def get_available_engines() -> list:
    """利用可能なエンジンのリストを取得"""
    available = []
    
    # Tesseractは常に利用可能（インストール済み前提）
    available.append(EngineType.TESSERACT)
    
    # OnnxOCRの確認
    try:
        import onnxruntime
        available.append(EngineType.ONNXOCR)
    except ImportError:
        pass
    
    # PaddleOCRの確認
    try:
        from paddleocr import PaddleOCR
        available.append(EngineType.PADDLEOCR)
        # PPStructureも同じパッケージにあるため利用可能とみなす
        # ただし、layoutparserなどの依存関係がある場合があるため、
        # 厳密にはPPStructureのインポートも試すべきだが、ここでは簡易的に追加
        available.append(EngineType.PADDLE_VL)
    except ImportError:
        pass
    
    return available


def create_engine(engine_type: EngineType, **kwargs) -> 'BaseOCREngine':
    """
    エンジンタイプに応じたOCRエンジンを作成
    
    Args:
        engine_type: エンジンタイプ
        **kwargs: エンジン固有のオプション
        
    Returns:
        BaseOCREngine: OCRエンジンインスタンス
    """
    if engine_type == EngineType.TESSERACT:
        from .tesseract_engine import TesseractEngine
        return TesseractEngine(**kwargs)
    
    elif engine_type == EngineType.ONNXOCR:
        from .onnxocr_engine import OnnxOCREngine
        return OnnxOCREngine(**kwargs)
    
    elif engine_type == EngineType.PADDLEOCR:
        from .paddle_engine import PaddleOCREngine
        return PaddleOCREngine(**kwargs)
        
    elif engine_type == EngineType.PADDLE_VL:
        from .paddle_structure_engine import PaddleStructureEngine
        return PaddleStructureEngine(**kwargs)
    
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


# 公開API
__all__ = [
    'EngineType',
    'get_available_engines',
    'create_engine',
]
