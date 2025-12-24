"""
Tesseract OCRエンジン

Tesseract OCRを使用した高精度日本語文字認識:
- LSTM モード（--oem 1）による高精度認識
- 文字単位の信頼度スコア取得
- 座標情報（bounding box）抽出
- 日本語＋英語の混在対応
"""

import time
import pytesseract
from pytesseract import Output
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

from .base import (
    BaseOCREngine, BoundingBox, OCRWord, OCRLine, OCRBlock, OCRResult
)


class OCRMode(Enum):
    """OCR動作モード"""
    LEGACY = 0          # レガシーエンジン
    LSTM = 1            # LSTMエンジン（推奨）
    LEGACY_LSTM = 2     # 両方を組み合わせ
    DEFAULT = 3         # デフォルト


class PageSegMode(Enum):
    """ページセグメンテーションモード"""
    OSD_ONLY = 0              # 方向とスクリプト検出のみ
    AUTO_OSD = 1              # 自動ページセグメンテーション + OSD
    AUTO_ONLY = 2             # 自動ページセグメンテーション（OSDなし）
    FULLY_AUTO = 3            # 完全自動（デフォルト）
    SINGLE_COLUMN = 4         # 単一カラム
    SINGLE_BLOCK_VERT = 5     # 単一ブロック（縦書き）
    SINGLE_BLOCK = 6          # 単一ブロック（推奨）
    SINGLE_LINE = 7           # 単一行
    SINGLE_WORD = 8           # 単一単語
    CIRCLE_WORD = 9           # 円形の単語
    SINGLE_CHAR = 10          # 単一文字
    SPARSE_TEXT = 11          # スパーステキスト
    SPARSE_TEXT_OSD = 12      # スパーステキスト + OSD
    RAW_LINE = 13             # 生の行


@dataclass
class TesseractConfig:
    """Tesseract設定"""
    language: str = "jpn+eng"  # 日本語＋英語
    oem: OCRMode = OCRMode.LSTM
    psm: PageSegMode = PageSegMode.FULLY_AUTO
    confidence_threshold: float = 60.0
    custom_config: str = ""


class TesseractEngine(BaseOCREngine):
    """Tesseract OCRエンジン"""
    
    def __init__(self, config: Optional[TesseractConfig] = None, **kwargs):
        self.config = config or TesseractConfig()
        
        # kwargsからの設定上書き
        if 'language' in kwargs:
            self.config.language = kwargs['language']
        
        self._tesseract_version = None
        self._available = self._verify_tesseract()
    
    @property
    def name(self) -> str:
        return "Tesseract"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _verify_tesseract(self) -> bool:
        """Tesseractが利用可能か確認"""
        try:
            version = pytesseract.get_tesseract_version()
            self._tesseract_version = str(version)
            return True
        except Exception:
            return False
    
    def _build_config(self) -> str:
        """Tesseract設定文字列を構築"""
        config_parts = [
            f"--oem {self.config.oem.value}",
            f"--psm {self.config.psm.value}",
        ]
        
        if self.config.custom_config:
            config_parts.append(self.config.custom_config)
        
        return " ".join(config_parts)
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        画像からテキストを認識
        
        Args:
            image: 前処理済み画像（グレースケールまたはBGR）
            
        Returns:
            OCRResult: 認識結果
        """
        start_time = time.time()
        
        if not self._available:
            return OCRResult(
                full_text="",
                blocks=[],
                words=[],
                average_confidence=0.0,
                language=self.config.language,
                engine_name=self.name,
                processing_time=0.0,
                warnings=["Tesseract OCRが利用できません"]
            )
        
        config = self._build_config()
        
        # 詳細データを取得（座標、信頼度含む）
        data = pytesseract.image_to_data(
            image,
            lang=self.config.language,
            config=config,
            output_type=Output.DICT
        )
        
        # 全文テキストを取得
        full_text = pytesseract.image_to_string(
            image,
            lang=self.config.language,
            config=config
        )
        
        # 結果をパース
        words = []
        warnings = []
        
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
            
            conf = float(data['conf'][i])
            
            word = OCRWord(
                text=text,
                confidence=conf,
                bbox=BoundingBox(
                    left=data['left'][i],
                    top=data['top'][i],
                    width=data['width'][i],
                    height=data['height'][i]
                ),
                block_num=data['block_num'][i],
                line_num=data['line_num'][i],
                word_num=data['word_num'][i]
            )
            words.append(word)
            
            # 低信頼度の警告
            if 0 < conf < self.config.confidence_threshold:
                warnings.append(
                    f"低信頼度: '{text}' (confidence: {conf:.1f}%)"
                )
        
        # ブロックと行を構築
        blocks = self._build_blocks_from_words(words)
        
        # 平均信頼度を計算
        valid_confidences = [w.confidence for w in words if w.confidence > 0]
        avg_confidence = (sum(valid_confidences) / len(valid_confidences) 
                         if valid_confidences else 0.0)
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            full_text=full_text.strip(),
            blocks=blocks,
            words=words,
            average_confidence=avg_confidence,
            language=self.config.language,
            engine_name=self.name,
            processing_time=processing_time,
            warnings=warnings[:20]  # 最大20件
        )


# 公開API
__all__ = [
    'TesseractEngine',
    'TesseractConfig',
    'OCRMode',
    'PageSegMode',
]
