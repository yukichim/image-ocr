"""
PaddleOCRエンジン

PaddleOCRを使用した高精度OCR:
- 日本語対応（PP-OCRv5）
- 方向検出・自動補正
- 高精度な文字認識
"""

import time
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from .base import (
    BaseOCREngine, BoundingBox, OCRWord, OCRResult
)


@dataclass
class PaddleOCRConfig:
    """PaddleOCR設定"""
    lang: str = "japan"          # 言語（japan, ch, en, etc.）
    use_angle_cls: bool = False  # 方向分類を使用（Falseで追加モデルダウンロード回避）
    use_gpu: bool = False        # GPU使用
    show_log: bool = False       # ログ表示
    drop_score: float = 0.5      # 信頼度閾値
    # 注意: use_angle_cls, use_doc_orientation_classify, use_doc_unwarping は
    # 追加のモデルダウンロードが必要なため、デフォルトでは無効化


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCRエンジン"""
    
    def __init__(self, config: Optional[PaddleOCRConfig] = None, **kwargs):
        self.config = config or PaddleOCRConfig()
        
        # kwargsからの設定上書き
        if 'lang' in kwargs:
            self.config.lang = kwargs['lang']
        if 'use_angle_cls' in kwargs:
            self.config.use_angle_cls = kwargs['use_angle_cls']
        if 'use_gpu' in kwargs:
            self.config.use_gpu = kwargs['use_gpu']
        
        self._model = None
        self._available = self._initialize()
    
    @property
    def name(self) -> str:
        return "PaddleOCR"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _get_local_model_dir(self, model_type: str, lang: str) -> Optional[str]:
        """ローカルのモデルディレクトリを検索"""
        from pathlib import Path
        
        base_dir = Path.home() / ".paddleocr" / "whl" / model_type
        if not base_dir.exists():
            return None
            
        # 検索候補の言語リスト
        langs_to_check = [lang]
        if model_type in ['det', 'cls']:
            # det/clsは ch (中国語) モデルがデフォルトで使われることが多い
            if 'ch' not in langs_to_check: langs_to_check.append('ch')
            if 'en' not in langs_to_check: langs_to_check.append('en')
            if 'ml' not in langs_to_check: langs_to_check.append('ml')
            
        for l in langs_to_check:
            lang_dir = base_dir / l
            if lang_dir.exists():
                subdirs = [d for d in lang_dir.iterdir() if d.is_dir()]
                if subdirs:
                    # 名前でソートして最新のものを使用
                    subdirs.sort(key=lambda x: x.name, reverse=True)
                    return str(subdirs[0])
        return None

    def _initialize(self) -> bool:
        """PaddleOCRモデルを初期化"""
        try:
            import logging
            
            # PaddleOCRのログレベルを警告以上に設定（ダウンロード進捗は表示）
            logging.getLogger('ppocr').setLevel(logging.WARNING)
            logging.getLogger('paddleocr').setLevel(logging.WARNING)
            
            from paddleocr import PaddleOCR
            
            print("PaddleOCR初期化中... (初回はモデルダウンロードに時間がかかります)")
            
            # ローカルモデルの検索と設定
            # これにより、既にモデルがある場合の不要なダウンロードチェック/ダウンロードを防ぐ
            kwargs = {
                'lang': self.config.lang,
                'use_angle_cls': False,  # 方向検出を無効化
                'use_doc_orientation_classify': False,  # ドキュメント方向分類を無効化
                'use_doc_unwarping': False,  # ドキュメント補正を無効化
                'show_log': False,
            }
            
            # 検出モデル (det)
            det_dir = self._get_local_model_dir('det', self.config.lang)
            if det_dir:
                kwargs['det_model_dir'] = det_dir
                
            # 認識モデル (rec)
            rec_dir = self._get_local_model_dir('rec', self.config.lang)
            if rec_dir:
                kwargs['rec_model_dir'] = rec_dir

            self._model = PaddleOCR(**kwargs)
            
            print("PaddleOCR初期化完了")
            return True
            
        except ImportError as e:
            print(f"PaddleOCR初期化エラー: {e}")
            print("インストール: pip install paddleocr paddlepaddle")
            return False
        except Exception as e:
            print(f"PaddleOCRモデル初期化エラー: {e}")
            # 最もシンプルな設定でリトライ
            try:
                from paddleocr import PaddleOCR
                self._model = PaddleOCR(lang=self.config.lang, use_angle_cls=False)
                return True
            except Exception as e2:
                print(f"PaddleOCR初期化失敗: {e2}")
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
                language=self.config.lang,
                engine_name=self.name,
                processing_time=0.0,
                warnings=["PaddleOCRが利用できません。pip install paddleocr paddlepaddle"]
            )
        
        try:
            # BGR画像を確保
            if len(image.shape) == 2:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # OCR実行（PaddleOCR 3.x API）
            # predict() または ocr() を使用
            try:
                # PaddleOCR 3.x
                result = self._model.predict(input=image)
            except AttributeError:
                # PaddleOCR 2.x
                result = self._model.ocr(image, cls=self.config.use_angle_cls)
            
            # 結果をOCRResult形式に変換
            words = []
            full_text_lines = []
            confidences = []
            warnings = []
            
            # PaddleOCR 3.x の結果形式を処理
            ocr_data = None
            if hasattr(result, '__iter__'):
                # 結果オブジェクトから抽出
                for res in result:
                    if hasattr(res, 'rec_texts') and hasattr(res, 'rec_scores'):
                        # PaddleOCR 3.x 形式
                        ocr_data = self._parse_paddleocr3_result(res)
                        break
                    elif hasattr(res, 'print'):
                        # 別の形式
                        pass
                
                # リスト形式の場合（2.x互換）
                if ocr_data is None and isinstance(result, list) and result:
                    if isinstance(result[0], list):
                        ocr_data = result[0]
            
            if ocr_data:
                for idx, line in enumerate(ocr_data):
                    if isinstance(line, dict):
                        # PaddleOCR 3.x dict形式
                        text = line.get('text', '')
                        confidence = line.get('score', 0.0)
                        points = line.get('points', [[0,0], [0,0], [0,0], [0,0]])
                    elif len(line) >= 2:
                        # 2.x形式: [points, (text, score)]
                        points = line[0]
                        text_data = line[1]
                        text = text_data[0] if isinstance(text_data, (list, tuple)) else str(text_data)
                        confidence = float(text_data[1]) if isinstance(text_data, (list, tuple)) and len(text_data) > 1 else 0.0
                    else:
                        continue
                    
                    # 信頼度を0-100スケールに変換
                    confidence_pct = confidence * 100 if confidence <= 1.0 else confidence
                    
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
                language=self.config.lang,
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
                language=self.config.lang,
                engine_name=self.name,
                processing_time=processing_time,
                warnings=[f"PaddleOCR認識エラー: {str(e)}"]
            )
    
    def _parse_paddleocr3_result(self, res) -> List[dict]:
        """PaddleOCR 3.x の結果オブジェクトをパース"""
        parsed = []
        
        try:
            # 辞書形式の場合（PaddleOCR v2.7+ / PaddleX）
            if isinstance(res, dict):
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                # dt_polys (検出ポリゴン) または rec_boxes (認識ボックス) を使用
                boxes = res.get('dt_polys', [])
                if not boxes:
                    boxes = res.get('rec_boxes', [])
            else:
                # オブジェクト形式の場合
                texts = getattr(res, 'rec_texts', [])
                scores = getattr(res, 'rec_scores', [])
                boxes = getattr(res, 'dt_polys', [])
                if not boxes:
                    boxes = getattr(res, 'rec_boxes', [])
            
            for i, (text, score) in enumerate(zip(texts, scores)):
                box = boxes[i] if i < len(boxes) else [[0,0], [0,0], [0,0], [0,0]]
                parsed.append({
                    'text': text,
                    'score': score,
                    'points': box
                })
        except Exception as e:
            print(f"PaddleOCR結果パースエラー: {e}")
            pass
        
        return parsed


# 公開API
__all__ = [
    'PaddleOCREngine',
    'PaddleOCRConfig',
]
