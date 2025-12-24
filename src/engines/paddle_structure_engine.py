"""
PaddleOCR-VL (PP-Structure) エンジン

PaddleOCRのPP-Structureを使用したレイアウト解析・表認識対応エンジン:
- レイアウト解析（ヘッダー、フッター、図、表の分離）
- 表構造認識
- 日本語対応
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

from .base import (
    BaseOCREngine, BoundingBox, OCRWord, OCRResult, OCRBlock
)


@dataclass
class PaddleStructureConfig:
    """PaddleOCR-VL設定"""
    lang: str = "japan"          # 言語
    use_gpu: bool = False        # GPU使用
    show_log: bool = False       # ログ表示
    layout: bool = True          # レイアウト解析を使用
    table: bool = False          # 表認識を使用 (メモリ消費が激しいためデフォルトFalse)
    ocr: bool = True             # OCRを使用
    use_angle_cls: bool = False  # 方向分類（追加モデルダウンロード回避のためデフォルトFalse）


class PaddleStructureEngine(BaseOCREngine):
    """PaddleOCR-VL (PP-Structure) エンジン"""
    
    def __init__(self, config: Optional[PaddleStructureConfig] = None, **kwargs):
        self.config = config or PaddleStructureConfig()
        
        # kwargsからの設定上書き
        if 'lang' in kwargs:
            self.config.lang = kwargs['lang']
        if 'use_gpu' in kwargs:
            self.config.use_gpu = kwargs['use_gpu']
        
        self._model = None
        self._available = self._initialize()
    
    @property
    def name(self) -> str:
        return "PaddleOCR-VL"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _initialize(self) -> bool:
        """PaddleOCR-VLモデルを初期化"""
        try:
            import logging
            
            # ログレベル設定
            logging.getLogger('ppocr').setLevel(logging.WARNING)
            logging.getLogger('paddleocr').setLevel(logging.WARNING)
            
            # デフォルトでPPStructureV3を使用（軽量で安定しているため）
            # PaddleOCRVLはモデルが大きくOOMのリスクがあるため、明示的に要求されない限りPPStructureV3を優先
            try:
                from paddleocr import PPStructureV3
                model_class = PPStructureV3
                model_name = "PPStructureV3"
            except ImportError:
                try:
                    from paddleocr import PaddleOCRVL
                    model_class = PaddleOCRVL
                    model_name = "PaddleOCRVL"
                except ImportError:
                    print("PaddleOCRの構造解析モデルが見つかりません")
                    return False
            
            print(f"{model_name} 初期化中... (初回はモデルダウンロードに時間がかかります)")
            
            # モデル初期化
            # ダウンロード回避のため、不要な機能は無効化
            if model_name == "PaddleOCRVL":
                # PaddleOCRVLはlang引数を持たない場合がある（多言語モデル使用）
                self._model = model_class(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_layout_detection=self.config.layout,
                )
            else:
                # PPStructureV3などはlang引数を持つ
                # use_layout_detection引数がない場合があるため、kwargsで渡すか省略
                # また、数式認識(formula)や印鑑認識(seal)などの追加モデルダウンロードを防ぐために無効化
                
                if self.config.table:
                    print("警告: 表認識(table=True)が有効です。メモリ消費量が増加し、初期化に時間がかかります。")
                
                self._model = model_class(
                    lang=self.config.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_table_recognition=self.config.table,
                    use_formula_recognition=False, # 数式認識を無効化
                    use_seal_recognition=False,    # 印鑑認識を無効化
                )
            
            print(f"{model_name} 初期化完了")
            return True
            
        except ImportError as e:
            print(f"PaddleOCR-VL初期化エラー: {e}")
            print("インストール: pip install paddleocr paddlepaddle")
            return False
        except Exception as e:
            print(f"PaddleOCR-VLモデル初期化エラー: {e}")
            return False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        画像からテキストと構造を認識
        
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
                warnings=["PaddleOCR-VLが利用できません"]
            )
        
        try:
            # BGR画像を確保
            if len(image.shape) == 2:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # 推論実行
            try:
                if hasattr(self._model, 'predict'):
                    results = self._model.predict(image)
                else:
                    results = self._model(image)
            except Exception as e:
                # フォールバック
                print(f"推論メソッド呼び出しエラー: {e}, __call__を試行します")
                results = self._model(image)
            
            words = []
            blocks = []
            full_text_lines = []
            confidences = []
            warnings = []
            
            # 結果の解析 (V3対応)
            # V3 returns a list containing a LayoutParsingResultV2 object/dict
            is_v3_format = False
            parsing_results = []
            
            if isinstance(results, list) and len(results) > 0:
                first_res = results[0]
                # Check for V3 keys/attributes
                if isinstance(first_res, dict) and 'parsing_res_list' in first_res:
                    parsing_results = first_res['parsing_res_list']
                    is_v3_format = True
                elif hasattr(first_res, 'parsing_res_list'):
                    parsing_results = first_res.parsing_res_list
                    is_v3_format = True
                elif hasattr(first_res, 'keys') and 'parsing_res_list' in first_res.keys():
                     parsing_results = first_res['parsing_res_list']
                     is_v3_format = True

            if is_v3_format:
                # V3 Format Parsing
                for idx, region in enumerate(parsing_results):
                    # region can be dict or object
                    def get_attr(obj, key, default=None):
                        if isinstance(obj, dict):
                            return obj.get(key, default)
                        return getattr(obj, key, default)

                    label = get_attr(region, 'label', 'Unknown')
                    bbox = get_attr(region, 'bbox', [0, 0, 0, 0])
                    content = get_attr(region, 'content', '')
                    
                    # BBox conversion (bbox format: [x1, y1, x2, y2])
                    block_bbox = BoundingBox(
                        left=int(bbox[0]),
                        top=int(bbox[1]),
                        width=int(bbox[2] - bbox[0]),
                        height=int(bbox[3] - bbox[1])
                    )
                    
                    # Clean content
                    block_text = content.strip() if content else ""
                    
                    if block_text:
                        full_text_lines.append(block_text)
                        
                        # Create a block
                        # Note: V3 parsing results might not give per-word confidence easily
                        # We might need to look into overall_ocr_res for that, but mapping is hard.
                        # For now, use a default confidence or try to find it.
                        block = OCRBlock(
                            text=block_text,
                            bbox=block_bbox,
                            confidence=0.9, # Placeholder as V3 parsing result might not have it directly
                            block_type=label,
                            words=[] # Words are hard to map back without more complex logic
                        )
                        blocks.append(block)
                        
                        # Create a pseudo-word for the whole block so it shows up in visualization
                        word = OCRWord(
                            text=block_text,
                            confidence=90.0,
                            bbox=block_bbox,
                            block_num=idx,
                            line_num=0,
                            word_num=0
                        )
                        words.append(word)
                        confidences.append(90.0)

            else:
                # Old Format Parsing (List of dicts)
                for idx, region in enumerate(results):
                    region_type = region.get('type', 'Unknown')
                    region_bbox = region.get('bbox', [0, 0, 0, 0])
                    res = region.get('res', [])
                    
                    # 領域全体のBBox (bbox format: [x1, y1, x2, y2])
                    block_bbox = BoundingBox(
                        left=int(region_bbox[0]),
                        top=int(region_bbox[1]),
                        width=int(region_bbox[2] - region_bbox[0]),
                        height=int(region_bbox[3] - region_bbox[1])
                    )
                    
                    block_text = ""
                    block_words = []
                    
                    if region_type == 'Table':
                        if isinstance(res, dict) and 'html' in res:
                            block_text = "[Table]"
                        elif isinstance(res, list):
                            for line in res:
                                text = line.get('text', '')
                                score = line.get('confidence', 0.0)
                                if not text and 'transcription' in line:
                                    text = line['transcription']
                                
                                if text:
                                    block_text += text + "\n"
                                    confidences.append(score * 100)
                    
                    elif isinstance(res, list):
                        for line_idx, line in enumerate(res):
                            text = line.get('text', '')
                            score = line.get('confidence', 0.0)
                            text_region = line.get('text_region', [])
                            
                            if not text:
                                continue
                                
                            confidence_pct = score * 100
                            confidences.append(confidence_pct)
                            
                            if text_region:
                                xs = [p[0] for p in text_region]
                                ys = [p[1] for p in text_region]
                                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                                word_bbox = BoundingBox(
                                    left=x1,
                                    top=y1,
                                    width=x2 - x1,
                                    height=y2 - y1
                                )
                            else:
                                word_bbox = block_bbox
                            
                            word = OCRWord(
                                text=text,
                                confidence=confidence_pct,
                                bbox=word_bbox,
                                block_num=idx,
                                line_num=line_idx,
                                word_num=0
                            )
                            words.append(word)
                            block_words.append(word)
                            block_text += text + "\n"
                    
                    if block_text:
                        full_text_lines.append(block_text.strip())
                        block = OCRBlock(
                            text=block_text.strip(),
                            bbox=block_bbox,
                            confidence=sum(w.confidence for w in block_words)/len(block_words) if block_words else 0.0,
                            block_type=region_type,
                            words=block_words
                        )
                        blocks.append(block)
            
            # 平均信頼度
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                full_text="\n\n".join(full_text_lines),
                blocks=blocks,
                words=words,
                average_confidence=avg_confidence,
                language=self.config.lang,
                engine_name=self.name,
                processing_time=processing_time,
                warnings=warnings
            )
            
        except Exception as e:
            print(f"PaddleOCR-VL認識エラー: {e}")
            import traceback
            traceback.print_exc()
            return OCRResult(
                full_text="",
                blocks=[],
                words=[],
                average_confidence=0.0,
                language=self.config.lang,
                engine_name=self.name,
                processing_time=time.time() - start_time,
                warnings=[f"エラーが発生しました: {e}"]
            )
