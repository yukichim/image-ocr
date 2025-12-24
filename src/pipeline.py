"""
OCRパイプライン統合モジュール

全ての処理モジュールを統合:
1. 画像前処理
2. OCR認識（複数エンジン対応）
3. 文書分類
4. 情報抽出
5. データ正規化
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from enum import Enum

from .preprocessor import ImagePreprocessor, PreprocessConfig, PreprocessResult
from .ocr_engine import OCREngine, OCRConfig, OCRResult
from .engines import EngineType, get_available_engines, create_engine
from .engines.base import BaseOCREngine
from .classifier import DocumentClassifier, DocumentType, ClassificationResult
from .extractors.receipt import ReceiptExtractor, ReceiptData
from .extractors.invoice import InvoiceExtractor, InvoiceData
from .normalizer import DataNormalizer


class OutputFormat(Enum):
    """出力形式"""
    JSON = "json"
    DICT = "dict"


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    confidence_threshold: float = 60.0  # 警告を出す信頼度閾値
    auto_classify: bool = True  # 自動分類を行うか
    engine_type: EngineType = EngineType.TESSERACT  # 使用するOCRエンジン


@dataclass
class PipelineResult:
    """パイプライン処理結果"""
    success: bool
    document_type: DocumentType
    confidence: float
    data: Union[ReceiptData, InvoiceData, None]
    ocr_text: str
    ocr_confidence: float
    preprocessing_info: Dict[str, Any]
    classification_info: Dict[str, Any]
    engine_name: str = ""  # 使用したOCRエンジン名
    processing_time: float = 0.0  # OCR処理時間
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "success": self.success,
            "document_type": self.document_type.value,
            "confidence": round(self.confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 2),
            "engine_name": self.engine_name,
            "processing_time": round(self.processing_time, 3),
        }
        
        if self.data:
            result["data"] = self.data.to_dict()
        
        if self.warnings:
            result["warnings"] = self.warnings
        
        if self.errors:
            result["errors"] = self.errors
        
        # デバッグ情報（オプション）
        result["_debug"] = {
            "preprocessing": self.preprocessing_info,
            "classification": self.classification_info,
        }
        
        return result
    
    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """JSON文字列に変換"""
        return json.dumps(
            self.to_dict(), 
            indent=indent, 
            ensure_ascii=ensure_ascii
        )


class OCRPipeline:
    """OCRパイプラインクラス"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # 各モジュールの初期化
        self.preprocessor = ImagePreprocessor(self.config.preprocess)
        
        # OCRエンジンの遅延初期化（実際に使用するまで初期化しない）
        self._current_engine_type = self.config.engine_type
        self._ocr_engine = None  # 遅延初期化
        self._engine_initialized = False
        
        # 後方互換性のため
        self.ocr_engine = None
        
        self.classifier = DocumentClassifier()
        self.receipt_extractor = ReceiptExtractor()
        self.invoice_extractor = InvoiceExtractor()
        self.normalizer = DataNormalizer()
    
    def _create_ocr_engine(self, engine_type: EngineType) -> BaseOCREngine:
        """OCRエンジンを作成"""
        try:
            return create_engine(engine_type)
        except Exception as e:
            print(f"Warning: {engine_type.value}エンジンの初期化に失敗: {e}")
            # フォールバックとしてTesseractを使用
            if engine_type != EngineType.TESSERACT:
                return create_engine(EngineType.TESSERACT)
            raise
    
    def _ensure_engine_initialized(self) -> None:
        """エンジンが初期化されていることを確認（遅延初期化）"""
        if not self._engine_initialized:
            print(f"初回使用: {self._current_engine_type.display_name}エンジンを初期化中...")
            self._ocr_engine = self._create_ocr_engine(self._current_engine_type)
            self.ocr_engine = self._ocr_engine
            self._engine_initialized = True
    
    def set_engine(self, engine_type: EngineType) -> None:
        """OCRエンジンを切り替え"""
        if engine_type != self._current_engine_type:
            self._current_engine_type = engine_type
            self._engine_initialized = False  # 再初期化が必要
            self._ocr_engine = None
            self.ocr_engine = None
        # 実際の初期化は使用時に行う
    
    def get_current_engine(self) -> EngineType:
        """現在のエンジンタイプを取得"""
        return self._current_engine_type
    
    def get_available_engines(self) -> List[EngineType]:
        """利用可能なエンジンのリストを取得"""
        return get_available_engines()
    
    def process(self, image_path: str, 
                document_type: Optional[DocumentType] = None) -> PipelineResult:
        """
        画像を処理してデータを抽出
        
        Args:
            image_path: 画像ファイルパス
            document_type: 文書タイプ（指定しない場合は自動判定）
            
        Returns:
            PipelineResult: 処理結果
        """
        warnings = []
        errors = []
        preprocessing_info = {}
        classification_info = {}
        
        try:
            # 1. 画像前処理
            preprocess_result = self._preprocess(image_path)
            preprocessing_info = {
                "original_size": preprocess_result.original_size,
                "processed_size": preprocess_result.processed_size,
                "deskew_angle": preprocess_result.deskew_angle,
                "perspective_corrected": preprocess_result.perspective_corrected,
                "roi_detected": preprocess_result.roi_detected,
                "steps": preprocess_result.preprocessing_applied,
            }
            
            # 2. OCR認識（マルチスケール対応）
            ocr_result = self._recognize(
                preprocess_result.image, 
                preprocess_result.multi_scale_images
            )
            
            engine_name = getattr(ocr_result, 'engine_name', self._current_engine_type.display_name)
            processing_time = getattr(ocr_result, 'processing_time', 0.0)
            
            if ocr_result.average_confidence < self.config.confidence_threshold:
                warnings.append(
                    f"OCR信頼度が低いです: {ocr_result.average_confidence:.1f}%"
                )
            
            # OCR警告を追加
            warnings.extend(ocr_result.warnings[:5])  # 最大5件
            
            # 3. 文書分類
            if document_type is None and self.config.auto_classify:
                # アスペクト比を計算
                h, w = preprocess_result.image.shape[:2]
                aspect_ratio = h / w if w > 0 else 1.0
                
                classification = self.classifier.classify(
                    ocr_result.full_text,
                    aspect_ratio=aspect_ratio
                )
                document_type = classification.document_type
                
                classification_info = {
                    "detected_type": classification.document_type.value,
                    "confidence": classification.confidence,
                    "scores": classification.scores,
                    "reasoning": classification.reasoning,
                }
            else:
                document_type = document_type or DocumentType.UNKNOWN
                classification_info = {
                    "detected_type": document_type.value,
                    "confidence": 1.0,
                    "scores": {},
                    "reasoning": "ユーザー指定",
                }
            
            # 4. 情報抽出
            data = self._extract(ocr_result, document_type)
            
            if data:
                warnings.extend(data.warnings)
            
            # 5. 結果を返す
            return PipelineResult(
                success=True,
                document_type=document_type,
                confidence=classification_info.get("confidence", 0.0),
                data=data,
                ocr_text=ocr_result.full_text,
                ocr_confidence=ocr_result.average_confidence,
                preprocessing_info=preprocessing_info,
                classification_info=classification_info,
                engine_name=engine_name,
                processing_time=processing_time,
                warnings=warnings,
                errors=errors,
            )
            
        except FileNotFoundError as e:
            errors.append(f"ファイルが見つかりません: {e}")
        except ValueError as e:
            errors.append(f"画像の読み込みエラー: {e}")
        except RuntimeError as e:
            errors.append(f"OCRエンジンエラー: {e}")
        except Exception as e:
            errors.append(f"予期しないエラー: {type(e).__name__}: {e}")
        
        return PipelineResult(
            success=False,
            document_type=DocumentType.UNKNOWN,
            confidence=0.0,
            data=None,
            ocr_text="",
            ocr_confidence=0.0,
            preprocessing_info=preprocessing_info,
            classification_info=classification_info,
            warnings=warnings,
            errors=errors,
        )
    
    def _preprocess(self, image_path: str) -> PreprocessResult:
        """画像前処理を実行"""
        return self.preprocessor.process_file(image_path)
    
    def _recognize(self, image, multi_scale_images: list = None) -> OCRResult:
        """
        OCR認識を実行
        
        Phase 2: マルチスケール推論対応
        複数スケールの画像を使用して小文字検出を改善
        
        Args:
            image: 前処理済み画像
            multi_scale_images: マルチスケール画像リスト（オプション）
            
        Returns:
            OCRResult: 認識結果
        """
        # エンジンが初期化されていない場合は初期化
        self._ensure_engine_initialized()
        
        # 新しいエンジンシステムを使用
        return self._ocr_engine.recognize(image)
    
    def _extract(self, ocr_result: OCRResult, 
                 document_type: DocumentType) -> Optional[Union[ReceiptData, InvoiceData]]:
        """文書タイプに応じて情報抽出"""
        if document_type == DocumentType.RECEIPT:
            return self.receipt_extractor.extract(ocr_result)
        elif document_type == DocumentType.INVOICE:
            return self.invoice_extractor.extract(ocr_result)
        else:
            return None
    
    def process_batch(self, image_paths: list, 
                      document_type: Optional[DocumentType] = None) -> list:
        """
        複数画像を一括処理
        
        Args:
            image_paths: 画像ファイルパスのリスト
            document_type: 文書タイプ（全ファイルに適用）
            
        Returns:
            PipelineResultのリスト
        """
        results = []
        for path in image_paths:
            result = self.process(path, document_type)
            results.append(result)
        return results


def process_image(image_path: str, 
                  document_type: Optional[str] = None) -> PipelineResult:
    """
    画像処理のショートカット関数
    
    Args:
        image_path: 画像ファイルパス
        document_type: 文書タイプ ("receipt" or "invoice")
        
    Returns:
        PipelineResult: 処理結果
    """
    pipeline = OCRPipeline()
    
    doc_type = None
    if document_type:
        if document_type.lower() == "receipt":
            doc_type = DocumentType.RECEIPT
        elif document_type.lower() == "invoice":
            doc_type = DocumentType.INVOICE
    
    return pipeline.process(image_path, doc_type)
