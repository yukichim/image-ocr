"""
パイプライン統合テスト
"""

import pytest
import sys
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import OCRPipeline, PipelineConfig, PipelineResult
from src.classifier import DocumentType


class TestPipeline:
    """パイプラインのテスト"""
    
    def test_pipeline_initialization(self):
        """パイプラインの初期化テスト"""
        pipeline = OCRPipeline()
        assert pipeline is not None
        assert pipeline.preprocessor is not None
        assert pipeline.ocr_engine is not None
        assert pipeline.classifier is not None
    
    def test_pipeline_with_custom_config(self):
        """カスタム設定でのパイプライン初期化"""
        config = PipelineConfig()
        config.confidence_threshold = 70.0
        
        pipeline = OCRPipeline(config)
        assert pipeline.config.confidence_threshold == 70.0
    
    def test_process_nonexistent_file(self):
        """存在しないファイルの処理"""
        pipeline = OCRPipeline()
        result = pipeline.process("/nonexistent/path/image.jpg")
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "ファイルが見つかりません" in result.errors[0]
    
    def test_result_to_dict(self):
        """結果の辞書変換テスト"""
        result = PipelineResult(
            success=True,
            document_type=DocumentType.RECEIPT,
            confidence=0.95,
            data=None,
            ocr_text="テストテキスト",
            ocr_confidence=85.0,
            preprocessing_info={},
            classification_info={},
            warnings=["警告1"],
            errors=[]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["document_type"] == "receipt"
        assert result_dict["confidence"] == 0.95
        assert "警告1" in result_dict["warnings"]
    
    def test_result_to_json(self):
        """結果のJSON変換テスト"""
        result = PipelineResult(
            success=True,
            document_type=DocumentType.INVOICE,
            confidence=0.88,
            data=None,
            ocr_text="請求書テスト",
            ocr_confidence=90.0,
            preprocessing_info={},
            classification_info={},
            warnings=[],
            errors=[]
        )
        
        json_str = result.to_json()
        
        assert "invoice" in json_str
        assert "請求書テスト" not in json_str  # ocr_textはto_dictに含まれない


class TestPipelineWithSample:
    """サンプル画像を使用したパイプラインテスト"""
    
    @pytest.fixture
    def sample_image_path(self):
        """サンプル画像パス"""
        return Path(__file__).parent / "samples" / "index-mg-20-2.webp"
    
    def test_process_sample_image(self, sample_image_path):
        """サンプル画像の処理テスト"""
        if not sample_image_path.exists():
            pytest.skip(f"サンプル画像が見つかりません: {sample_image_path}")
        
        pipeline = OCRPipeline()
        result = pipeline.process(str(sample_image_path))
        
        # 処理が成功すること
        assert result.success is True
        
        # 何らかの文書タイプが判定されること
        assert result.document_type in [
            DocumentType.RECEIPT, 
            DocumentType.INVOICE, 
            DocumentType.UNKNOWN
        ]
        
        # OCRテキストが取得されること
        assert len(result.ocr_text) > 0
        
        print(f"\n処理結果:")
        print(f"  文書タイプ: {result.document_type.value}")
        print(f"  信頼度: {result.confidence:.2%}")
        print(f"  OCR信頼度: {result.ocr_confidence:.1f}%")
        print(f"  OCRテキスト（先頭200文字）:\n{result.ocr_text[:200]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
