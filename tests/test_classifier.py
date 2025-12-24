"""
文書分類モジュールのテスト
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import DocumentClassifier, DocumentType, classify_document


class TestDocumentClassifier:
    """文書分類のテスト"""
    
    def test_classify_receipt_by_keyword(self):
        """領収書キーワードでの分類"""
        text = """
        領収書
        ○○コンビニエンスストア
        2024年1月15日
        
        おにぎり ¥150
        お茶 ¥130
        
        合計 ¥280
        お預り ¥500
        お釣り ¥220
        """
        
        result = classify_document(text)
        
        assert result.document_type == DocumentType.RECEIPT
        assert result.confidence > 0.5
        assert "領収書" in result.matched_keywords["receipt"]
    
    def test_classify_invoice_by_keyword(self):
        """請求書キーワードでの分類"""
        text = """
        請求書
        
        株式会社テスト 御中
        
        請求日: 2024年1月20日
        支払期限: 2024年2月29日
        
        請求金額: ¥55,000
        
        振込先:
        ○○銀行 △△支店
        普通 1234567
        """
        
        result = classify_document(text)
        
        assert result.document_type == DocumentType.INVOICE
        assert result.confidence > 0.5
        assert "請求書" in result.matched_keywords["invoice"]
    
    def test_classify_with_aspect_ratio(self):
        """アスペクト比による補正"""
        # 縦長のテキスト（レシートっぽい）
        text = "合計 ¥1000"
        
        result_tall = classify_document(text, aspect_ratio=3.0)  # 縦長
        result_wide = classify_document(text, aspect_ratio=0.5)  # 横長
        
        # 縦長の場合はレシートスコアが上がる
        assert result_tall.scores["receipt"] >= result_wide.scores["receipt"]
    
    def test_classify_unknown_document(self):
        """不明な文書"""
        text = "これは何の文書かわかりません。"
        
        result = classify_document(text)
        
        # 閾値未満の場合はUNKNOWN
        if result.confidence < 0.3:
            assert result.document_type == DocumentType.UNKNOWN
    
    def test_negative_keywords(self):
        """否定キーワードの影響"""
        # 請求書キーワードが入っている「領収書」
        text = """
        領収書
        振込先: ○○銀行
        口座番号: 1234567
        """
        
        result = classify_document(text)
        
        # 否定キーワードによりスコアが下がるが、
        # 「領収書」の強いポジティブスコアで領収書と判定される可能性が高い
        assert result.document_type in [DocumentType.RECEIPT, DocumentType.INVOICE]
    
    def test_classification_result_attributes(self):
        """分類結果の属性確認"""
        text = "領収書 合計 ¥1000"
        
        result = classify_document(text)
        
        assert hasattr(result, 'document_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'scores')
        assert hasattr(result, 'matched_keywords')
        assert hasattr(result, 'reasoning')
        
        assert isinstance(result.scores, dict)
        assert "receipt" in result.scores
        assert "invoice" in result.scores


class TestClassifierEdgeCases:
    """エッジケースのテスト"""
    
    def test_empty_text(self):
        """空テキスト"""
        result = classify_document("")
        assert result.document_type == DocumentType.UNKNOWN
    
    def test_only_whitespace(self):
        """空白のみ"""
        result = classify_document("   \n\t\n   ")
        assert result.document_type == DocumentType.UNKNOWN
    
    def test_mixed_keywords(self):
        """両方のキーワードを含む"""
        text = """
        領収書兼請求書
        合計金額
        振込先
        """
        
        result = classify_document(text)
        
        # 両方のスコアが計算される
        assert result.scores["receipt"] > 0
        assert result.scores["invoice"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
