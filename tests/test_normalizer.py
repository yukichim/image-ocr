"""
正規化モジュールのテスト
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.normalizer import (
    DateNormalizer, 
    AmountNormalizer, 
    OCRCorrector,
    TextCleaner,
    DataNormalizer,
    normalize_date,
    normalize_amount
)


class TestDateNormalizer:
    """日付正規化のテスト"""
    
    def test_western_date_slash(self):
        """西暦スラッシュ形式"""
        assert normalize_date("2024/01/15") == "2024-01-15"
    
    def test_western_date_hyphen(self):
        """西暦ハイフン形式"""
        assert normalize_date("2024-01-15") == "2024-01-15"
    
    def test_western_date_japanese(self):
        """西暦日本語形式"""
        assert normalize_date("2024年1月15日") == "2024-01-15"
    
    def test_reiwa_date(self):
        """令和日付"""
        assert normalize_date("令和6年1月15日") == "2024-01-15"
        assert normalize_date("R6.1.15") == "2024-01-15"
    
    def test_heisei_date(self):
        """平成日付"""
        assert normalize_date("平成31年4月30日") == "2019-04-30"
        assert normalize_date("H31.4.30") == "2019-04-30"
    
    def test_fullwidth_digits(self):
        """全角数字"""
        assert normalize_date("２０２４年１月１５日") == "2024-01-15"
    
    def test_invalid_date(self):
        """無効な日付"""
        assert normalize_date("invalid") is None
        assert normalize_date("") is None


class TestAmountNormalizer:
    """金額正規化のテスト"""
    
    def test_simple_number(self):
        """単純な数字"""
        assert normalize_amount("1000") == 1000
    
    def test_with_comma(self):
        """カンマ区切り"""
        assert normalize_amount("1,000") == 1000
        assert normalize_amount("1,234,567") == 1234567
    
    def test_with_yen_symbol(self):
        """円記号付き"""
        assert normalize_amount("¥1,000") == 1000
        assert normalize_amount("￥1,000") == 1000
    
    def test_with_yen_suffix(self):
        """円サフィックス"""
        assert normalize_amount("1,000円") == 1000
    
    def test_with_trailing_hyphen(self):
        """末尾ハイフン（レシートでよくある形式）"""
        assert normalize_amount("1,000-") == 1000
    
    def test_fullwidth_digits(self):
        """全角数字"""
        assert normalize_amount("１，０００") == 1000
    
    def test_invalid_amount(self):
        """無効な金額"""
        assert normalize_amount("") is None
        assert normalize_amount("abc") is None


class TestOCRCorrector:
    """OCR補正のテスト"""
    
    def test_company_name_correction(self):
        """会社名の補正"""
        corrector = OCRCorrector()
        result = corrector.correct_text("株式合社テスト")
        
        assert result.normalized == "株式会社テスト"
        assert len(result.corrections) > 0
    
    def test_amount_correction(self):
        """金額の補正"""
        corrector = OCRCorrector()
        
        # 英字Oを数字0に
        result = corrector.correct_amount("1O,OOO")
        assert result.normalized == "10,000"


class TestTextCleaner:
    """テキストクリーニングのテスト"""
    
    def test_normalize_whitespace(self):
        """空白の正規化"""
        cleaner = TextCleaner()
        
        result = cleaner.clean("テスト  テスト")
        assert "  " not in result
    
    def test_remove_excessive_newlines(self):
        """過剰な改行の除去"""
        cleaner = TextCleaner()
        
        result = cleaner.clean("行1\n\n\n\n行2")
        assert result.count("\n") <= 2
    
    def test_trim_lines(self):
        """行の前後空白を除去"""
        cleaner = TextCleaner()
        
        result = cleaner.clean("  テスト  \n  テスト2  ")
        lines = result.split("\n")
        assert lines[0] == "テスト"


class TestDataNormalizer:
    """統合正規化のテスト"""
    
    def test_normalize_date(self):
        """日付正規化"""
        normalizer = DataNormalizer()
        assert normalizer.normalize_date("R6.1.15") == "2024-01-15"
    
    def test_normalize_amount(self):
        """金額正規化"""
        normalizer = DataNormalizer()
        assert normalizer.normalize_amount("¥1,000") == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
