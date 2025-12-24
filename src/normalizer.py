"""
データ正規化モジュール

OCR結果の正規化処理:
- 日付正規化（和暦→西暦、形式統一）
- 金額正規化（全角→半角、記号除去）
- OCR誤認識補正（辞書ベース）
- テキストクリーニング
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime


@dataclass
class NormalizationResult:
    """正規化結果"""
    original: str
    normalized: str
    corrections: List[str]  # 適用された補正のリスト


class DateNormalizer:
    """日付正規化クラス"""
    
    # 和暦の元号と開始年
    ERA_MAP = {
        '令和': 2018,
        'R': 2018,
        'Ｒ': 2018,
        '平成': 1988,
        'H': 1988,
        'Ｈ': 1988,
        '昭和': 1925,
        'S': 1925,
        'Ｓ': 1925,
        '大正': 1911,
        'T': 1911,
        'Ｔ': 1911,
        '明治': 1867,
        'M': 1867,
        'Ｍ': 1867,
    }
    
    # 日付パターン
    PATTERNS = [
        # 西暦（YYYY年MM月DD日）
        (r'(\d{4})\s*[年/\-.\s]\s*(\d{1,2})\s*[月/\-.\s]\s*(\d{1,2})\s*日?', 'western'),
        # 西暦（YYYY/MM/DD, YYYY-MM-DD）
        (r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', 'western'),
        # 西暦コンパクト（YYYYMMDD）
        (r'^(\d{4})(\d{2})(\d{2})$', 'western'),
        # 和暦（令和X年X月X日）
        (r'(令和|平成|昭和|大正|明治|R|H|S|T|M|Ｒ|Ｈ|Ｓ|Ｔ|Ｍ)\s*[元\d]{1,2}\s*[年.]\s*(\d{1,2})\s*[月.]\s*(\d{1,2})\s*日?', 'japanese'),
    ]
    
    def normalize(self, date_str: str) -> Optional[str]:
        """
        日付文字列をISO形式（YYYY-MM-DD）に正規化
        
        Args:
            date_str: 日付文字列
            
        Returns:
            ISO形式の日付文字列、またはNone
        """
        if not date_str:
            return None
        
        # 全角数字を半角に変換
        date_str = self._normalize_digits(date_str)
        
        for pattern, pattern_type in self.PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if pattern_type == 'western':
                        year = int(match.group(1))
                        month = int(match.group(2))
                        day = int(match.group(3))
                    else:  # japanese
                        era = match.group(1)
                        era_year_str = re.search(r'[元\d]+', date_str[match.start():]).group()
                        era_year = 1 if era_year_str == '元' else int(era_year_str)
                        base_year = self.ERA_MAP.get(era, 2018)
                        year = base_year + era_year
                        month = int(match.group(2))
                        day = int(match.group(3))
                    
                    # 妥当性チェック
                    if self._is_valid_date(year, month, day):
                        return f"{year:04d}-{month:02d}-{day:02d}"
                except (ValueError, AttributeError):
                    continue
        
        return None
    
    def _normalize_digits(self, text: str) -> str:
        """全角数字を半角に変換"""
        return text.translate(str.maketrans(
            '０１２３４５６７８９',
            '0123456789'
        ))
    
    def _is_valid_date(self, year: int, month: int, day: int) -> bool:
        """日付の妥当性をチェック"""
        try:
            datetime(year, month, day)
            return 1900 <= year <= 2100
        except ValueError:
            return False


class AmountNormalizer:
    """金額正規化クラス"""
    
    # 通貨記号
    CURRENCY_SYMBOLS = ['¥', '￥', '\\', '円', '$', '€']
    
    # 漢数字マッピング
    KANJI_NUMBERS = {
        '〇': '0', '零': '0',
        '一': '1', '壱': '1',
        '二': '2', '弐': '2',
        '三': '3', '参': '3',
        '四': '4',
        '五': '5', '伍': '5',
        '六': '6',
        '七': '7',
        '八': '8',
        '九': '9',
        '十': '10',
        '百': '100',
        '千': '1000',
        '万': '10000',
        '億': '100000000',
    }
    
    def normalize(self, amount_str: str) -> Optional[int]:
        """
        金額文字列を整数に正規化
        
        Args:
            amount_str: 金額文字列
            
        Returns:
            整数の金額、またはNone
        """
        if not amount_str:
            return None
        
        # 全角を半角に
        amount_str = self._to_halfwidth(amount_str)
        
        # 通貨記号を除去
        for symbol in self.CURRENCY_SYMBOLS:
            amount_str = amount_str.replace(symbol, '')
        
        # カンマ・スペース・ハイフン（末尾）を除去
        amount_str = amount_str.replace(',', '').replace(' ', '').rstrip('-')
        
        # 数字のみ抽出
        digits = re.sub(r'[^\d]', '', amount_str)
        
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
        
        return None
    
    def _to_halfwidth(self, text: str) -> str:
        """全角英数字を半角に変換"""
        # 全角数字
        text = text.translate(str.maketrans(
            '０１２３４５６７８９',
            '0123456789'
        ))
        # 全角カンマ
        text = text.replace('，', ',')
        return text


class OCRCorrector:
    """OCR誤認識補正クラス"""
    
    # よくある誤認識パターン（誤認識 → 正しい文字）
    COMMON_CORRECTIONS: Dict[str, str] = {
        # 数字の誤認識
        'O': '0',  # 英字Oと数字0
        'l': '1',  # 小文字Lと1
        'I': '1',  # 大文字Iと1
        'Z': '2',
        'S': '5',
        'B': '8',
        # 日本語の誤認識
        '円': '円',
        '圓': '円',
        '圆': '円',
        '閂': '円',
        '甲': '円',
        '卩': '円',
        # カタカナ
        'ー': 'ー',  # 長音記号の正規化
        '一': 'ー',  # 漢数字の一と長音記号
        # 記号
        '〇': '0',
    }
    
    # 金額コンテキストでの補正（金額文字列内のみ適用）
    AMOUNT_CORRECTIONS: Dict[str, str] = {
        'O': '0',
        'o': '0',
        'l': '1',
        'I': '1',
        'i': '1',
        'S': '5',
        's': '5',
        'B': '8',
        'Z': '2',
        'z': '2',
    }
    
    # 店舗名・会社名でよくある誤認識
    COMPANY_CORRECTIONS: Dict[str, str] = {
        '株式合社': '株式会社',
        '株式公社': '株式会社',
        '侏式会社': '株式会社',
        '有隈会社': '有限会社',
        '有眼会社': '有限会社',
    }
    
    def correct_text(self, text: str) -> NormalizationResult:
        """
        テキスト全体を補正
        
        Args:
            text: 入力テキスト
            
        Returns:
            NormalizationResult: 補正結果
        """
        original = text
        corrections = []
        
        # 会社名の補正
        for wrong, correct in self.COMPANY_CORRECTIONS.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                corrections.append(f"'{wrong}' → '{correct}'")
        
        return NormalizationResult(
            original=original,
            normalized=text,
            corrections=corrections
        )
    
    def correct_amount(self, amount_str: str) -> NormalizationResult:
        """
        金額文字列を補正
        
        Args:
            amount_str: 金額文字列
            
        Returns:
            NormalizationResult: 補正結果
        """
        original = amount_str
        corrections = []
        
        result = []
        for char in amount_str:
            if char in self.AMOUNT_CORRECTIONS:
                corrected = self.AMOUNT_CORRECTIONS[char]
                result.append(corrected)
                if char != corrected:
                    corrections.append(f"'{char}' → '{corrected}'")
            else:
                result.append(char)
        
        return NormalizationResult(
            original=original,
            normalized=''.join(result),
            corrections=corrections
        )


class TextCleaner:
    """テキストクリーニングクラス"""
    
    def clean(self, text: str) -> str:
        """
        テキストをクリーニング
        
        - 制御文字の除去
        - 連続する空白の正規化
        - 全角・半角の統一
        """
        if not text:
            return ""
        
        # 制御文字を除去（改行・タブは保持）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 連続する空白を単一スペースに
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 連続する改行を2つまでに
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 行頭・行末の空白を除去
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def normalize_whitespace(self, text: str) -> str:
        """空白文字を正規化"""
        # 全角スペースを半角に
        text = text.replace('\u3000', ' ')
        # 連続する空白を単一に
        text = re.sub(r' +', ' ', text)
        return text


class DataNormalizer:
    """データ正規化統合クラス"""
    
    def __init__(self):
        self.date_normalizer = DateNormalizer()
        self.amount_normalizer = AmountNormalizer()
        self.ocr_corrector = OCRCorrector()
        self.text_cleaner = TextCleaner()
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """日付を正規化"""
        return self.date_normalizer.normalize(date_str)
    
    def normalize_amount(self, amount_str: str) -> Optional[int]:
        """金額を正規化"""
        # まずOCR補正
        corrected = self.ocr_corrector.correct_amount(amount_str)
        # 次に金額正規化
        return self.amount_normalizer.normalize(corrected.normalized)
    
    def clean_text(self, text: str) -> str:
        """テキストをクリーニング"""
        return self.text_cleaner.clean(text)
    
    def correct_ocr_errors(self, text: str) -> NormalizationResult:
        """OCR誤認識を補正"""
        return self.ocr_corrector.correct_text(text)


# ショートカット関数
def normalize_date(date_str: str) -> Optional[str]:
    """日付正規化のショートカット"""
    return DateNormalizer().normalize(date_str)


def normalize_amount(amount_str: str) -> Optional[int]:
    """金額正規化のショートカット"""
    return AmountNormalizer().normalize(amount_str)


def clean_text(text: str) -> str:
    """テキストクリーニングのショートカット"""
    return TextCleaner().clean(text)
