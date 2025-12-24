"""
領収書・レシート抽出モジュール

領収書から以下の情報を抽出:
- 日付（和暦・西暦対応）
- 合計金額
- 取引先名（店舗名）
- 明細（品名・単価）
- 軽減税率フラグ
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from ..ocr_engine import OCRResult, OCRWord


@dataclass
class ReceiptItem:
    """領収書の明細項目"""
    name: str
    price: Optional[int]
    quantity: int = 1
    reduced_tax: bool = False  # 軽減税率対象
    confidence: float = 1.0


@dataclass
class TaxDetails:
    """税額詳細"""
    rate_8_percent: Optional[int] = None   # 8%対象額
    rate_10_percent: Optional[int] = None  # 10%対象額
    tax_8_percent: Optional[int] = None    # 8%税額
    tax_10_percent: Optional[int] = None   # 10%税額


@dataclass
class ReceiptData:
    """領収書の抽出データ"""
    date: Optional[str] = None              # ISO形式 (YYYY-MM-DD)
    date_raw: Optional[str] = None          # 元の日付文字列
    total_amount: Optional[int] = None      # 合計金額
    subtotal: Optional[int] = None          # 小計
    store_name: Optional[str] = None        # 店舗名
    store_address: Optional[str] = None     # 店舗住所
    store_phone: Optional[str] = None       # 店舗電話番号
    items: List[ReceiptItem] = field(default_factory=list)
    tax_details: Optional[TaxDetails] = None
    payment_method: Optional[str] = None    # 支払方法
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "日付": self.date,
            "合計金額": self.total_amount,
            "店舗名": self.store_name,
        }
        
        if self.subtotal:
            result["小計"] = self.subtotal
        
        if self.store_address:
            result["店舗住所"] = self.store_address
            
        if self.store_phone:
            result["店舗電話番号"] = self.store_phone
        
        if self.items:
            result["明細"] = [
                {
                    "品名": item.name,
                    "価格": item.price,
                    "数量": item.quantity,
                    "軽減税率": item.reduced_tax,
                }
                for item in self.items
            ]
        
        if self.tax_details:
            result["税額詳細"] = {
                "8%対象額": self.tax_details.rate_8_percent,
                "10%対象額": self.tax_details.rate_10_percent,
            }
        
        if self.payment_method:
            result["支払方法"] = self.payment_method
        
        return result


class ReceiptExtractor:
    """領収書情報抽出クラス"""
    
    # 日付パターン（優先順位順）
    DATE_PATTERNS = [
        # 西暦パターン
        (r'(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})[日]?', 'western'),
        (r'(\d{4})(\d{2})(\d{2})', 'western_compact'),
        # 和暦パターン
        (r'(令和|R|Ｒ)\s*(\d{1,2})[年.](\d{1,2})[月.](\d{1,2})[日]?', 'reiwa'),
        (r'(平成|H|Ｈ)\s*(\d{1,2})[年.](\d{1,2})[月.](\d{1,2})[日]?', 'heisei'),
        (r'(昭和|S|Ｓ)\s*(\d{1,2})[年.](\d{1,2})[月.](\d{1,2})[日]?', 'showa'),
    ]
    
    # 金額パターン
    AMOUNT_PATTERNS = [
        r'[¥￥\\]?\s*([\d,，]+)\s*円?',
        r'([\d,，]+)\s*円',
        r'[¥￥\\]\s*([\d,，]+)',
    ]
    
    # 合計金額を示すキーワード
    TOTAL_KEYWORDS = [
        '合計', '計', 'TOTAL', 'Total', '税込合計', '税込計',
        'お会計', 'お買上合計', '買上合計', '総合計', 'ご請求額'
    ]
    
    # 小計を示すキーワード
    SUBTOTAL_KEYWORDS = ['小計', '税抜合計', '税抜計', 'SUBTOTAL']
    
    # 軽減税率を示すマーカー
    REDUCED_TAX_MARKERS = ['※', '＊', '*', '軽', '軽減', '8%', '８%', '8％', '８％']
    
    # 支払方法のキーワード
    PAYMENT_KEYWORDS = {
        '現金': '現金',
        'CASH': '現金',
        'クレジット': 'クレジットカード',
        'CREDIT': 'クレジットカード',
        'カード': 'クレジットカード',
        '電子マネー': '電子マネー',
        'Suica': '電子マネー',
        'PASMO': '電子マネー',
        'PayPay': 'QRコード決済',
        'd払い': 'QRコード決済',
        'au PAY': 'QRコード決済',
        'QR': 'QRコード決済',
    }
    
    def __init__(self):
        pass
    
    def extract(self, ocr_result: OCRResult) -> ReceiptData:
        """
        OCR結果から領収書情報を抽出
        
        Args:
            ocr_result: OCRResult オブジェクト
            
        Returns:
            ReceiptData: 抽出された領収書データ
        """
        text = ocr_result.full_text
        words = ocr_result.words
        
        data = ReceiptData()
        
        # 1. 日付を抽出
        date_raw, date_iso = self._extract_date(text)
        data.date = date_iso
        data.date_raw = date_raw
        
        # 2. 合計金額を抽出
        data.total_amount = self._extract_total_amount(text, words)
        
        # 3. 小計を抽出
        data.subtotal = self._extract_subtotal(text)
        
        # 4. 店舗名を抽出（通常、最上部にある）
        data.store_name = self._extract_store_name(text, words)
        
        # 5. 電話番号を抽出
        data.store_phone = self._extract_phone(text)
        
        # 6. 明細を抽出
        data.items = self._extract_items(text)
        
        # 7. 税額詳細を抽出
        data.tax_details = self._extract_tax_details(text)
        
        # 8. 支払方法を抽出
        data.payment_method = self._extract_payment_method(text)
        
        # 9. 警告を生成
        data.warnings = self._generate_warnings(data, ocr_result)
        
        return data
    
    def _extract_date(self, text: str) -> tuple:
        """日付を抽出"""
        for pattern, pattern_type in self.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                raw_date = match.group(0)
                iso_date = self._convert_date_to_iso(match, pattern_type)
                return raw_date, iso_date
        
        return None, None
    
    def _convert_date_to_iso(self, match, pattern_type: str) -> Optional[str]:
        """日付をISO形式に変換"""
        try:
            if pattern_type == 'western':
                year, month, day = match.group(1), match.group(2), match.group(3)
            elif pattern_type == 'western_compact':
                year, month, day = match.group(1), match.group(2), match.group(3)
            elif pattern_type == 'reiwa':
                era_year = int(match.group(2))
                year = str(2018 + era_year)  # 令和1年 = 2019年
                month, day = match.group(3), match.group(4)
            elif pattern_type == 'heisei':
                era_year = int(match.group(2))
                year = str(1988 + era_year)  # 平成1年 = 1989年
                month, day = match.group(3), match.group(4)
            elif pattern_type == 'showa':
                era_year = int(match.group(2))
                year = str(1925 + era_year)  # 昭和1年 = 1926年
                month, day = match.group(3), match.group(4)
            else:
                return None
            
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        except (ValueError, IndexError):
            return None
    
    def _extract_total_amount(self, text: str, words: List[OCRWord]) -> Optional[int]:
        """合計金額を抽出"""
        lines = text.split('\n')
        
        for line in lines:
            # 合計キーワードを含む行を探す
            for keyword in self.TOTAL_KEYWORDS:
                if keyword in line:
                    # 同じ行から金額を抽出
                    amount = self._extract_amount_from_text(line)
                    if amount is not None and amount > 0:
                        return amount
        
        # キーワードベースで見つからない場合、座標ベースで探す
        for keyword in self.TOTAL_KEYWORDS:
            amount_text = self._find_value_near_keyword(words, keyword)
            if amount_text:
                amount = self._parse_amount(amount_text)
                if amount is not None:
                    return amount
        
        return None
    
    def _extract_subtotal(self, text: str) -> Optional[int]:
        """小計を抽出"""
        lines = text.split('\n')
        
        for line in lines:
            for keyword in self.SUBTOTAL_KEYWORDS:
                if keyword in line:
                    amount = self._extract_amount_from_text(line)
                    if amount is not None:
                        return amount
        
        return None
    
    def _extract_amount_from_text(self, text: str) -> Optional[int]:
        """テキストから金額を抽出"""
        for pattern in self.AMOUNT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return self._parse_amount(match.group(1))
        return None
    
    def _parse_amount(self, amount_str: str) -> Optional[int]:
        """金額文字列を整数に変換"""
        try:
            # カンマ、全角数字を除去
            cleaned = amount_str.replace(',', '').replace('，', '')
            cleaned = cleaned.translate(str.maketrans(
                '０１２３４５６７８９', '0123456789'
            ))
            # 数字のみ抽出
            digits = re.sub(r'[^\d]', '', cleaned)
            if digits:
                return int(digits)
        except ValueError:
            pass
        return None
    
    def _extract_store_name(self, text: str, words: List[OCRWord]) -> Optional[str]:
        """店舗名を抽出（通常は上部にある）"""
        lines = text.split('\n')
        
        # 上部の行から店舗名らしきものを探す
        for i, line in enumerate(lines[:5]):  # 上位5行
            line = line.strip()
            if not line:
                continue
            
            # 除外パターン
            if any(x in line for x in ['領収書', '領収証', 'レシート', '登録番号']):
                continue
            
            # 日付っぽい行は除外
            if re.search(r'\d{4}[年/\-.]', line):
                continue
            
            # 電話番号だけの行は除外
            if re.match(r'^[\d\-\(\)]+$', line.replace(' ', '')):
                continue
            
            # 2文字以上で、店舗名らしければ採用
            if len(line) >= 2:
                return line
        
        return None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """電話番号を抽出"""
        # 日本の電話番号パターン
        patterns = [
            r'(?:TEL|Tel|tel|電話)?[:\s]*(\d{2,4}[-\s]?\d{2,4}[-\s]?\d{3,4})',
            r'(\d{3}-\d{4}-\d{4})',  # 携帯
            r'(\d{2,4}-\d{2,4}-\d{4})',  # 固定
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group(1)
                # 正規化
                phone = re.sub(r'[^\d\-]', '', phone)
                if len(phone.replace('-', '')) >= 10:
                    return phone
        
        return None
    
    def _extract_items(self, text: str) -> List[ReceiptItem]:
        """明細を抽出"""
        items = []
        lines = text.split('\n')
        
        in_item_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 合計行に到達したら終了
            if any(kw in line for kw in self.TOTAL_KEYWORDS):
                if '小計' not in line:  # 小計は除外
                    break
            
            # 明細っぽい行を検出（品名と金額が含まれる）
            # パターン: 品名 数量 金額 or 品名 金額
            item = self._parse_item_line(line)
            if item:
                items.append(item)
        
        return items
    
    def _parse_item_line(self, line: str) -> Optional[ReceiptItem]:
        """明細行をパース"""
        # 軽減税率マーカーをチェック
        reduced_tax = any(marker in line for marker in self.REDUCED_TAX_MARKERS)
        
        # 金額パターン
        amount_match = re.search(r'[¥￥\\]?\s*([\d,，]+)\s*円?$', line)
        if not amount_match:
            # 行末が金額でない場合も試行
            amount_match = re.search(r'[¥￥\\]([\d,，]+)', line)
        
        if amount_match:
            price = self._parse_amount(amount_match.group(1))
            
            # 品名を抽出（金額部分を除去）
            name_part = line[:amount_match.start()].strip()
            
            # 軽減税率マーカーを除去
            for marker in self.REDUCED_TAX_MARKERS:
                name_part = name_part.replace(marker, '').strip()
            
            # 数量を抽出
            quantity = 1
            qty_match = re.search(r'[×x]\s*(\d+)', name_part)
            if qty_match:
                quantity = int(qty_match.group(1))
                name_part = name_part[:qty_match.start()].strip()
            
            if name_part and price and price > 0:
                # 明らかに明細でないものを除外
                if len(name_part) >= 1 and not any(
                    kw in name_part for kw in ['合計', '小計', '税', '預り', '釣り']
                ):
                    return ReceiptItem(
                        name=name_part,
                        price=price,
                        quantity=quantity,
                        reduced_tax=reduced_tax
                    )
        
        return None
    
    def _extract_tax_details(self, text: str) -> Optional[TaxDetails]:
        """税額詳細を抽出"""
        details = TaxDetails()
        
        # 8%対象額
        match_8 = re.search(r'(?:8%|８%|8％|８％).*?[¥￥]?\s*([\d,，]+)', text)
        if match_8:
            details.rate_8_percent = self._parse_amount(match_8.group(1))
        
        # 10%対象額
        match_10 = re.search(r'(?:10%|１０%|10％|１０％).*?[¥￥]?\s*([\d,，]+)', text)
        if match_10:
            details.rate_10_percent = self._parse_amount(match_10.group(1))
        
        if details.rate_8_percent or details.rate_10_percent:
            return details
        
        return None
    
    def _extract_payment_method(self, text: str) -> Optional[str]:
        """支払方法を抽出"""
        text_upper = text.upper()
        
        for keyword, method in self.PAYMENT_KEYWORDS.items():
            if keyword.upper() in text_upper:
                return method
        
        return None
    
    def _find_value_near_keyword(self, words: List[OCRWord], 
                                  keyword: str) -> Optional[str]:
        """キーワードの近くにある値を取得"""
        keyword_words = [w for w in words if keyword in w.text]
        
        if not keyword_words:
            return None
        
        keyword_word = keyword_words[0]
        
        # 右側にある単語を探す
        candidates = []
        for word in words:
            if (word.bbox.left > keyword_word.bbox.right and
                abs(word.bbox.top - keyword_word.bbox.top) < 30):
                distance = word.bbox.left - keyword_word.bbox.right
                if distance < 200:
                    candidates.append((distance, word))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1].text
        
        return None
    
    def _generate_warnings(self, data: ReceiptData, 
                          ocr_result: OCRResult) -> List[str]:
        """警告メッセージを生成"""
        warnings = []
        
        # 必須項目の欠落チェック
        if not data.date:
            warnings.append("日付を抽出できませんでした")
        
        if not data.total_amount:
            warnings.append("合計金額を抽出できませんでした")
        
        if not data.store_name:
            warnings.append("店舗名を抽出できませんでした")
        
        # OCR信頼度が低い場合
        if ocr_result.average_confidence < 70:
            warnings.append(
                f"OCR信頼度が低いです（{ocr_result.average_confidence:.1f}%）"
            )
        
        # 軽減税率の整合性チェック
        reduced_items = [i for i in data.items if i.reduced_tax]
        if reduced_items and not data.tax_details:
            warnings.append("軽減税率対象品目がありますが、税額詳細が見つかりません")
        
        return warnings
