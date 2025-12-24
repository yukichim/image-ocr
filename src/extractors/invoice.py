"""
請求書抽出モジュール

請求書から以下の情報を抽出:
- 請求日 / 支払期限
- 請求総額（消費税内訳含む）
- 取引先名（発行元）
- 振込先口座情報
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from ..ocr_engine import OCRResult, OCRWord


@dataclass
class BankInfo:
    """振込先口座情報"""
    bank_name: Optional[str] = None
    branch_name: Optional[str] = None
    account_type: Optional[str] = None  # 普通 / 当座
    account_number: Optional[str] = None
    account_holder: Optional[str] = None  # 口座名義


@dataclass
class TaxBreakdown:
    """消費税内訳"""
    subtotal: Optional[int] = None         # 税抜金額
    tax_amount: Optional[int] = None       # 消費税額
    total: Optional[int] = None            # 税込金額
    tax_rate: Optional[float] = None       # 税率


@dataclass
class InvoiceItem:
    """請求書の明細項目"""
    name: str
    quantity: Optional[int] = None
    unit_price: Optional[int] = None
    amount: Optional[int] = None
    confidence: float = 1.0


@dataclass
class InvoiceData:
    """請求書の抽出データ"""
    invoice_date: Optional[str] = None       # 請求日（ISO形式）
    invoice_date_raw: Optional[str] = None   # 請求日（元の文字列）
    due_date: Optional[str] = None           # 支払期限（ISO形式）
    due_date_raw: Optional[str] = None       # 支払期限（元の文字列）
    invoice_number: Optional[str] = None     # 請求書番号
    total_amount: Optional[int] = None       # 請求総額
    tax_breakdown: Optional[TaxBreakdown] = None
    vendor_name: Optional[str] = None        # 発行元名
    vendor_address: Optional[str] = None     # 発行元住所
    customer_name: Optional[str] = None      # 宛先名
    bank_info: Optional[BankInfo] = None     # 振込先
    items: List[InvoiceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "請求日": self.invoice_date,
            "支払期限": self.due_date,
            "請求総額": self.total_amount,
            "発行元名": self.vendor_name,
        }
        
        if self.invoice_number:
            result["請求書番号"] = self.invoice_number
        
        if self.customer_name:
            result["宛先名"] = self.customer_name
        
        if self.tax_breakdown:
            result["消費税内訳"] = {
                "税抜金額": self.tax_breakdown.subtotal,
                "消費税額": self.tax_breakdown.tax_amount,
                "税込金額": self.tax_breakdown.total,
            }
        
        if self.bank_info:
            result["振込先"] = {
                "銀行名": self.bank_info.bank_name,
                "支店名": self.bank_info.branch_name,
                "口座種別": self.bank_info.account_type,
                "口座番号": self.bank_info.account_number,
                "口座名義": self.bank_info.account_holder,
            }
        
        if self.items:
            result["明細"] = [
                {
                    "品名": item.name,
                    "数量": item.quantity,
                    "単価": item.unit_price,
                    "金額": item.amount,
                }
                for item in self.items
            ]
        
        return result


class InvoiceExtractor:
    """請求書情報抽出クラス"""
    
    # 日付パターン（優先順位順）
    DATE_PATTERNS = [
        # 西暦パターン
        (r'(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})[日]?', 'western'),
        # 和暦パターン
        (r'(令和|R|Ｒ)\s*(\d{1,2})[年.](\d{1,2})[月.](\d{1,2})[日]?', 'reiwa'),
        (r'(平成|H|Ｈ)\s*(\d{1,2})[年.](\d{1,2})[月.](\d{1,2})[日]?', 'heisei'),
    ]
    
    # 請求日を示すキーワード
    INVOICE_DATE_KEYWORDS = [
        '請求日', '発行日', '作成日', '請求年月日', 'DATE', 'Date'
    ]
    
    # 支払期限を示すキーワード
    DUE_DATE_KEYWORDS = [
        '支払期限', 'お支払期限', '振込期限', '期日', '支払期日',
        'お支払い期限', '入金期限', 'DUE DATE', 'Due Date'
    ]
    
    # 合計金額を示すキーワード
    TOTAL_KEYWORDS = [
        '請求金額', '御請求金額', 'ご請求金額', '請求額', 
        '合計金額', '税込合計', '請求合計', 'TOTAL'
    ]
    
    # 銀行名パターン
    BANK_PATTERNS = [
        r'([^\s]{2,10}銀行)',
        r'([^\s]{2,10}信用金庫)',
        r'([^\s]{2,10}信金)',
        r'(ゆうちょ銀行)',
        r'(楽天銀行)',
        r'(PayPay銀行)',
    ]
    
    # 支店名パターン
    BRANCH_PATTERNS = [
        r'([^\s]{2,10}支店)',
        r'([^\s]{2,10}営業所)',
        r'([^\s]{2,10}出張所)',
    ]
    
    # 口座種別
    ACCOUNT_TYPES = {
        '普通': '普通',
        '当座': '当座',
        '貯蓄': '貯蓄',
    }
    
    def __init__(self):
        pass
    
    def extract(self, ocr_result: OCRResult) -> InvoiceData:
        """
        OCR結果から請求書情報を抽出
        
        Args:
            ocr_result: OCRResult オブジェクト
            
        Returns:
            InvoiceData: 抽出された請求書データ
        """
        text = ocr_result.full_text
        words = ocr_result.words
        
        data = InvoiceData()
        
        # 1. 請求日を抽出
        date_raw, date_iso = self._extract_date_near_keywords(
            text, self.INVOICE_DATE_KEYWORDS
        )
        data.invoice_date = date_iso
        data.invoice_date_raw = date_raw
        
        # 2. 支払期限を抽出
        due_raw, due_iso = self._extract_date_near_keywords(
            text, self.DUE_DATE_KEYWORDS
        )
        data.due_date = due_iso
        data.due_date_raw = due_raw
        
        # 3. 請求書番号を抽出
        data.invoice_number = self._extract_invoice_number(text)
        
        # 4. 請求総額を抽出
        data.total_amount = self._extract_total_amount(text, words)
        
        # 5. 消費税内訳を抽出
        data.tax_breakdown = self._extract_tax_breakdown(text)
        
        # 6. 発行元（ベンダー）を抽出
        data.vendor_name = self._extract_vendor_name(text)
        
        # 7. 宛先を抽出
        data.customer_name = self._extract_customer_name(text)
        
        # 8. 振込先口座情報を抽出
        data.bank_info = self._extract_bank_info(text)
        
        # 9. 明細を抽出
        data.items = self._extract_items(text)
        
        # 10. 警告を生成
        data.warnings = self._generate_warnings(data, ocr_result)
        
        return data
    
    def _extract_date_near_keywords(self, text: str, 
                                     keywords: List[str]) -> tuple:
        """キーワード付近の日付を抽出"""
        lines = text.split('\n')
        
        for line in lines:
            # キーワードを含む行を探す
            for keyword in keywords:
                if keyword in line:
                    # 同じ行から日付を抽出
                    for pattern, pattern_type in self.DATE_PATTERNS:
                        match = re.search(pattern, line)
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
            elif pattern_type == 'reiwa':
                era_year = int(match.group(2))
                year = str(2018 + era_year)
                month, day = match.group(3), match.group(4)
            elif pattern_type == 'heisei':
                era_year = int(match.group(2))
                year = str(1988 + era_year)
                month, day = match.group(3), match.group(4)
            else:
                return None
            
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        except (ValueError, IndexError):
            return None
    
    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """請求書番号を抽出"""
        patterns = [
            r'請求書番号[:\s：]*([A-Za-z0-9\-]+)',
            r'No[.:\s]*([A-Za-z0-9\-]+)',
            r'番号[:\s：]*([A-Za-z0-9\-]+)',
            r'INVOICE\s*(?:NO|#)[.:\s]*([A-Za-z0-9\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_total_amount(self, text: str, 
                              words: List[OCRWord]) -> Optional[int]:
        """請求総額を抽出"""
        lines = text.split('\n')
        
        for line in lines:
            for keyword in self.TOTAL_KEYWORDS:
                if keyword in line:
                    amount = self._extract_amount_from_text(line)
                    if amount is not None and amount > 0:
                        return amount
        
        return None
    
    def _extract_amount_from_text(self, text: str) -> Optional[int]:
        """テキストから金額を抽出"""
        patterns = [
            r'[¥￥\\]?\s*([\d,，]+)\s*円?',
            r'([\d,，]+)\s*円',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._parse_amount(match.group(1))
        
        return None
    
    def _parse_amount(self, amount_str: str) -> Optional[int]:
        """金額文字列を整数に変換"""
        try:
            cleaned = amount_str.replace(',', '').replace('，', '')
            cleaned = cleaned.translate(str.maketrans(
                '０１２３４５６７８９', '0123456789'
            ))
            digits = re.sub(r'[^\d]', '', cleaned)
            if digits:
                return int(digits)
        except ValueError:
            pass
        return None
    
    def _extract_tax_breakdown(self, text: str) -> Optional[TaxBreakdown]:
        """消費税内訳を抽出"""
        breakdown = TaxBreakdown()
        
        # 税抜金額
        subtotal_patterns = [
            r'税抜[金額]*[:\s：]*[¥￥]?\s*([\d,，]+)',
            r'小計[:\s：]*[¥￥]?\s*([\d,，]+)',
        ]
        for pattern in subtotal_patterns:
            match = re.search(pattern, text)
            if match:
                breakdown.subtotal = self._parse_amount(match.group(1))
                break
        
        # 消費税額
        tax_patterns = [
            r'消費税[額]*[:\s：]*[¥￥]?\s*([\d,，]+)',
            r'税額[:\s：]*[¥￥]?\s*([\d,，]+)',
        ]
        for pattern in tax_patterns:
            match = re.search(pattern, text)
            if match:
                breakdown.tax_amount = self._parse_amount(match.group(1))
                break
        
        if breakdown.subtotal or breakdown.tax_amount:
            return breakdown
        
        return None
    
    def _extract_vendor_name(self, text: str) -> Optional[str]:
        """発行元（ベンダー）名を抽出"""
        lines = text.split('\n')
        
        # 「株式会社」「有限会社」などを含む行を探す
        company_patterns = [
            r'(株式会社[^\s\n]{1,30})',
            r'([^\s\n]{1,20}株式会社)',
            r'(有限会社[^\s\n]{1,30})',
            r'([^\s\n]{1,20}有限会社)',
            r'(合同会社[^\s\n]{1,30})',
        ]
        
        for line in lines:
            for pattern in company_patterns:
                match = re.search(pattern, line)
                if match:
                    company = match.group(1).strip()
                    # 宛先ではないことを確認
                    if '御中' not in line and '様' not in line:
                        return company
        
        return None
    
    def _extract_customer_name(self, text: str) -> Optional[str]:
        """宛先（顧客）名を抽出"""
        lines = text.split('\n')
        
        for line in lines:
            # 「御中」「様」を含む行
            if '御中' in line:
                # 「御中」の前の部分を取得
                parts = line.split('御中')
                if parts[0].strip():
                    return parts[0].strip() + ' 御中'
            
            if '殿' in line:
                parts = line.split('殿')
                if parts[0].strip():
                    return parts[0].strip() + ' 殿'
        
        return None
    
    def _extract_bank_info(self, text: str) -> Optional[BankInfo]:
        """振込先口座情報を抽出"""
        bank_info = BankInfo()
        
        # 銀行名
        for pattern in self.BANK_PATTERNS:
            match = re.search(pattern, text)
            if match:
                bank_info.bank_name = match.group(1)
                break
        
        # 支店名
        for pattern in self.BRANCH_PATTERNS:
            match = re.search(pattern, text)
            if match:
                bank_info.branch_name = match.group(1)
                break
        
        # 口座種別
        for key, value in self.ACCOUNT_TYPES.items():
            if key in text:
                bank_info.account_type = value
                break
        
        # 口座番号
        account_patterns = [
            r'口座番号[:\s：]*(\d{7,8})',
            r'口座[:\s：]*(\d{7,8})',
            r'(?:普通|当座)[:\s：]*(\d{7,8})',
        ]
        for pattern in account_patterns:
            match = re.search(pattern, text)
            if match:
                bank_info.account_number = match.group(1)
                break
        
        # 口座名義
        holder_patterns = [
            r'口座名義[:\s：]*([^\n\r]{2,30})',
            r'名義[:\s：]*([^\n\r]{2,30})',
        ]
        for pattern in holder_patterns:
            match = re.search(pattern, text)
            if match:
                bank_info.account_holder = match.group(1).strip()
                break
        
        # 最低限の情報があれば返す
        if bank_info.bank_name or bank_info.account_number:
            return bank_info
        
        return None
    
    def _extract_items(self, text: str) -> List[InvoiceItem]:
        """明細を抽出"""
        items = []
        lines = text.split('\n')
        
        # 表形式の明細を検出するパターン
        # 品名 | 数量 | 単価 | 金額 のような形式
        item_pattern = re.compile(
            r'(.{2,30}?)\s+(\d+)\s+[¥￥]?([\d,，]+)\s+[¥￥]?([\d,，]+)'
        )
        
        for line in lines:
            match = item_pattern.search(line)
            if match:
                name = match.group(1).strip()
                
                # 明らかにヘッダー行や合計行を除外
                if any(kw in name for kw in ['品名', '商品', '数量', '単価', '合計', '小計']):
                    continue
                
                quantity = int(match.group(2))
                unit_price = self._parse_amount(match.group(3))
                amount = self._parse_amount(match.group(4))
                
                if name and amount:
                    items.append(InvoiceItem(
                        name=name,
                        quantity=quantity,
                        unit_price=unit_price,
                        amount=amount
                    ))
        
        return items
    
    def _generate_warnings(self, data: InvoiceData, 
                          ocr_result: OCRResult) -> List[str]:
        """警告メッセージを生成"""
        warnings = []
        
        # 必須項目の欠落チェック
        if not data.invoice_date:
            warnings.append("請求日を抽出できませんでした")
        
        if not data.total_amount:
            warnings.append("請求総額を抽出できませんでした")
        
        if not data.vendor_name:
            warnings.append("発行元名を抽出できませんでした")
        
        if not data.bank_info:
            warnings.append("振込先口座情報を抽出できませんでした")
        elif not data.bank_info.account_number:
            warnings.append("口座番号を抽出できませんでした")
        
        # OCR信頼度が低い場合
        if ocr_result.average_confidence < 70:
            warnings.append(
                f"OCR信頼度が低いです（{ocr_result.average_confidence:.1f}%）"
            )
        
        return warnings
