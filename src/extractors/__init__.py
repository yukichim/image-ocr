"""文書タイプ別の意味抽出モジュール"""

from .receipt import ReceiptExtractor
from .invoice import InvoiceExtractor

__all__ = ["ReceiptExtractor", "InvoiceExtractor"]
