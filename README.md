# 税務・経理書類特化型 OCRエンジン

領収書・請求書などの税務書類から構造化データ（JSON）を抽出するOCRエンジンです。

## 機能

### MVP対応書類
- **領収書・レシート**: 日付、合計金額、取引先名、明細、軽減税率判定
- **請求書**: 請求日、支払期限、請求総額、取引先名、振込先口座情報

### 処理パイプライン
1. 画像前処理（傾き補正、二値化、ノイズ除去）
2. 文書分類（ルールベース）
3. OCR文字認識（Tesseract LSTM）
4. 意味抽出（Key-Value Extraction）
5. データ正規化（和暦→西暦、金額整形）

## セットアップ

### 1. Tesseract OCRのインストール

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert
```

#### macOS
```bash
brew install tesseract tesseract-lang
```

#### Windows
[Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) からインストーラをダウンロード

### 2. Python依存関係のインストール

```bash
pip install -e .
```

開発用:
```bash
pip install -e ".[dev]"
```

## 使い方

### コマンドライン

```bash
# 単一ファイルの処理
tax-ocr path/to/receipt.jpg

# 出力先を指定
tax-ocr path/to/receipt.jpg -o output.json

# 文書タイプを指定（自動判定をスキップ）
tax-ocr path/to/invoice.pdf --type invoice
```

### Pythonから利用

```python
from src.pipeline import OCRPipeline

pipeline = OCRPipeline()
result = pipeline.process("path/to/receipt.jpg")

print(result.to_json())
```

## 出力形式

### 領収書
```json
{
  "document_type": "receipt",
  "confidence": 0.95,
  "data": {
    "date": "2024-01-15",
    "total_amount": 1980,
    "store_name": "コンビニエンスストア○○",
    "items": [
      {"name": "おにぎり", "price": 150, "reduced_tax": true},
      {"name": "お茶", "price": 130, "reduced_tax": true}
    ],
    "tax_details": {
      "rate_8_percent": 280,
      "rate_10_percent": 1700
    }
  },
  "warnings": []
}
```

### 請求書
```json
{
  "document_type": "invoice",
  "confidence": 0.92,
  "data": {
    "invoice_date": "2024-01-20",
    "due_date": "2024-02-29",
    "total_amount": 55000,
    "tax_amount": 5000,
    "vendor_name": "株式会社○○",
    "bank_info": {
      "bank_name": "○○銀行",
      "branch_name": "△△支店",
      "account_type": "普通",
      "account_number": "1234567"
    }
  },
  "warnings": []
}
```

## ライセンス

MIT License
