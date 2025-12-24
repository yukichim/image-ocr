"""
税務・経理書類特化型OCRエンジン - メインエントリーポイント

コマンドラインインターフェース:
    tax-ocr <image_path> [options]

オプション:
    -o, --output    出力ファイルパス（省略時は標準出力）
    -t, --type      文書タイプ（receipt/invoice）
    --debug         デバッグ情報を出力
    --no-color      カラー出力を無効化
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .pipeline import OCRPipeline, PipelineConfig, PipelineResult
from .classifier import DocumentType


def create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        prog="tax-ocr",
        description="税務・経理書類特化型OCRエンジン - 領収書・請求書から構造化データを抽出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  tax-ocr receipt.jpg                    # 自動判定で処理
  tax-ocr invoice.png -t invoice         # 請求書として処理
  tax-ocr receipt.jpg -o result.json     # 結果をファイルに出力
  tax-ocr receipt.jpg --debug            # デバッグ情報付きで出力
        """
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="処理する画像ファイルのパス"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="出力ファイルパス（省略時は標準出力）"
    )
    
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["receipt", "invoice", "auto"],
        default="auto",
        help="文書タイプ（receipt: 領収書, invoice: 請求書, auto: 自動判定）"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグ情報を含めて出力"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="カラー出力を無効化"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser


def print_colored(text: str, color: str, no_color: bool = False) -> None:
    """カラー出力"""
    if no_color:
        print(text)
        return
    
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def format_result_summary(result: PipelineResult, no_color: bool = False) -> str:
    """結果のサマリーを整形"""
    lines = []
    
    if result.success:
        status = "✓ 処理成功"
        color = "green"
    else:
        status = "✗ 処理失敗"
        color = "red"
    
    lines.append(f"\n{'='*50}")
    lines.append(f"  {status}")
    lines.append(f"{'='*50}")
    
    lines.append(f"  文書タイプ: {result.document_type.value}")
    lines.append(f"  分類信頼度: {result.confidence:.1%}")
    lines.append(f"  OCR信頼度:  {result.ocr_confidence:.1f}%")
    
    if result.data:
        lines.append(f"\n  --- 抽出データ ---")
        data_dict = result.data.to_dict()
        for key, value in data_dict.items():
            if value is not None:
                lines.append(f"  {key}: {value}")
    
    if result.warnings:
        lines.append(f"\n  --- 警告 ({len(result.warnings)}件) ---")
        for w in result.warnings[:5]:
            lines.append(f"  ⚠ {w}")
    
    if result.errors:
        lines.append(f"\n  --- エラー ({len(result.errors)}件) ---")
        for e in result.errors:
            lines.append(f"  ✗ {e}")
    
    lines.append("")
    
    return "\n".join(lines)


def process_file(image_path: str, 
                 document_type: Optional[str] = None,
                 debug: bool = False) -> PipelineResult:
    """ファイルを処理"""
    config = PipelineConfig()
    pipeline = OCRPipeline(config)
    
    doc_type = None
    if document_type and document_type != "auto":
        if document_type == "receipt":
            doc_type = DocumentType.RECEIPT
        elif document_type == "invoice":
            doc_type = DocumentType.INVOICE
    
    return pipeline.process(image_path, doc_type)


def main() -> int:
    """メインエントリーポイント"""
    parser = create_parser()
    args = parser.parse_args()
    
    # ファイル存在確認
    image_path = Path(args.image_path)
    if not image_path.exists():
        print_colored(f"エラー: ファイルが見つかりません: {args.image_path}", 
                     "red", args.no_color)
        return 1
    
    # 対応形式確認
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if image_path.suffix.lower() not in supported_extensions:
        print_colored(
            f"エラー: 非対応の画像形式です: {image_path.suffix}\n"
            f"対応形式: {', '.join(supported_extensions)}",
            "red", args.no_color
        )
        return 1
    
    # 処理実行
    print_colored(f"処理中: {args.image_path}", "blue", args.no_color)
    
    try:
        result = process_file(
            str(image_path),
            document_type=args.type,
            debug=args.debug
        )
    except Exception as e:
        print_colored(f"エラー: {e}", "red", args.no_color)
        return 1
    
    # 結果出力
    if args.output:
        # ファイルに出力
        output_path = Path(args.output)
        
        if args.debug:
            output_data = result.to_dict()
        else:
            # デバッグ情報を除外
            output_data = result.to_dict()
            output_data.pop("_debug", None)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print_colored(f"出力完了: {args.output}", "green", args.no_color)
        
        # サマリーも表示
        print(format_result_summary(result, args.no_color))
    else:
        # 標準出力
        if args.debug:
            output_data = result.to_dict()
        else:
            output_data = result.to_dict()
            output_data.pop("_debug", None)
        
        # サマリー表示
        print(format_result_summary(result, args.no_color))
        
        # JSON出力
        print("\n--- JSON出力 ---")
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
