#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル事前ダウンロードスクリプト

全てのOCRエンジンのモデルを事前にダウンロードします。
初回実行時の待ち時間を解消するために使用します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engines import EngineType, create_engine


def download_all_models():
    """全エンジンのモデルをダウンロード"""
    print("\n" + "="*70)
    print("OCRエンジンモデル 事前ダウンロード")
    print("="*70)
    print("\n初回実行時は各エンジンのモデルダウンロードに時間がかかります。")
    print("ダウンロード済みのモデルはスキップされます。\n")
    
    engines = [
        (EngineType.TESSERACT, "軽量・高速な汎用OCR"),
        (EngineType.ONNXOCR, "PP-OCRv5 ONNX版（日本語特化）"),
        (EngineType.PADDLEOCR, "PaddleOCR（高精度）"),
        (EngineType.PADDLE_VL, "PaddleOCR-VL（レイアウト解析）"),
    ]
    
    results = []
    
    for engine_type, description in engines:
        print(f"\n{'='*70}")
        print(f"📦 {engine_type.display_name}")
        print(f"   {description}")
        print(f"{'='*70}\n")
        
        try:
            engine = create_engine(engine_type)
            
            if engine.is_available:
                print(f"✓ {engine_type.display_name} - モデル準備完了")
                results.append((engine_type.display_name, "成功", "✓"))
            else:
                print(f"✗ {engine_type.display_name} - 利用不可")
                results.append((engine_type.display_name, "利用不可", "✗"))
                
        except Exception as e:
            print(f"✗ エラー: {e}")
            results.append((engine_type.display_name, f"エラー: {e}", "✗"))
    
    # サマリー表示
    print(f"\n\n{'='*70}")
    print("ダウンロード結果サマリー")
    print(f"{'='*70}\n")
    
    for name, status, icon in results:
        print(f"{icon} {name:25s} - {status}")
    
    print(f"\n{'='*70}")
    print("完了！GUIアプリケーションを起動できます。")
    print(f"{'='*70}\n")
    
    # 成功数をカウント
    success_count = sum(1 for _, _, icon in results if icon == "✓")
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count} エンジン")
    
    return success_count == total_count


def download_specific_engine(engine_name: str):
    """特定のエンジンのみダウンロード"""
    try:
        engine_type = EngineType.from_string(engine_name.lower())
        
        print(f"\n{'='*70}")
        print(f"📦 {engine_type.display_name} モデルをダウンロード中...")
        print(f"{'='*70}\n")
        
        engine = create_engine(engine_type)
        
        if engine.is_available:
            print(f"\n✓ {engine_type.display_name} - ダウンロード完了")
            return True
        else:
            print(f"\n✗ {engine_type.display_name} - 利用不可")
            return False
            
    except ValueError:
        print(f"✗ エラー: 不明なエンジン名 '{engine_name}'")
        print(f"\n利用可能なエンジン:")
        for et in EngineType:
            print(f"  - {et.value}")
        return False
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OCRエンジンのモデルを事前ダウンロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全エンジンのモデルをダウンロード
  python scripts/download_models.py
  
  # 特定のエンジンのみダウンロード
  python scripts/download_models.py --engine paddleocr
  python scripts/download_models.py --engine paddle_vl
  python scripts/download_models.py --engine onnxocr
        """
    )
    
    parser.add_argument(
        '--engine',
        type=str,
        help='特定のエンジンのみダウンロード (tesseract, onnxocr, paddleocr, paddleocr_vl)',
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        if args.engine:
            success = download_specific_engine(args.engine)
        else:
            success = download_all_models()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
