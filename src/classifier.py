"""
文書分類モジュール

ルールベースで文書タイプを判定:
- キーワード頻度分析
- レイアウト特徴（アスペクト比）
- 対応タイプ: receipt（領収書）, invoice（請求書）
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re


class DocumentType(Enum):
    """文書タイプ"""
    RECEIPT = "receipt"          # 領収書・レシート
    INVOICE = "invoice"          # 請求書
    UNKNOWN = "unknown"          # 不明


@dataclass
class ClassificationResult:
    """分類結果"""
    document_type: DocumentType
    confidence: float
    scores: Dict[str, float]
    matched_keywords: Dict[str, List[str]]
    reasoning: str


class DocumentClassifier:
    """文書分類クラス（ルールベース）"""
    
    # 領収書に特徴的なキーワード（重み付き）
    RECEIPT_KEYWORDS: Dict[str, float] = {
        # 高確度キーワード
        "領収書": 3.0,
        "領収証": 3.0,
        "レシート": 3.0,
        "RECEIPT": 2.5,
        # 中確度キーワード
        "合計": 1.5,
        "小計": 1.5,
        "税込": 1.5,
        "税抜": 1.5,
        "内税": 1.5,
        "外税": 1.5,
        "お預り": 1.2,
        "お釣り": 1.2,
        "おつり": 1.2,
        "現金": 1.0,
        "クレジット": 1.0,
        "電子マネー": 1.0,
        # 軽減税率関連
        "軽減税率": 1.5,
        "8%対象": 1.5,
        "10%対象": 1.2,
        "※": 0.8,
        # 店舗関連
        "店舗": 0.8,
        "TEL": 0.8,
        "電話": 0.5,
        "いらっしゃいませ": 1.0,
        "ありがとうございました": 1.0,
        # 日時関連
        "登録": 0.5,
        "精算": 0.8,
    }
    
    # 請求書に特徴的なキーワード（重み付き）
    INVOICE_KEYWORDS: Dict[str, float] = {
        # 高確度キーワード
        "請求書": 3.5,
        "御請求書": 3.5,
        "INVOICE": 3.0,
        "請求金額": 3.0,
        # 中確度キーワード
        "支払期限": 2.0,
        "お支払期限": 2.0,
        "振込期限": 2.0,
        "期日": 1.5,
        "振込先": 2.5,
        "お振込先": 2.5,
        "銀行": 2.0,
        "口座番号": 2.5,
        "口座": 1.5,
        "普通": 1.0,
        "当座": 1.0,
        "支店": 1.5,
        # 宛先関連
        "御中": 2.0,
        "様": 0.5,
        "宛": 1.0,
        "殿": 1.5,
        # 金額関連
        "税抜金額": 1.5,
        "消費税額": 1.5,
        "合計金額": 1.5,
        "小計": 1.0,
        # 取引関連
        "納品": 1.0,
        "取引": 0.8,
        "品目": 0.8,
        "数量": 0.8,
        "単価": 0.8,
        # 発行関連
        "発行日": 1.5,
        "請求日": 2.0,
    }
    
    # 領収書に特有の否定キーワード（これがあると領収書ではない可能性が高い）
    RECEIPT_NEGATIVE_KEYWORDS: Dict[str, float] = {
        "請求書": 2.0,
        "振込先": 1.5,
        "口座番号": 1.5,
        "支払期限": 1.5,
    }
    
    # 請求書に特有の否定キーワード
    INVOICE_NEGATIVE_KEYWORDS: Dict[str, float] = {
        "レシート": 2.0,
        "お預り": 1.5,
        "お釣り": 1.5,
    }
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Args:
            confidence_threshold: 分類確信度の閾値（これ未満はUNKNOWN）
        """
        self.confidence_threshold = confidence_threshold
    
    def classify(self, text: str, 
                 aspect_ratio: Optional[float] = None) -> ClassificationResult:
        """
        テキストから文書タイプを分類
        
        Args:
            text: OCR結果のテキスト
            aspect_ratio: 画像のアスペクト比（高さ/幅）。レシートは通常縦長。
            
        Returns:
            ClassificationResult: 分類結果
        """
        # 正規化
        text_normalized = self._normalize_text(text)
        
        # 各タイプのスコアを計算
        receipt_score, receipt_matches = self._calculate_score(
            text_normalized,
            self.RECEIPT_KEYWORDS,
            self.RECEIPT_NEGATIVE_KEYWORDS
        )
        
        invoice_score, invoice_matches = self._calculate_score(
            text_normalized,
            self.INVOICE_KEYWORDS,
            self.INVOICE_NEGATIVE_KEYWORDS
        )
        
        # アスペクト比による補正
        if aspect_ratio is not None:
            if aspect_ratio > 2.0:  # 縦長（レシートの特徴）
                receipt_score *= 1.2
            elif aspect_ratio < 0.8:  # 横長（請求書の特徴）
                invoice_score *= 1.1
        
        # スコアを正規化
        total_score = receipt_score + invoice_score + 0.001  # ゼロ除算防止
        receipt_confidence = receipt_score / total_score
        invoice_confidence = invoice_score / total_score
        
        # 分類結果を決定
        scores = {
            "receipt": receipt_confidence,
            "invoice": invoice_confidence,
        }
        
        matched_keywords = {
            "receipt": receipt_matches,
            "invoice": invoice_matches,
        }
        
        # 最高スコアのタイプを選択
        if receipt_confidence > invoice_confidence:
            doc_type = DocumentType.RECEIPT
            confidence = receipt_confidence
            reasoning = self._build_reasoning("領収書", receipt_matches, receipt_confidence)
        else:
            doc_type = DocumentType.INVOICE
            confidence = invoice_confidence
            reasoning = self._build_reasoning("請求書", invoice_matches, invoice_confidence)
        
        # 閾値未満の場合はUNKNOWN
        if confidence < self.confidence_threshold:
            doc_type = DocumentType.UNKNOWN
            reasoning = f"確信度が閾値未満（{confidence:.2%} < {self.confidence_threshold:.2%}）"
        
        return ClassificationResult(
            document_type=doc_type,
            confidence=confidence,
            scores=scores,
            matched_keywords=matched_keywords,
            reasoning=reasoning
        )
    
    def _normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        # 全角英数字を半角に
        text = text.translate(str.maketrans(
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        ))
        return text.upper()
    
    def _calculate_score(self, text: str,
                        positive_keywords: Dict[str, float],
                        negative_keywords: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        キーワードマッチングでスコアを計算
        
        Returns:
            (スコア, マッチしたキーワードのリスト)
        """
        score = 0.0
        matched = []
        
        # ポジティブキーワード
        for keyword, weight in positive_keywords.items():
            keyword_upper = keyword.upper()
            count = text.count(keyword_upper)
            if count > 0:
                # 出現回数に応じてスコア加算（ただし減衰あり）
                score += weight * (1 + 0.3 * (count - 1))
                matched.append(keyword)
        
        # ネガティブキーワード（減点）
        for keyword, weight in negative_keywords.items():
            keyword_upper = keyword.upper()
            if keyword_upper in text:
                score -= weight * 0.5  # 減点は控えめに
        
        return max(0.0, score), matched
    
    def _build_reasoning(self, doc_type_name: str, 
                        matches: List[str], 
                        confidence: float) -> str:
        """分類理由の説明文を生成"""
        if not matches:
            return f"{doc_type_name}と判定（キーワードマッチなし、confidence: {confidence:.2%}）"
        
        top_matches = matches[:5]
        keywords_str = ", ".join(f'"{k}"' for k in top_matches)
        
        return (f"{doc_type_name}と判定 "
               f"(confidence: {confidence:.2%}, "
               f"検出キーワード: {keywords_str})")


def classify_document(text: str, 
                     aspect_ratio: Optional[float] = None) -> ClassificationResult:
    """
    文書分類のショートカット関数
    
    Args:
        text: OCR結果のテキスト
        aspect_ratio: 画像のアスペクト比（高さ/幅）
        
    Returns:
        ClassificationResult: 分類結果
    """
    classifier = DocumentClassifier()
    return classifier.classify(text, aspect_ratio)
