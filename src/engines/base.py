"""
OCRエンジン抽象基底クラス

全てのOCRエンジンが実装すべきインターフェースを定義
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class BoundingBox:
    """テキストの座標情報"""
    left: int
    top: int
    width: int
    height: int
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def bottom(self) -> int:
        return self.top + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)
    
    def contains_point(self, x: int, y: int) -> bool:
        """点が領域内にあるか判定"""
        return (self.left <= x <= self.right and 
                self.top <= y <= self.bottom)
    
    def overlaps(self, other: 'BoundingBox') -> bool:
        """他の領域と重なっているか判定"""
        return not (self.right < other.left or 
                   other.right < self.left or
                   self.bottom < other.top or 
                   other.bottom < self.top)
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """他の領域との中心間距離"""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    
    @classmethod
    def from_points(cls, points: List[List[float]]) -> 'BoundingBox':
        """4点座標からBoundingBoxを作成（OnnxOCR/PaddleOCR形式）"""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        left = int(min(xs))
        top = int(min(ys))
        right = int(max(xs))
        bottom = int(max(ys))
        return cls(left, top, right - left, bottom - top)


@dataclass
class OCRWord:
    """認識された単語/テキスト"""
    text: str
    confidence: float
    bbox: BoundingBox
    block_num: int = 0
    line_num: int = 0
    word_num: int = 0


@dataclass
class OCRLine:
    """認識された行"""
    text: str
    words: List[OCRWord]
    confidence: float
    bbox: BoundingBox
    block_num: int = 0
    line_num: int = 0


@dataclass
class OCRBlock:
    """認識されたテキストブロック"""
    text: str
    lines: List[OCRLine]
    confidence: float
    bbox: BoundingBox
    block_num: int = 0


@dataclass
class OCRResult:
    """OCR結果"""
    full_text: str
    blocks: List[OCRBlock]
    words: List[OCRWord]
    average_confidence: float
    language: str
    engine_name: str = ""  # 使用したエンジン名
    processing_time: float = 0.0  # 処理時間（秒）
    warnings: List[str] = field(default_factory=list)
    
    def get_words_near(self, x: int, y: int, radius: int = 50) -> List[OCRWord]:
        """指定座標付近の単語を取得"""
        result = []
        for word in self.words:
            cx, cy = word.bbox.center
            distance = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if distance <= radius:
                result.append(word)
        return result
    
    def get_words_in_region(self, left: int, top: int, 
                            right: int, bottom: int) -> List[OCRWord]:
        """指定領域内の単語を取得"""
        region = BoundingBox(left, top, right - left, bottom - top)
        return [w for w in self.words if w.bbox.overlaps(region)]
    
    def find_text(self, keyword: str) -> List[OCRWord]:
        """キーワードを含む単語を検索"""
        return [w for w in self.words if keyword in w.text]
    
    def get_text_right_of(self, keyword: str, 
                          max_distance: int = 200) -> Optional[str]:
        """キーワードの右側にあるテキストを取得（Key-Value抽出用）"""
        keyword_words = self.find_text(keyword)
        if not keyword_words:
            return None
        
        keyword_word = keyword_words[0]
        candidates = []
        
        for word in self.words:
            if (word.bbox.left > keyword_word.bbox.right and
                abs(word.bbox.top - keyword_word.bbox.top) < 30):
                distance = word.bbox.left - keyword_word.bbox.right
                if distance <= max_distance:
                    candidates.append((distance, word))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            result_words = [candidates[0][1]]
            for i in range(1, len(candidates)):
                if candidates[i][0] - candidates[i-1][0] < 50:
                    result_words.append(candidates[i][1])
            return " ".join(w.text for w in result_words)
        
        return None
    
    def get_text_below(self, keyword: str, 
                       max_distance: int = 100) -> Optional[str]:
        """キーワードの下にあるテキストを取得"""
        keyword_words = self.find_text(keyword)
        if not keyword_words:
            return None
        
        keyword_word = keyword_words[0]
        candidates = []
        
        for word in self.words:
            if (word.bbox.top > keyword_word.bbox.bottom and
                abs(word.bbox.left - keyword_word.bbox.left) < 50):
                distance = word.bbox.top - keyword_word.bbox.bottom
                if distance <= max_distance:
                    candidates.append((distance, word))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1].text
        
        return None


class BaseOCREngine(ABC):
    """OCRエンジン抽象基底クラス"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """エンジン名を返す"""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """エンジンが利用可能かどうか"""
        pass
    
    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        画像からテキストを認識
        
        Args:
            image: 入力画像（BGR or グレースケール）
            
        Returns:
            OCRResult: 認識結果
        """
        pass
    
    def _merge_bboxes(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """複数のbounding boxを統合"""
        if not bboxes:
            return BoundingBox(0, 0, 0, 0)
        
        left = min(b.left for b in bboxes)
        top = min(b.top for b in bboxes)
        right = max(b.right for b in bboxes)
        bottom = max(b.bottom for b in bboxes)
        
        return BoundingBox(left, top, right - left, bottom - top)
    
    def _build_blocks_from_words(self, words: List[OCRWord]) -> List[OCRBlock]:
        """単語リストからブロック・行構造を構築"""
        if not words:
            return []
        
        # ブロック番号でグループ化
        block_map: Dict[int, List[OCRWord]] = {}
        for word in words:
            if word.block_num not in block_map:
                block_map[word.block_num] = []
            block_map[word.block_num].append(word)
        
        blocks = []
        for block_num, block_words in sorted(block_map.items()):
            # 行番号でグループ化
            line_map: Dict[int, List[OCRWord]] = {}
            for word in block_words:
                if word.line_num not in line_map:
                    line_map[word.line_num] = []
                line_map[word.line_num].append(word)
            
            lines = []
            for line_num, line_words in sorted(line_map.items()):
                line_words.sort(key=lambda w: w.bbox.left)
                
                line_bbox = self._merge_bboxes([w.bbox for w in line_words])
                valid_confs = [w.confidence for w in line_words if w.confidence > 0]
                line_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
                line_text = " ".join(w.text for w in line_words)
                
                lines.append(OCRLine(
                    text=line_text,
                    words=line_words,
                    confidence=line_conf,
                    bbox=line_bbox,
                    block_num=block_num,
                    line_num=line_num
                ))
            
            block_bbox = self._merge_bboxes([l.bbox for l in lines])
            valid_confs = [l.confidence for l in lines if l.confidence > 0]
            block_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
            block_text = "\n".join(l.text for l in lines)
            
            blocks.append(OCRBlock(
                text=block_text,
                lines=lines,
                confidence=block_conf,
                bbox=block_bbox,
                block_num=block_num
            ))
        
        return blocks


# 公開API
__all__ = [
    'BoundingBox',
    'OCRWord',
    'OCRLine',
    'OCRBlock',
    'OCRResult',
    'BaseOCREngine',
]
