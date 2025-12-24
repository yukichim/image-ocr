"""
画像前処理モジュール

高精度OCRを実現するための画像前処理機能を提供:
- 解像度正規化（300 DPI相当）
- 適応的二値化（Otsu法 + 適応的閾値）
- 傾き補正（Hough変換ベース）
- ノイズ除去（モルフォロジー処理）
- Exif orientation対応
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from PIL import Image as PILImage
from PIL import ImageOps


@dataclass
class PreprocessConfig:
    """前処理設定"""
    target_dpi: int = 300
    max_dimension: int = 1500  # 最大辺のピクセル数（速度向上のため）
    min_dimension: int = 800   # 最小辺のピクセル数（精度確保のため）
    denoise_strength: int = 5  # ノイズ除去強度（下げて高速化）
    morph_kernel_size: int = 1
    adaptive_block_size: int = 15  # ブロックサイズを大きく（写真向け）
    adaptive_c: int = 5  # Cを大きく（写真向け）
    deskew_enabled: bool = False  # 傾き補正を無効化（ユーザー手動回転を推奨）
    perspective_correction_enabled: bool = False  # 透視変換補正（デフォルト無効）
    roi_detection_enabled: bool = False  # ROI検出（デフォルト無効、境界が明確な場合のみ推奨）
    contrast_enabled: bool = True
    shadow_removal_enabled: bool = True
    unsharp_mask_enabled: bool = True
    auto_orientation: bool = True  # Exif orientationの自動補正
    binarize_enabled: bool = False  # 二値化無効化（写真画像向け）
    photo_mode: bool = True  # 写真モード（軽量前処理）
    # ROI検出パラメータ
    roi_min_area_ratio: float = 0.15  # ROI最小面積比（画像全体に対する割合）
    roi_max_area_ratio: float = 0.85  # ROI最大面積比（85%以上は画像全体とみなす）
    roi_aspect_ratio_min: float = 0.3  # 最小アスペクト比
    roi_aspect_ratio_max: float = 3.5  # 最大アスペクト比
    roi_padding_ratio: float = 0.02  # ROIクロップ時のパディング率（2%）
    # Phase 1: 照明・影対策
    lab_clahe_enabled: bool = True  # Lab色空間でのCLAHE適用
    clahe_clip_limit: float = 2.0  # CLAHE clipLimit
    clahe_grid_size: int = 8  # CLAHE tileGridSize
    sauvola_enabled: bool = True  # Sauvola二値化（適応的二値化の強化版）
    sauvola_window_size: int = 25  # Sauvola窓サイズ（奇数）
    sauvola_k: float = 0.2  # Sauvolaパラメータk
    # Phase 2: テキスト検出強化
    multi_scale_enabled: bool = False  # マルチスケール推論（大規模画像向け、デフォルト無効）
    multi_scale_factors: tuple = (0.75, 1.0, 1.25)  # スケール係数（軽量化）
    text_dilation_enabled: bool = True  # テキスト領域の膨張
    text_dilation_kernel: tuple = (7, 2)  # 膨張カーネル（水平, 垂直）


@dataclass
class PreprocessResult:
    """前処理結果"""
    image: np.ndarray
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    deskew_angle: float
    perspective_corrected: bool  # 透視変換補正が適用されたか
    roi_detected: bool  # ROI検出が適用されたか
    roi_corners: Optional[np.ndarray]  # 検出されたROIの4隅座標
    preprocessing_applied: list
    # マルチスケール結果（Phase 2用）
    multi_scale_images: Optional[list] = None  # 各スケールの画像リスト


class ImagePreprocessor:
    """画像前処理クラス"""
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        画像ファイルを読み込む（Exif orientation対応）
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            BGR形式のnumpy配列（正しい向きに補正済み）
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: 画像の読み込みに失敗した場合
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        
        # PILで読み込み（Exif orientation自動補正）
        if self.config.auto_orientation:
            try:
                pil_image = PILImage.open(str(path))
                
                # Exif orientationを自動補正（ImageOps.exif_transposeを使用）
                try:
                    pil_image = ImageOps.exif_transpose(pil_image)
                except Exception:
                    pass  # Exif情報がない場合は何もしない
                
                # RGB -> BGR変換
                if pil_image.mode == 'RGB':
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                elif pil_image.mode == 'RGBA':
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
                elif pil_image.mode == 'L':
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_GRAY2BGR)
                else:
                    pil_image = pil_image.convert('RGB')
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                return image
                
            except Exception:
                pass  # PILでの読み込みに失敗した場合はOpenCVを使用
        
        # OpenCVで読み込み（フォールバック）
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
        
        return image
    
    def process(self, image: np.ndarray) -> PreprocessResult:
        """
        画像に対して全ての前処理を適用
        
        Phase 1: 照明・影対策（Lab CLAHE、Sauvola二値化）
        Phase 2: テキスト検出強化（マルチスケール、膨張処理）
        Phase 3: 境界処理（パディング、傾き補正）
        
        Args:
            image: 入力画像（BGR形式）
            
        Returns:
            PreprocessResult: 前処理結果
        """
        original_size = (image.shape[1], image.shape[0])
        preprocessing_applied = []
        deskew_angle = 0.0
        roi_detected = False
        roi_corners = None
        multi_scale_images = None
        
        # カラー画像を保持
        if len(image.shape) == 3:
            color_image = image.copy()
        else:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # === Phase 1: 照明・影対策 ===
        
        # 1a. Lab色空間でのCLAHE適用（影領域の局所コントラスト改善）
        # Zuiderveld, K. (1994): Lab色空間のL（輝度）チャンネルに適用することで
        # 色情報を保持しつつ、影領域のコントラストを改善
        if self.config.lab_clahe_enabled and len(color_image.shape) == 3:
            color_image = self._apply_lab_clahe(color_image)
            preprocessing_applied.append("lab_clahe")
        
        # 1b. グレースケール変換
        if len(color_image.shape) == 3:
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            preprocessing_applied.append("grayscale")
        else:
            gray = color_image.copy()
        
        # 2. ROI検出と透視変換補正（パディング付き）
        perspective_corrected = False
        if self.config.roi_detection_enabled or self.config.perspective_correction_enabled:
            corrected, was_corrected, corners = self._detect_roi_and_correct(color_image)
            if was_corrected:
                gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
                color_image = corrected
                if self.config.roi_detection_enabled:
                    preprocessing_applied.append("roi_detection")
                    roi_detected = True
                    roi_corners = corners
                if self.config.perspective_correction_enabled:
                    preprocessing_applied.append("perspective_correction")
                    perspective_corrected = True
        
        # 3. 解像度正規化
        gray = self._normalize_resolution(gray)
        color_image = self._normalize_resolution(color_image)
        preprocessing_applied.append("resolution_normalize")
        
        # === Phase 2: マルチスケール処理（小文字検出改善） ===
        # 高解像度画像で小さい文字が検出漏れする問題に対応
        # 複数スケールで処理し、後でNMSで統合
        if self.config.multi_scale_enabled:
            multi_scale_images = self._create_multi_scale_images(gray)
            preprocessing_applied.append(f"multi_scale({len(multi_scale_images)} scales)")
        
        # 写真モード: 軽量な前処理のみ
        if self.config.photo_mode:
            # 影除去（強化版）
            if self.config.shadow_removal_enabled:
                gray = self._remove_shadow_advanced(gray)
                preprocessing_applied.append("shadow_removal_advanced")
            
            # コントラスト強調
            if self.config.contrast_enabled:
                gray = self._enhance_contrast(gray)
                preprocessing_applied.append("contrast_enhance")
            
            processed_size = (gray.shape[1], gray.shape[0])
            
            return PreprocessResult(
                image=gray,
                original_size=original_size,
                processed_size=processed_size,
                deskew_angle=0.0,
                perspective_corrected=perspective_corrected,
                roi_detected=roi_detected,
                roi_corners=roi_corners,
                preprocessing_applied=preprocessing_applied,
                multi_scale_images=multi_scale_images
            )
        
        # 通常モード: フル前処理
        # 4. 影除去（照明ムラ補正）
        if self.config.shadow_removal_enabled:
            gray = self._remove_shadow(gray)
            preprocessing_applied.append("shadow_removal")
        
        # 5. コントラスト強調
        if self.config.contrast_enabled:
            gray = self._enhance_contrast(gray)
            preprocessing_applied.append("contrast_enhance")
        
        # 6. ノイズ除去
        gray = self._denoise(gray)
        preprocessing_applied.append("denoise")
        
        # 7. 傾き補正
        if self.config.deskew_enabled:
            gray, deskew_angle = self._deskew(gray)
            if abs(deskew_angle) > 0.1:
                preprocessing_applied.append(f"deskew({deskew_angle:.2f}deg)")
        
        # 8. アンシャープマスク（エッジ強調）
        if self.config.unsharp_mask_enabled:
            gray = self._unsharp_mask(gray)
            preprocessing_applied.append("unsharp_mask")
        
        # 9. 適応的二値化（オプション）- Sauvola法に強化
        if self.config.binarize_enabled:
            if self.config.sauvola_enabled:
                # Sauvola二値化: 照明ムラに強い適応的二値化
                # Sauvola & Pietikainen (2000): 局所統計量（平均・標準偏差）で閾値決定
                output = self._sauvola_binarize(gray)
                preprocessing_applied.append("sauvola_binarize")
            else:
                output = self._adaptive_binarize(gray)
                preprocessing_applied.append("adaptive_binarize")
            
            # 10. テキスト領域の膨張（文字ブロック統合）
            if self.config.text_dilation_enabled:
                output = self._dilate_text_regions(output)
                preprocessing_applied.append("text_dilation")
            
            # 11. モルフォロジー処理（細かいノイズ除去）
            output = self._morphology_clean(output)
            preprocessing_applied.append("morphology_clean")
        else:
            output = gray
        
        processed_size = (output.shape[1], output.shape[0])
        
        return PreprocessResult(
            image=output,
            original_size=original_size,
            processed_size=processed_size,
            deskew_angle=deskew_angle,
            perspective_corrected=perspective_corrected,
            roi_detected=roi_detected,
            roi_corners=roi_corners,
            preprocessing_applied=preprocessing_applied,
            multi_scale_images=multi_scale_images
        )
    
    def _normalize_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        解像度を正規化（OCRに適したサイズに調整）
        
        - 大きすぎる画像: max_dimension以下にダウンスケール（高速化）
        - 小さすぎる画像: min_dimension以上にアップスケール（精度確保）
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        min_dim = min(h, w)
        
        # 大きすぎる場合はダウンスケール
        if max_dim > self.config.max_dimension:
            scale = self.config.max_dimension / max_dim
            new_width = int(w * scale)
            new_height = int(h * scale)
            image = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)  # ダウンスケールにはINTER_AREA
        # 小さすぎる場合はアップスケール
        elif min_dim < self.config.min_dimension:
            scale = self.config.min_dimension / min_dim
            new_width = int(w * scale)
            new_height = int(h * scale)
            image = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def _detect_roi_and_correct(self, image: np.ndarray) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
        """
        ROI検出と透視変換補正（Region of Interest Detection + Perspective Correction）
        
        文書（レシート/請求書）の境界を検出し、背景をクロッピングして正面視点に変換する。
        これにより以下の効果が得られる：
        - ノイズ除去：手、テーブル、影などの背景要素を排除
        - 最適な平坦化：正確な4点座標による透視変換
        - セグメンテーション精度向上：テキスト領域のみにフォーカス
        
        Args:
            image: 入力画像（BGR形式）
            
        Returns:
            (補正後画像, 補正が適用されたか, 検出された4隅座標)
        """
        try:
            h, w = image.shape[:2]
            img_area = h * w
            
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 複数のエッジ検出手法を試行（精度の高い順）
            doc_contour = None
            ordered_pts = None
            
            # 手法1: 色ベースのセグメンテーション（白/明るい領域の検出）
            # レシートや請求書は通常白い紙なので、まず色で検出
            if len(image.shape) == 3:
                doc_contour, ordered_pts = self._find_document_contour_color(image, img_area)
            
            # 手法2: Cannyエッジ検出（複数閾値）
            if doc_contour is None:
                doc_contour, ordered_pts = self._find_document_contour_canny(gray, img_area)
            
            # 手法3: 適応的閾値を使用
            if doc_contour is None:
                doc_contour, ordered_pts = self._find_document_contour_adaptive(gray, img_area)
            
            # 手法4: 勾配ベースの検出（Sobel）
            if doc_contour is None:
                doc_contour, ordered_pts = self._find_document_contour_gradient(gray, img_area)
            
            if doc_contour is None or ordered_pts is None:
                return image, False, None
            
            # Phase 3: コンテキストパディングの追加
            # 検出領域を少し広げることで、境界付近の文字の欠損を防止
            # "E"が"F"と誤認識される等の問題を回避
            if self.config.roi_padding_ratio > 0:
                ordered_pts = self._add_padding_to_corners(
                    ordered_pts, h, w, self.config.roi_padding_ratio
                )
            
            # 変換後のサイズを計算
            (tl, tr, br, bl) = ordered_pts
            
            # 幅の計算（上辺と下辺の長い方）
            width_top = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
            width_bottom = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
            max_width = int(max(width_top, width_bottom))
            
            # 高さの計算（左辺と右辺の長い方）
            height_left = np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
            height_right = np.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
            max_height = int(max(height_left, height_right))
            
            # 最小サイズチェック
            if max_width < 100 or max_height < 100:
                return image, False, None
            
            # 極端なアスペクト比は拒否
            aspect_ratio = max_width / max_height if max_height > 0 else 0
            if aspect_ratio < self.config.roi_aspect_ratio_min or aspect_ratio > self.config.roi_aspect_ratio_max:
                return image, False, None
            
            # 変換先の座標
            dst_pts = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype=np.float32)
            
            # ホモグラフィ行列を計算
            matrix = cv2.getPerspectiveTransform(ordered_pts.astype(np.float32), dst_pts)
            
            # 透視変換を適用（背景をクロッピング）
            warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
            
            return warped, True, ordered_pts
            
        except Exception:
            return image, False, None
    
    def _add_padding_to_corners(self, corners: np.ndarray, img_h: int, img_w: int, 
                                 padding_ratio: float) -> np.ndarray:
        """
        検出された4隅座標にパディングを追加
        
        Phase 3.1: コンテキストパディング
        検出したROIを少し広げることで、境界付近の文字欠損を防止
        
        Args:
            corners: 4隅座標 [tl, tr, br, bl]
            img_h: 画像の高さ
            img_w: 画像の幅
            padding_ratio: パディング率（0.02 = 2%）
            
        Returns:
            パディング追加後の4隅座標
        """
        # 各辺の長さを計算
        (tl, tr, br, bl) = corners
        width = max(
            np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2),
            np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
        )
        height = max(
            np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2),
            np.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
        )
        
        # パディング量を計算
        pad_w = width * padding_ratio
        pad_h = height * padding_ratio
        
        # 各隅を外側に拡張
        new_corners = np.array([
            [max(0, tl[0] - pad_w), max(0, tl[1] - pad_h)],  # 左上
            [min(img_w - 1, tr[0] + pad_w), max(0, tr[1] - pad_h)],  # 右上
            [min(img_w - 1, br[0] + pad_w), min(img_h - 1, br[1] + pad_h)],  # 右下
            [max(0, bl[0] - pad_w), min(img_h - 1, bl[1] + pad_h)]  # 左下
        ], dtype=np.float32)
        
        return new_corners
    
    def _find_document_contour_canny(self, gray: np.ndarray, img_area: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Cannyエッジ検出を使用した文書輪郭の検出
        複数の閾値設定を試行し、最適な結果を選択
        """
        # 前処理：ぼかしでノイズ除去
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 複数の閾値設定を試行
        threshold_pairs = [
            (30, 100),   # 低感度（明確なエッジのみ）
            (50, 150),   # 中感度（デフォルト）
            (75, 200),   # 高感度
            (20, 80),    # 非常に低感度（強いエッジのみ）
        ]
        
        best_result = None
        best_area = 0
        
        for low, high in threshold_pairs:
            edges = cv2.Canny(blurred, low, high)
            
            # 膨張処理でエッジを強調・接続
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
            
            contour, pts = self._find_quadrilateral(edges, img_area)
            if contour is not None:
                area = cv2.contourArea(contour)
                if area > best_area:
                    best_area = area
                    best_result = (contour, pts)
        
        if best_result:
            return best_result
        return None, None
    
    def _find_document_contour_adaptive(self, gray: np.ndarray, img_area: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        適応的閾値を使用した文書輪郭の検出
        """
        # ぼかし
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 適応的閾値
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )
        
        # モルフォロジー処理
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return self._find_quadrilateral(thresh, img_area)
    
    def _find_document_contour_gradient(self, gray: np.ndarray, img_area: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        勾配ベースの文書輪郭検出（Sobelフィルタ使用）
        エッジの方向情報を活用して直線的な境界を検出
        """
        # ぼかし
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Sobelフィルタで勾配を計算
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # 勾配の大きさ
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
        
        # 閾値処理
        _, thresh = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)
        
        # モルフォロジー処理でエッジを接続
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return self._find_quadrilateral(thresh, img_area)

    def _find_document_contour_color(self, image: np.ndarray, img_area: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        色ベースのセグメンテーションを使用した文書輪郭の検出
        紙の領域（レシート/請求書）を検出 - 白、クリーム、明るいグレーに対応
        """
        # HSVに変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 複数の色範囲でマスクを作成
        masks = []
        
        # 白い領域のマスク（彩度が低く、明度が高い）
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        masks.append(mask_white)
        
        # クリーム/ベージュ色（レシート用紙によく使われる）
        lower_cream = np.array([10, 10, 180])
        upper_cream = np.array([30, 80, 255])
        mask_cream = cv2.inRange(hsv, lower_cream, upper_cream)
        masks.append(mask_cream)
        
        # 明るいグレー（感熱紙など）
        lower_gray = np.array([0, 0, 140])
        upper_gray = np.array([180, 30, 210])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        masks.append(mask_gray)
        
        # マスクを統合
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # モルフォロジー処理（ノイズ除去と穴埋め）
        kernel = np.ones((9, 9), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return self._find_quadrilateral(combined_mask, img_area)
    
    def _find_quadrilateral(self, binary: np.ndarray, img_area: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        二値画像から四角形の輪郭を検出
        複数の近似精度を試行し、最適な四角形を選択
        """
        # 輪郭検出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # 面積でソート
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        min_area = img_area * self.config.roi_min_area_ratio
        max_area = img_area * self.config.roi_max_area_ratio
        
        # 複数の近似精度を試行
        epsilon_factors = [0.015, 0.02, 0.025, 0.03, 0.04]
        
        for contour in contours[:10]:  # 上位10個の輪郭を確認
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            # 複数の近似精度を試行
            peri = cv2.arcLength(contour, True)
            for eps_factor in epsilon_factors:
                approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                
                # 4角形ならば文書として認識
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    ordered_pts = self._order_points(pts)
                    if ordered_pts is not None:
                        # 角度チェック：各角が60-120度の範囲にあることを確認
                        if self._check_quadrilateral_angles(ordered_pts):
                            return approx, ordered_pts
            
            # 4角形が見つからない場合、凸包から矩形を近似
            hull = cv2.convexHull(contour)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 回転矩形の面積チェック
            rect_area = rect[1][0] * rect[1][1]
            if min_area < rect_area < max_area:
                ordered_pts = self._order_points(box.astype(np.float32))
                if ordered_pts is not None:
                    return box, ordered_pts
        
        return None, None
    
    def _check_quadrilateral_angles(self, pts: np.ndarray) -> bool:
        """
        四角形の各角度が妥当な範囲（60-120度）にあるかチェック
        これにより、極端に歪んだ形状を排除
        """
        try:
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i + 1) % 4]
                p3 = pts[(i + 2) % 4]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                if angle < 45 or angle > 135:
                    return False
            return True
        except Exception:
            return True  # エラー時は通過させる
    
    def _correct_perspective(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        透視変換補正（後方互換性のためのラッパー）
        """
        result, corrected, _ = self._detect_roi_and_correct(image)
        return result, corrected
    
    def _order_points(self, pts: np.ndarray) -> Optional[np.ndarray]:
        """
        4点を左上、右上、右下、左下の順に並べ替え
        
        Args:
            pts: 4点の座標配列
            
        Returns:
            順序付けられた4点（失敗時はNone）
        """
        try:
            # 合計値（x+y）でソート：左上が最小、右下が最大
            s = pts.sum(axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            
            # 差分（x-y）でソート：右上が最大、左下が最小
            d = np.diff(pts, axis=1)
            tr = pts[np.argmax(d)]
            bl = pts[np.argmin(d)]
            
            return np.array([tl, tr, br, bl], dtype=np.float32)
        except Exception:
            return None
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        コントラスト強調（CLAHE: 適応的ヒストグラム均等化）
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit, 
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
        )
        return clahe.apply(image)
    
    # ========== Phase 1: 照明・影対策メソッド ==========
    
    def _apply_lab_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Lab色空間でのCLAHE適用
        
        Zuiderveld, K. (1994) "Contrast Limited Adaptive Histogram Equalization"
        
        Lab色空間のL（輝度）チャンネルにCLAHEを適用することで:
        - 色情報（a, b）を保持しつつ輝度のみを補正
        - 影領域の局所コントラストを改善
        - 標準的なヒストグラム均等化によるノイズ増幅を防止
        
        Args:
            image: BGR形式のカラー画像
            
        Returns:
            輝度補正されたBGR画像
        """
        # BGR -> Lab変換
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Lチャンネル（輝度）を分離
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # LチャンネルにCLAHEを適用
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
        )
        l_enhanced = clahe.apply(l_channel)
        
        # チャンネルを再統合
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Lab -> BGR変換
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def _remove_shadow_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        強化版影除去（照明ムラ補正）
        
        複数の手法を組み合わせて影を効果的に除去:
        1. 大域的背景推定による正規化
        2. 局所コントラスト改善
        
        ただし、過度な補正を避けるためクリッピングを慎重に行う
        
        Args:
            image: グレースケール画像
            
        Returns:
            影除去された画像
        """
        # 入力画像が既に高コントラストの場合はスキップ
        img_std = np.std(image)
        if img_std < 10:  # 既にフラットな画像
            return image
        
        # 手法1: ガウシアンブラーによる背景推定と除算
        kernel_size = max(image.shape[0], image.shape[1]) // 8
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel_size = max(kernel_size, 51)
        kernel_size = min(kernel_size, 151)  # 最大サイズ制限
        
        # 背景を推定
        background = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # ゼロ除算を防ぐ
        background = np.maximum(background, 1)
        
        # 背景で正規化（除算）
        # 明るい背景（レシートなど）では、暗い部分を強調
        normalized = image.astype(np.float32) / background.astype(np.float32)
        
        # スケーリング: 中央値を基準にコントラストを調整
        median_val = np.median(normalized)
        if median_val > 0:
            normalized = normalized / median_val * 128.0
        
        # クリッピング
        result = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return result
    
    def _sauvola_binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Sauvola二値化アルゴリズム
        
        Sauvola, J., & Pietikainen, M. (2000). "Adaptive document image binarization"
        Pattern Recognition, 33(2), 225-236.
        
        局所的な平均と標準偏差を使用して閾値を計算:
        T(x,y) = mean(x,y) * (1 + k * (std(x,y) / R - 1))
        
        ここで:
        - k: パラメータ（通常0.2〜0.5）
        - R: 標準偏差の最大値（通常128）
        
        この方法は不均一な照明に強く、影領域でも適切に二値化できる
        
        Args:
            image: グレースケール画像
            
        Returns:
            二値化された画像
        """
        window_size = self.config.sauvola_window_size
        k = self.config.sauvola_k
        R = 128  # 標準偏差の動的レンジ
        
        # 窓サイズは奇数である必要がある
        if window_size % 2 == 0:
            window_size += 1
        
        # 局所平均と局所分散を計算
        # 効率的な計算のためにintegral imageを使用
        mean = cv2.blur(image.astype(np.float32), (window_size, window_size))
        
        # 局所分散の計算: E[X^2] - E[X]^2
        sq_mean = cv2.blur((image.astype(np.float32) ** 2), (window_size, window_size))
        variance = sq_mean - mean ** 2
        variance = np.maximum(variance, 0)  # 数値誤差による負の値を防止
        std = np.sqrt(variance)
        
        # Sauvolaの閾値計算
        threshold = mean * (1 + k * (std / R - 1))
        
        # 二値化
        binary = np.zeros_like(image)
        binary[image > threshold] = 255
        
        return binary
    
    # ========== Phase 2: テキスト検出強化メソッド ==========
    
    def _create_multi_scale_images(self, image: np.ndarray) -> list:
        """
        マルチスケール画像の生成
        
        小さい文字や大きい文字の両方を検出するため、
        複数のスケールで画像を生成する。
        これにより、パースペクティブによる文字サイズの変動に対応し、
        Recall（再現率）を大幅に改善する。
        
        Args:
            image: 入力画像（グレースケール）
            
        Returns:
            各スケールの画像リスト [(scale_factor, scaled_image), ...]
        """
        multi_scale_images = []
        h, w = image.shape[:2]
        
        for scale in self.config.multi_scale_factors:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 最小サイズチェック
            if new_w < 100 or new_h < 100:
                continue
            
            if scale < 1.0:
                # ダウンスケール
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif scale > 1.0:
                # アップスケール
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                scaled = image.copy()
            
            multi_scale_images.append((scale, scaled))
        
        return multi_scale_images
    
    def _dilate_text_regions(self, binary: np.ndarray) -> np.ndarray:
        """
        テキスト領域の膨張（水平方向に強い膨張）
        
        個別に検出された文字を「単語」や「行」のブロックとして統合。
        これにより、ノイズとして誤って破棄されるのを防ぐ。
        
        水平方向の膨張が強いカーネル (e.g., 7x2) を使用することで、
        同じ行の文字を連結しつつ、異なる行は分離を維持。
        
        Args:
            binary: 二値化された画像
            
        Returns:
            膨張処理された画像
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.text_dilation_kernel
        )
        dilated = cv2.dilate(binary, kernel, iterations=1)
        return dilated

    def _remove_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        影除去（照明ムラ補正）
        
        大きなカーネルでぼかした画像を背景として推定し、
        元画像から差し引くことで照明ムラを除去
        """
        # 背景推定（大きなカーネルでぼかす）
        kernel_size = max(image.shape[0], image.shape[1]) // 10
        # カーネルサイズは奇数である必要がある
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel_size = max(kernel_size, 51)  # 最小51
        
        # ガウシアンブラーで背景を推定
        background = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # 背景を差し引いて正規化
        # 元画像 / 背景 * 255 で照明ムラを補正
        # ゼロ除算を防ぐ
        background = np.maximum(background, 1)
        
        # 浮動小数点で計算
        normalized = image.astype(np.float32) / background.astype(np.float32) * 255.0
        
        # 0-255にクリップしてuint8に変換
        result = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return result
    
    def _unsharp_mask(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        アンシャープマスクによるエッジ強調
        
        ぼかした画像との差分を加えることで、エッジを強調
        """
        # ガウシアンブラーでぼかす
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # アンシャープマスク: 元画像 + (元画像 - ぼかし画像) * 強度
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
        
        return sharpened
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        ノイズ除去（Non-local Means Denoising）
        """
        return cv2.fastNlMeansDenoising(
            image, 
            None, 
            h=self.config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        傾き補正（Hough変換ベース）
        
        Returns:
            補正後画像と検出された傾き角度（度）
        """
        # エッジ検出
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough変換で直線検出
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return image, 0.0
        
        # 各直線の角度を計算
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # 水平に近い線のみ考慮（-45° から 45°）
                if -45 < angle < 45:
                    angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # 中央値で傾き角度を決定（外れ値に強い）
        median_angle = np.median(angles)
        
        # 小さすぎる傾きは無視
        if abs(median_angle) < 0.5:
            return image, 0.0
        
        # 画像を回転
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated, median_angle
    
    def _adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """
        適応的二値化
        
        背景の明るさムラに対応するため、局所的な閾値を使用
        """
        # まずOtsu法でグローバル閾値を試行
        _, otsu = cv2.threshold(
            image, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # 適応的閾値（局所的な明るさ変化に対応）
        adaptive = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
        
        # 両方の結果を組み合わせ（論理AND）
        # これにより、両方で白と判定された部分のみ白になる
        combined = cv2.bitwise_and(otsu, adaptive)
        
        return combined
    
    def _morphology_clean(self, image: np.ndarray) -> np.ndarray:
        """
        モルフォロジー処理でノイズ除去と文字の補完
        """
        kernel_size = self.config.morph_kernel_size
        if kernel_size < 1:
            return image
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (kernel_size, kernel_size)
        )
        
        # オープニング（小さなノイズ除去）
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # クロージング（文字の切れ目を補完）
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def process_file(self, image_path: str) -> PreprocessResult:
        """
        ファイルパスから画像を読み込んで前処理を実行
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            PreprocessResult: 前処理結果
        """
        image = self.load_image(image_path)
        return self.process(image)


def preprocess_image(image_path: str, config: Optional[PreprocessConfig] = None) -> PreprocessResult:
    """
    画像前処理のショートカット関数
    
    Args:
        image_path: 画像ファイルパス
        config: 前処理設定（省略時はデフォルト）
        
    Returns:
        PreprocessResult: 前処理結果
    """
    preprocessor = ImagePreprocessor(config)
    return preprocessor.process_file(image_path)
