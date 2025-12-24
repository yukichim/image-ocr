#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Experiment GUI Tool

TkInter-based GUI application for OCR testing:
- Left: Input image display (file selection)
- Right: OCR result with bounding boxes
- Test image generation with multiple patterns
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
from pathlib import Path
import json
import threading
import tempfile
import math
import random
import cv2
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

from .pipeline import OCRPipeline, PipelineConfig, PipelineResult
from .classifier import DocumentType
from .ocr_engine import OCRResult, BoundingBox
from .engines import EngineType, get_available_engines


@dataclass
class GUIConfig:
    """GUI configuration"""
    window_width: int = 1400
    window_height: int = 1200
    image_max_width: int = 600
    image_max_height: int = 500
    bbox_color_high: str = "#00FF00"    # Green: high confidence
    bbox_color_mid: str = "#FFFF00"     # Yellow: medium confidence
    bbox_color_low: str = "#FF0000"     # Red: low confidence
    bbox_width: int = 2
    confidence_high_threshold: float = 80.0
    confidence_mid_threshold: float = 60.0


class TestImageGenerator:
    """Test image generator with realistic patterns and font variations"""
    
    # Japanese font paths by category
    FONT_CATEGORIES = {
        "gothic": [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "C:\\Windows\\Fonts\\msgothic.ttc",
        ],
        "mincho": [
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "C:\\Windows\\Fonts\\msmincho.ttc",
        ],
        "mono": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "C:\\Windows\\Fonts\\consola.ttf",
        ],
    }
    
    # Background patterns
    BACKGROUND_STYLES = {
        "white": "Plain white paper",
        "thermal": "Thermal paper (receipt)",
        "aged": "Aged/yellowed paper",
        "lined": "Lined paper",
        "grid": "Grid paper",
        "textured": "Textured paper",
    }
    
    # Test pattern definitions (expanded)
    PATTERNS = {
        # Basic receipt patterns (English)
        "receipt_normal": "Receipt - Normal (Gothic)",
        "receipt_mincho": "Receipt - Mincho font",
        "receipt_thermal": "Receipt - Thermal paper",
        "receipt_aged": "Receipt - Aged paper",
        "receipt_tilt_5": "Receipt - Tilted 5 deg",
        "receipt_tilt_-7": "Receipt - Tilted -7 deg",
        "receipt_distorted": "Receipt - Distorted text",
        "receipt_noisy": "Receipt - Noisy",
        "receipt_low_contrast": "Receipt - Low contrast",
        "receipt_shadow": "Receipt - With shadow",
        "receipt_crumpled": "Receipt - Crumpled/Wrinkled",
        "receipt_faded": "Receipt - Faded print",
        "receipt_partial_shadow": "Receipt - Partial shadow",
        # Japanese receipt patterns
        "receipt_jp_normal": "Receipt - Japanese normal",
        "receipt_jp_thermal": "Receipt - Japanese thermal",
        "receipt_jp_convenience": "Receipt - Japanese convenience store",
        # Invoice patterns
        "invoice_normal": "Invoice - Normal",
        "invoice_mincho": "Invoice - Mincho font",
        "invoice_textured": "Invoice - Textured paper",
        "invoice_tilt_-3": "Invoice - Tilted -3 deg",
        "invoice_shadow": "Invoice - With shadow",
        "invoice_stamp": "Invoice - With stamp overlay",
        # Japanese invoice patterns
        "invoice_jp_normal": "Invoice - Japanese normal",
        "invoice_jp_hotel": "Invoice - Japanese hotel",
    }
    
    def __init__(self, font_style: str = "gothic", bg_style: str = "white"):
        self.font_style = font_style
        self.bg_style = bg_style
        self._load_fonts()
    
    def _load_fonts(self):
        """Load fonts for current style"""
        self.font = self._load_font_by_style(self.font_style, 28)
        self.font_large = self._load_font_by_style(self.font_style, 40)
        self.font_small = self._load_font_by_style(self.font_style, 20)
        self.font_stamp = self._load_font_by_style("gothic", 60)
    
    def _load_font_by_style(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        """Load font by style category with fallback"""
        paths = self.FONT_CATEGORIES.get(style, self.FONT_CATEGORIES["gothic"])
        for path in paths:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
        # Fallback to any available font
        for category in self.FONT_CATEGORIES.values():
            for path in category:
                try:
                    return ImageFont.truetype(path, size)
                except (IOError, OSError):
                    continue
        # Final fallback
        try:
            return ImageFont.load_default()
        except:
            return None
    
    def _create_background(self, width: int, height: int, style: str = None) -> Image.Image:
        """Create background with specified style"""
        style = style or self.bg_style
        
        if style == "white":
            return Image.new('RGB', (width, height), color='white')
        
        elif style == "thermal":
            # Thermal paper - slight off-white with subtle texture
            img = Image.new('RGB', (width, height), color=(252, 250, 245))
            img_array = np.array(img)
            # Add very subtle noise
            noise = np.random.normal(0, 2, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Add subtle horizontal lines (thermal print artifact)
            for y in range(0, height, 3):
                if random.random() < 0.3:
                    img_array[y, :, :] = np.clip(img_array[y, :, :].astype(np.int16) - 5, 0, 255)
            return Image.fromarray(img_array)
        
        elif style == "aged":
            # Aged/yellowed paper
            base_color = (250, 242, 220)
            img = Image.new('RGB', (width, height), color=base_color)
            img_array = np.array(img).astype(np.float32)
            # Add age spots
            for _ in range(random.randint(5, 15)):
                cx, cy = random.randint(0, width), random.randint(0, height)
                radius = random.randint(20, 80)
                for y in range(max(0, cy - radius), min(height, cy + radius)):
                    for x in range(max(0, cx - radius), min(width, cx + radius)):
                        dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                        if dist < radius:
                            factor = 0.9 + 0.1 * (dist / radius)
                            img_array[y, x] = img_array[y, x] * factor
            # Add edge darkening
            for y in range(height):
                for x in range(width):
                    edge_dist = min(x, width - x, y, height - y) / 50
                    edge_dist = min(edge_dist, 1.0)
                    img_array[y, x] = img_array[y, x] * (0.85 + 0.15 * edge_dist)
            return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        
        elif style == "lined":
            # Lined paper
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            line_spacing = 30
            for y in range(line_spacing, height, line_spacing):
                draw.line([(0, y), (width, y)], fill=(200, 200, 230), width=1)
            return img
        
        elif style == "grid":
            # Grid paper
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            grid_spacing = 25
            for y in range(0, height, grid_spacing):
                draw.line([(0, y), (width, y)], fill=(220, 230, 220), width=1)
            for x in range(0, width, grid_spacing):
                draw.line([(x, 0), (x, height)], fill=(220, 230, 220), width=1)
            return img
        
        elif style == "textured":
            # Textured paper (like bond paper)
            img = Image.new('RGB', (width, height), color=(253, 253, 250))
            img_array = np.array(img)
            # Add fiber-like texture
            noise = np.random.normal(0, 4, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        
        return Image.new('RGB', (width, height), color='white')
    
    def generate_receipt_normal(self, bg_style: str = None) -> Image.Image:
        """Generate normal receipt image"""
        img = self._create_background(600, 900, bg_style)
        draw = ImageDraw.Draw(img)
        
        # Header
        self._draw_text_safe(draw, (220, 30), 'RECEIPT', self.font_large, 'black')
        draw.line((50, 90, 550, 90), fill='black', width=2)
        
        # Store info
        self._draw_text_safe(draw, (50, 110), 'Test Convenience Store', self.font, 'black')
        self._draw_text_safe(draw, (50, 150), 'TEL: 03-1234-5678', self.font, 'black')
        self._draw_text_safe(draw, (50, 190), 'Tokyo, Shibuya-ku', self.font_small, 'black')
        
        # Date/Time
        self._draw_text_safe(draw, (50, 240), '2024/12/08 14:30', self.font, 'black')
        draw.line((50, 280, 550, 280), fill='gray', width=1)
        
        # Items
        items = [
            ('Rice Ball', '150', False),
            ('Green Tea *', '130', True),
            ('Sandwich *', '280', True),
            ('Coffee', '150', False),
            ('Gum', '100', False),
        ]
        
        y = 300
        for name, price, reduced in items:
            self._draw_text_safe(draw, (50, y), name, self.font, 'black')
            self._draw_text_safe(draw, (450, y), f'\\{price}', self.font, 'black')
            y += 45
        
        draw.line((50, y, 550, y), fill='gray', width=1)
        y += 20
        
        # Subtotal
        self._draw_text_safe(draw, (50, y), 'Subtotal', self.font, 'black')
        self._draw_text_safe(draw, (450, y), '\\810', self.font, 'black')
        y += 45
        
        self._draw_text_safe(draw, (50, y), '(8% items)', self.font_small, 'gray')
        self._draw_text_safe(draw, (450, y), '\\410', self.font_small, 'gray')
        y += 35
        
        self._draw_text_safe(draw, (50, y), '(10% items)', self.font_small, 'gray')
        self._draw_text_safe(draw, (450, y), '\\400', self.font_small, 'gray')
        y += 45
        
        draw.line((50, y, 550, y), fill='black', width=2)
        y += 15
        
        # Total
        self._draw_text_safe(draw, (50, y), 'TOTAL', self.font_large, 'black')
        self._draw_text_safe(draw, (400, y), '\\810', self.font_large, 'black')
        y += 60
        
        self._draw_text_safe(draw, (50, y), 'Cash', self.font, 'black')
        self._draw_text_safe(draw, (450, y), '\\1,000', self.font, 'black')
        y += 45
        
        self._draw_text_safe(draw, (50, y), 'Change', self.font, 'black')
        self._draw_text_safe(draw, (450, y), '\\190', self.font, 'black')
        y += 60
        
        self._draw_text_safe(draw, (50, y), '* = Reduced tax rate items', self.font_small, 'gray')
        
        return img
    
    def _draw_text_safe(self, draw, pos, text, font, fill):
        """Safely draw text with font fallback"""
        try:
            if font:
                draw.text(pos, text, fill=fill, font=font)
            else:
                draw.text(pos, text, fill=fill)
        except Exception:
            draw.text(pos, text, fill=fill)
    
    def generate_receipt_jp_normal(self, bg_style: str = None) -> Image.Image:
        """Generate Japanese receipt (normal)"""
        img = self._create_background(600, 900, bg_style or "white")
        draw = ImageDraw.Draw(img)
        
        # Header
        self._draw_text_safe(draw, (200, 30), '領 収 書', self.font_large, 'black')
        draw.line((50, 90, 550, 90), fill='black', width=2)
        
        # Store info
        self._draw_text_safe(draw, (50, 110), 'テストコンビニエンスストア', self.font, 'black')
        self._draw_text_safe(draw, (50, 150), 'TEL: 03-1234-5678', self.font, 'black')
        self._draw_text_safe(draw, (50, 190), '東京都渋谷区○○1-2-3', self.font_small, 'black')
        
        # Date/Time
        self._draw_text_safe(draw, (50, 240), '2024年12月08日 14:30', self.font, 'black')
        draw.line((50, 280, 550, 280), fill='gray', width=1)
        
        # Items
        items = [
            ('おにぎり(鮭)', '¥150', False),
            ('緑茶 *', '¥130', True),
            ('サンドイッチ *', '¥280', True),
            ('コーヒー', '¥150', False),
            ('ガム', '¥100', False),
        ]
        
        y = 300
        for name, price, reduced in items:
            self._draw_text_safe(draw, (50, y), name, self.font, 'black')
            self._draw_text_safe(draw, (450, y), price, self.font, 'black')
            y += 45
        
        draw.line((50, y, 550, y), fill='gray', width=1)
        y += 20
        
        # Subtotal
        self._draw_text_safe(draw, (50, y), '小計', self.font, 'black')
        self._draw_text_safe(draw, (450, y), '¥810', self.font, 'black')
        y += 45
        
        self._draw_text_safe(draw, (50, y), '(税率8%対象)', self.font_small, 'gray')
        self._draw_text_safe(draw, (420, y), '¥410', self.font_small, 'gray')
        y += 35
        
        self._draw_text_safe(draw, (50, y), '(税率10%対象)', self.font_small, 'gray')
        self._draw_text_safe(draw, (420, y), '¥400', self.font_small, 'gray')
        y += 45
        
        draw.line((50, y, 550, y), fill='black', width=2)
        y += 15
        
        # Total
        self._draw_text_safe(draw, (50, y), '合計', self.font_large, 'black')
        self._draw_text_safe(draw, (380, y), '¥810', self.font_large, 'black')
        y += 60
        
        self._draw_text_safe(draw, (50, y), '現金', self.font, 'black')
        self._draw_text_safe(draw, (420, y), '¥1,000', self.font, 'black')
        y += 45
        
        self._draw_text_safe(draw, (50, y), 'お釣り', self.font, 'black')
        self._draw_text_safe(draw, (430, y), '¥190', self.font, 'black')
        y += 60
        
        self._draw_text_safe(draw, (50, y), '* = 軽減税率対象商品', self.font_small, 'gray')
        
        return img
    
    def generate_receipt_jp_convenience(self, bg_style: str = None) -> Image.Image:
        """Generate Japanese convenience store receipt"""
        img = self._create_background(600, 1000, bg_style or "thermal")
        draw = ImageDraw.Draw(img)
        
        # Store name
        self._draw_text_safe(draw, (150, 30), 'セブン-イレブン', self.font_large, 'black')
        self._draw_text_safe(draw, (150, 80), '渋谷駅東口店', self.font, 'black')
        
        self._draw_text_safe(draw, (50, 120), '〒150-0002', self.font_small, 'black')
        self._draw_text_safe(draw, (50, 145), '東京都渋谷区渋谷2-24-1', self.font_small, 'black')
        self._draw_text_safe(draw, (50, 170), 'TEL 03-1234-5678', self.font_small, 'black')
        
        draw.line((50, 200, 550, 200), fill='black', width=1)
        
        # Registration info
        self._draw_text_safe(draw, (50, 210), '登録番号: T1234567890123', self.font_small, 'black')
        self._draw_text_safe(draw, (50, 235), '2024/12/08(日) 14:30', self.font, 'black')
        self._draw_text_safe(draw, (350, 235), 'レジ#01', self.font_small, 'black')
        
        draw.line((50, 270, 550, 270), fill='gray', width=1)
        
        # Items
        items = [
            ('おにぎり鮭', '¥150', ''),
            ('緑茶500ml', '¥130', '*'),
            ('ミックスサンド', '¥280', '*'),
            ('ホットコーヒーR', '¥150', ''),
            ('ボトルガム', '¥100', ''),
        ]
        
        y = 290
        for name, price, mark in items:
            self._draw_text_safe(draw, (50, y), name + mark, self.font, 'black')
            self._draw_text_safe(draw, (430, y), price, self.font, 'black')
            y += 40
        
        draw.line((50, y, 550, y), fill='gray', width=1)
        y += 15
        
        # Tax breakdown
        self._draw_text_safe(draw, (50, y), '(税抜合計)', self.font_small, 'gray')
        self._draw_text_safe(draw, (430, y), '¥737', self.font_small, 'gray')
        y += 30
        
        self._draw_text_safe(draw, (50, y), '(内 8%税対象 ¥410 消費税 ¥32)', self.font_small, 'gray')
        y += 25
        self._draw_text_safe(draw, (50, y), '(内10%税対象 ¥400 消費税 ¥40)', self.font_small, 'gray')
        y += 35
        
        draw.line((50, y, 550, y), fill='black', width=2)
        y += 15
        
        # Total
        self._draw_text_safe(draw, (50, y), '合計', self.font_large, 'black')
        self._draw_text_safe(draw, (350, y), '¥810', self.font_large, 'black')
        y += 55
        
        # Payment
        self._draw_text_safe(draw, (50, y), 'クレジット扱い', self.font, 'black')
        self._draw_text_safe(draw, (380, y), '¥810', self.font, 'black')
        y += 45
        
        draw.line((50, y, 550, y), fill='gray', width=1)
        y += 15
        
        self._draw_text_safe(draw, (50, y), '*印は軽減税率8%対象', self.font_small, 'gray')
        y += 30
        self._draw_text_safe(draw, (50, y), '上記金額を領収いたしました', self.font_small, 'black')
        
        return img
    
    def generate_invoice_jp_normal(self, bg_style: str = None) -> Image.Image:
        """Generate Japanese invoice (請求書)"""
        img = self._create_background(800, 1000, bg_style or "white")
        draw = ImageDraw.Draw(img)
        
        # Header
        self._draw_text_safe(draw, (320, 30), '請 求 書', self.font_large, 'black')
        draw.line((50, 90, 750, 90), fill='black', width=2)
        
        # Recipient
        self._draw_text_safe(draw, (50, 110), 'テスト株式会社 御中', self.font, 'black')
        
        # Issuer
        self._draw_text_safe(draw, (500, 110), '発行者:', self.font_small, 'gray')
        self._draw_text_safe(draw, (500, 140), 'サンプル株式会社', self.font, 'black')
        self._draw_text_safe(draw, (500, 180), '東京都港区○○1-2-3', self.font_small, 'black')
        self._draw_text_safe(draw, (500, 205), 'TEL: 03-1234-5678', self.font_small, 'black')
        
        # Date and Invoice number
        self._draw_text_safe(draw, (50, 220), '請求日: 2024年12月08日', self.font_small, 'black')
        self._draw_text_safe(draw, (50, 250), '請求書番号: INV-2024-001234', self.font, 'black')
        
        draw.line((50, 290, 750, 290), fill='gray', width=1)
        
        # Total box
        draw.rectangle([50, 310, 350, 380], outline='black', width=2)
        self._draw_text_safe(draw, (70, 325), '御請求金額', self.font, 'black')
        self._draw_text_safe(draw, (180, 330), '¥55,000', self.font_large, 'black')
        
        # Table header
        y = 410
        draw.rectangle([50, y, 750, y+40], fill='#f0f0f0', outline='black')
        self._draw_text_safe(draw, (60, y+8), '品目', self.font, 'black')
        self._draw_text_safe(draw, (350, y+8), '数量', self.font, 'black')
        self._draw_text_safe(draw, (450, y+8), '単価', self.font, 'black')
        self._draw_text_safe(draw, (600, y+8), '金額', self.font, 'black')
        
        y += 40
        items = [
            ('コンサルティング費用', '10', '¥5,000', '¥50,000'),
            ('交通費実費', '1', '¥5,000', '¥5,000'),
        ]
        
        for name, qty, unit, amount in items:
            draw.line((50, y, 750, y), fill='gray', width=1)
            self._draw_text_safe(draw, (60, y+8), name, self.font, 'black')
            self._draw_text_safe(draw, (360, y+8), qty, self.font, 'black')
            self._draw_text_safe(draw, (450, y+8), unit, self.font, 'black')
            self._draw_text_safe(draw, (600, y+8), amount, self.font, 'black')
            y += 45
        
        draw.line((50, y, 750, y), fill='black', width=1)
        y += 20
        
        # Summary
        self._draw_text_safe(draw, (450, y), '小計', self.font, 'black')
        self._draw_text_safe(draw, (600, y), '¥50,000', self.font, 'black')
        y += 40
        
        self._draw_text_safe(draw, (450, y), '消費税(10%)', self.font, 'black')
        self._draw_text_safe(draw, (600, y), '¥5,000', self.font, 'black')
        y += 40
        
        draw.line((450, y, 750, y), fill='black', width=2)
        y += 10
        
        self._draw_text_safe(draw, (450, y), '合計', self.font_large, 'black')
        self._draw_text_safe(draw, (580, y), '¥55,000', self.font_large, 'black')
        y += 80
        
        # Bank info
        draw.line((50, y, 750, y), fill='gray', width=1)
        y += 20
        
        self._draw_text_safe(draw, (50, y), '【お振込先】', self.font, 'black')
        y += 40
        self._draw_text_safe(draw, (50, y), '三菱UFJ銀行 東京営業部', self.font, 'black')
        y += 35
        self._draw_text_safe(draw, (50, y), '普通 1234567', self.font, 'black')
        y += 35
        self._draw_text_safe(draw, (50, y), 'サンプル(カ', self.font, 'black')
        
        return img
    
    def generate_invoice_jp_hotel(self, bg_style: str = None) -> Image.Image:
        """Generate Japanese hotel invoice (宿泊明細書)"""
        img = self._create_background(800, 1000, bg_style or "white")
        draw = ImageDraw.Draw(img)
        
        # Header
        self._draw_text_safe(draw, (280, 30), '御請求明細書', self.font_large, 'black')
        draw.line((50, 90, 750, 90), fill='black', width=2)
        
        # Guest info
        self._draw_text_safe(draw, (50, 110), '様', self.font, 'black')
        draw.line((50, 140, 250, 140), fill='black', width=1)
        
        # Hotel info
        self._draw_text_safe(draw, (500, 110), 'グローリーホテル東京', self.font, 'black')
        self._draw_text_safe(draw, (500, 145), '〒100-0001', self.font_small, 'black')
        self._draw_text_safe(draw, (500, 170), '東京都千代田区○○1-2-3', self.font_small, 'black')
        self._draw_text_safe(draw, (500, 195), 'TEL: 03-1234-5678', self.font_small, 'black')
        
        # Stay info
        self._draw_text_safe(draw, (50, 230), 'ご利用日: 2024年12月07日 〜 12月08日', self.font, 'black')
        self._draw_text_safe(draw, (50, 270), '部屋番号: 1219', self.font, 'black')
        
        draw.line((50, 310, 750, 310), fill='gray', width=1)
        
        # Items
        y = 330
        items = [
            ('シングル室料', '1泊', '¥8,000'),
            ('宿泊税', '', '¥200'),
        ]
        
        for name, qty, amount in items:
            self._draw_text_safe(draw, (60, y), name, self.font, 'black')
            self._draw_text_safe(draw, (400, y), qty, self.font, 'black')
            self._draw_text_safe(draw, (600, y), amount, self.font, 'black')
            y += 45
        
        draw.line((50, y, 750, y), fill='black', width=2)
        y += 20
        
        # Total
        self._draw_text_safe(draw, (400, y), '宿泊料金合計', self.font, 'black')
        self._draw_text_safe(draw, (580, y), '¥8,200', self.font, 'black')
        y += 45
        
        self._draw_text_safe(draw, (400, y), '(内消費税10%対象)', self.font_small, 'gray')
        self._draw_text_safe(draw, (600, y), '¥727', self.font_small, 'gray')
        y += 40
        
        draw.line((400, y, 750, y), fill='black', width=2)
        y += 15
        
        self._draw_text_safe(draw, (400, y), '税込合計金額', self.font_large, 'black')
        self._draw_text_safe(draw, (560, y), '¥8,200', self.font_large, 'black')
        y += 70
        
        # Payment method
        draw.line((50, y, 750, y), fill='gray', width=1)
        y += 20
        
        self._draw_text_safe(draw, (50, y), '【お支払方法】', self.font, 'black')
        y += 40
        self._draw_text_safe(draw, (50, y), 'クレジットカード扱い', self.font, 'black')
        
        return img
    
    def generate_receipt_tilted(self, angle: float = 5.0) -> Image.Image:
        """Generate tilted receipt"""
        img = self.generate_receipt_normal()
        # Rotate with white background
        rotated = img.rotate(angle, expand=True, fillcolor='white', resample=Image.BICUBIC)
        return rotated
    
    def generate_receipt_distorted(self) -> Image.Image:
        """Generate receipt with distorted/wavy text"""
        img = Image.new('RGB', (600, 900), color='white')
        draw = ImageDraw.Draw(img)
        
        def draw_wavy_text(x: int, y: int, text: str, font, amplitude: int = 3, fill='black'):
            """Draw wavy text"""
            char_width = 20
            for i, char in enumerate(text):
                offset_y = int(amplitude * math.sin(i * 0.8))
                offset_x = random.randint(-1, 1)
                self._draw_text_safe(draw, (x + i * char_width + offset_x, y + offset_y), 
                                    char, font, fill)
        
        # Header with wave effect
        draw_wavy_text(200, 30, 'RECEIPT', self.font_large, amplitude=4)
        draw.line((50, 90, 550, 90), fill='black', width=2)
        
        # Store info with slight distortion
        draw_wavy_text(50, 110, 'Test Store', self.font, amplitude=2)
        draw_wavy_text(50, 150, 'TEL: 03-1234-5678', self.font, amplitude=2)
        
        # Date
        draw_wavy_text(50, 200, '2024/12/08 14:30', self.font, amplitude=3)
        draw.line((50, 240, 550, 240), fill='gray', width=1)
        
        # Items
        y = 260
        items = [('Rice Ball', '\\150'), ('Tea *', '\\130'), ('Sandwich *', '\\280')]
        
        for name, price in items:
            draw_wavy_text(50, y, name, self.font, amplitude=2)
            draw_wavy_text(420, y, price, self.font, amplitude=2)
            y += 50
        
        draw.line((50, y, 550, y), fill='gray', width=1)
        y += 20
        
        # Total
        draw_wavy_text(50, y, 'TOTAL', self.font_large, amplitude=3)
        draw_wavy_text(380, y, '\\560', self.font_large, amplitude=3)
        y += 70
        
        self._draw_text_safe(draw, (50, y), '* = Reduced tax rate', self.font_small, 'gray')
        
        return img
    
    def generate_receipt_noisy(self) -> Image.Image:
        """Generate noisy receipt with stains (realistic level)"""
        img = self.generate_receipt_normal()
        
        # Convert to numpy for noise addition
        img_array = np.array(img)
        
        # Add light Gaussian noise (realistic scanner noise)
        noise = np.random.normal(0, 6, img_array.shape).astype(np.int16)
        noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add a few random stains/spots (coffee marks, etc.)
        result = Image.fromarray(noisy)
        draw = ImageDraw.Draw(result)
        
        for _ in range(4):
            x = random.randint(0, 580)
            y = random.randint(0, 880)
            r = random.randint(3, 15)
            gray = random.randint(180, 220)
            # Draw light ellipse stain
            draw.ellipse([x, y, x+r, y+r*0.7], fill=(gray, gray-5, gray-3))
        
        return result
    
    def generate_receipt_low_contrast(self) -> Image.Image:
        """Generate low contrast (faded) receipt"""
        img = self.generate_receipt_normal()
        
        # Reduce contrast and add brightness
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.4)
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.4)
        
        return img
    
    def generate_receipt_shadow(self) -> Image.Image:
        """Generate receipt with realistic shadow overlay"""
        img = self.generate_receipt_normal()
        
        # Create shadow gradient (diagonal)
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Create diagonal shadow band
        width, height = img.size
        for i in range(height):
            # Shadow intensity varies
            intensity = int(80 * (1 - abs(i - height/3) / (height/2)))
            intensity = max(0, min(intensity, 80))
            shadow_draw.line([(0, i), (width, i)], fill=(0, 0, 0, intensity))
        
        # Apply shadow
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, shadow)
        
        return img.convert('RGB')
    
    def generate_receipt_partial_shadow(self) -> Image.Image:
        """Generate receipt with partial shadow (like finger/hand shadow)"""
        img = self.generate_receipt_normal()
        
        # Create partial shadow mask
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Draw irregular shadow shape (simulating hand shadow)
        width, height = img.size
        points = [
            (0, height * 0.3),
            (width * 0.2, height * 0.25),
            (width * 0.5, height * 0.4),
            (width * 0.7, height * 0.35),
            (width, height * 0.5),
            (width, 0),
            (0, 0),
        ]
        shadow_draw.polygon(points, fill=(0, 0, 0, 60))
        
        # Blur the shadow for realism
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, shadow)
        
        return img.convert('RGB')
    
    def generate_receipt_crumpled(self) -> Image.Image:
        """Generate slightly crumpled/wrinkled receipt effect (realistic)"""
        img = self.generate_receipt_normal()
        img_array = np.array(img)
        
        height, width = img_array.shape[:2]
        
        # Create displacement map for subtle wrinkle effect
        x_displacement = np.zeros((height, width), dtype=np.float32)
        y_displacement = np.zeros((height, width), dtype=np.float32)
        
        # Add subtle wrinkles (like paper that was folded once)
        for freq in [0.008, 0.015]:
            for y in range(height):
                for x in range(width):
                    x_displacement[y, x] += 1.0 * math.sin(freq * x * 10 + freq * y * 5)
                    y_displacement[y, x] += 0.8 * math.sin(freq * y * 8 + freq * x * 3)
        
        # Apply displacement
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                map_x[y, x] = x + x_displacement[y, x]
                map_y[y, x] = y + y_displacement[y, x]
        
        # Remap image
        result = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT)
        
        # Add very subtle brightness variation for fold shadows
        brightness_var = np.random.uniform(0.95, 1.05, (height, width, 1))
        result = np.clip(result * brightness_var, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def generate_receipt_faded(self) -> Image.Image:
        """Generate faded thermal paper effect"""
        img = self.generate_receipt_normal()
        
        # Create uneven fading pattern
        img_array = np.array(img).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Create fade mask (more faded at edges and random spots)
        fade_mask = np.ones((height, width), dtype=np.float32)
        
        # Edge fading
        for y in range(height):
            for x in range(width):
                # Vertical fade
                edge_dist = min(y, height - y) / (height / 4)
                edge_dist = min(edge_dist, 1.0)
                
                # Random spots of fading
                if random.random() < 0.001:
                    fade_mask[y, x] = random.uniform(0.3, 0.7)
                else:
                    fade_mask[y, x] = 0.5 + 0.5 * edge_dist
        
        # Apply Gaussian blur to fade mask for smooth transitions
        fade_mask_cv = cv2.GaussianBlur(fade_mask, (51, 51), 0)
        
        # Expand mask to 3 channels
        fade_mask_3ch = np.stack([fade_mask_cv] * 3, axis=2)
        
        # Apply fading
        result = 255 - (255 - img_array) * fade_mask_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Add slight yellow tint (aged paper)
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.98, 0, 255)  # Reduce blue
        result[:, :, 2] = np.clip(result[:, :, 2] + 5, 0, 255)     # Add red
        
        return Image.fromarray(result)
    
    def generate_invoice_normal(self, bg_style: str = None) -> Image.Image:
        """Generate normal invoice"""
        img = self._create_background(800, 1000, bg_style or "white")
        draw = ImageDraw.Draw(img)
        
        # Header
        self._draw_text_safe(draw, (320, 30), 'INVOICE', self.font_large, 'black')
        draw.line((50, 90, 750, 90), fill='black', width=2)
        
        # Recipient
        self._draw_text_safe(draw, (50, 110), 'To: Test Company Ltd.', self.font, 'black')
        
        # Issuer
        self._draw_text_safe(draw, (500, 110), 'From:', self.font_small, 'gray')
        self._draw_text_safe(draw, (500, 140), 'Sample Corp.', self.font, 'black')
        self._draw_text_safe(draw, (500, 180), 'Tokyo, Japan', self.font_small, 'black')
        
        # Invoice info
        self._draw_text_safe(draw, (50, 180), 'Date: 2024/12/08', self.font, 'black')
        self._draw_text_safe(draw, (50, 220), 'Due: 2024/12/31', self.font, 'black')
        self._draw_text_safe(draw, (50, 260), 'Invoice No: INV-2024-001234', self.font, 'black')
        
        draw.line((50, 310, 750, 310), fill='gray', width=1)
        
        # Total box
        draw.rectangle([50, 330, 350, 400], outline='black', width=2)
        self._draw_text_safe(draw, (70, 345), 'Total Amount', self.font, 'black')
        self._draw_text_safe(draw, (180, 345), '\\55,000', self.font_large, 'black')
        
        # Table header
        y = 430
        draw.rectangle([50, y, 750, y+40], fill='#f0f0f0', outline='black')
        self._draw_text_safe(draw, (60, y+8), 'Item', self.font, 'black')
        self._draw_text_safe(draw, (350, y+8), 'Qty', self.font, 'black')
        self._draw_text_safe(draw, (450, y+8), 'Unit Price', self.font, 'black')
        self._draw_text_safe(draw, (600, y+8), 'Amount', self.font, 'black')
        
        y += 40
        items = [
            ('Consulting Fee', '10', '\\5,000', '\\50,000'),
            ('Transportation', '1', '\\5,000', '\\5,000'),
        ]
        
        for name, qty, unit, amount in items:
            draw.line((50, y, 750, y), fill='gray', width=1)
            self._draw_text_safe(draw, (60, y+8), name, self.font, 'black')
            self._draw_text_safe(draw, (360, y+8), qty, self.font, 'black')
            self._draw_text_safe(draw, (450, y+8), unit, self.font, 'black')
            self._draw_text_safe(draw, (600, y+8), amount, self.font, 'black')
            y += 45
        
        draw.line((50, y, 750, y), fill='black', width=1)
        y += 20
        
        # Summary
        self._draw_text_safe(draw, (450, y), 'Subtotal', self.font, 'black')
        self._draw_text_safe(draw, (600, y), '\\50,000', self.font, 'black')
        y += 40
        
        self._draw_text_safe(draw, (450, y), 'Tax (10%)', self.font, 'black')
        self._draw_text_safe(draw, (600, y), '\\5,000', self.font, 'black')
        y += 40
        
        draw.line((450, y, 750, y), fill='black', width=2)
        y += 10
        
        self._draw_text_safe(draw, (450, y), 'Total', self.font_large, 'black')
        self._draw_text_safe(draw, (580, y), '\\55,000', self.font_large, 'black')
        y += 80
        
        # Bank info
        draw.line((50, y, 750, y), fill='gray', width=1)
        y += 20
        
        self._draw_text_safe(draw, (50, y), '[Bank Transfer Info]', self.font, 'black')
        y += 40
        self._draw_text_safe(draw, (50, y), 'ABC Bank - Main Branch', self.font, 'black')
        y += 35
        self._draw_text_safe(draw, (50, y), 'Account: 1234567', self.font, 'black')
        
        return img
    
    def generate_invoice_tilted(self, angle: float = -3.0) -> Image.Image:
        """Generate tilted invoice"""
        img = self.generate_invoice_normal()
        return img.rotate(angle, expand=True, fillcolor='white', resample=Image.BICUBIC)
    
    def generate_invoice_shadow(self) -> Image.Image:
        """Generate invoice with shadow"""
        img = self.generate_invoice_normal()
        
        # Create corner shadow
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        width, height = img.size
        # Bottom-right corner shadow
        for i in range(100):
            alpha = int(50 * (1 - i / 100))
            shadow_draw.rectangle([width - 200 + i, height - 200 + i, 
                                  width, height], fill=(0, 0, 0, alpha))
        
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=20))
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, shadow)
        
        return img.convert('RGB')
    
    def generate_invoice_stamp(self) -> Image.Image:
        """Generate invoice with stamp overlay"""
        img = self.generate_invoice_normal()
        draw = ImageDraw.Draw(img)
        
        # Draw red stamp circle
        stamp_x, stamp_y = 600, 500
        stamp_r = 60
        
        # Draw stamp border
        draw.ellipse([stamp_x - stamp_r, stamp_y - stamp_r, 
                     stamp_x + stamp_r, stamp_y + stamp_r],
                    outline='red', width=3)
        
        # Inner circle
        draw.ellipse([stamp_x - stamp_r + 10, stamp_y - stamp_r + 10, 
                     stamp_x + stamp_r - 10, stamp_y + stamp_r - 10],
                    outline='red', width=2)
        
        # Stamp text (simplified)
        self._draw_text_safe(draw, (stamp_x - 30, stamp_y - 15), 'PAID', self.font, 'red')
        
        return img
    
    def generate_pattern(self, pattern_key: str) -> Image.Image:
        """Generate image for a specific pattern"""
        # Handle font/background variations first
        if pattern_key == "receipt_mincho":
            old_style = self.font_style
            self.font_style = "mincho"
            self._load_fonts()
            img = self.generate_receipt_normal()
            self.font_style = old_style
            self._load_fonts()
            return img
        
        if pattern_key == "receipt_thermal":
            return self.generate_receipt_normal(bg_style="thermal")
        
        if pattern_key == "receipt_aged":
            return self.generate_receipt_normal(bg_style="aged")
        
        if pattern_key == "invoice_mincho":
            old_style = self.font_style
            self.font_style = "mincho"
            self._load_fonts()
            img = self.generate_invoice_normal()
            self.font_style = old_style
            self._load_fonts()
            return img
        
        if pattern_key == "invoice_textured":
            return self.generate_invoice_normal(bg_style="textured")
        
        generators = {
            "receipt_normal": self.generate_receipt_normal,
            "receipt_tilt_5": lambda: self.generate_receipt_tilted(5.0),
            "receipt_tilt_-7": lambda: self.generate_receipt_tilted(-7.0),
            "receipt_distorted": self.generate_receipt_distorted,
            "receipt_noisy": self.generate_receipt_noisy,
            "receipt_low_contrast": self.generate_receipt_low_contrast,
            "receipt_shadow": self.generate_receipt_shadow,
            "receipt_crumpled": self.generate_receipt_crumpled,
            "receipt_faded": self.generate_receipt_faded,
            "receipt_partial_shadow": self.generate_receipt_partial_shadow,
            # Japanese receipt patterns
            "receipt_jp_normal": self.generate_receipt_jp_normal,
            "receipt_jp_thermal": lambda: self.generate_receipt_jp_normal(bg_style="thermal"),
            "receipt_jp_convenience": self.generate_receipt_jp_convenience,
            # Invoice patterns
            "invoice_normal": self.generate_invoice_normal,
            "invoice_tilt_-3": lambda: self.generate_invoice_tilted(-3.0),
            "invoice_shadow": self.generate_invoice_shadow,
            "invoice_stamp": self.generate_invoice_stamp,
            # Japanese invoice patterns
            "invoice_jp_normal": self.generate_invoice_jp_normal,
            "invoice_jp_hotel": self.generate_invoice_jp_hotel,
        }
        
        generator = generators.get(pattern_key, self.generate_receipt_normal)
        return generator()
    
    def generate_all_patterns(self) -> List[Tuple[str, Image.Image]]:
        """Generate all test patterns"""
        results = []
        for key in self.PATTERNS.keys():
            try:
                img = self.generate_pattern(key)
                results.append((key, img))
            except Exception as e:
                print(f"Error generating {key}: {e}")
        return results


class OCRExperimentGUI:
    """OCR Experiment GUI Application"""
    
    # UI Text - English only to avoid encoding issues
    UI_TEXT = {
        "window_title": "Tax OCR Experiment Tool v0.2.0",
        "control_panel": "Controls",
        "test_panel": "Test Image Generator", 
        "input_panel": "Input Image",
        "result_panel": "Preprocessed Image + BBox",
        "extraction_panel": "Extraction Results",
        "btn_open": "Open Image",
        "btn_ocr": "Run OCR",
        "btn_compare": "Compare All",
        "btn_save": "Save Result",
        "btn_clear": "Clear",
        "btn_generate": "Generate",
        "btn_generate_ocr": "Generate + OCR",
        "btn_test_all": "Test All",
        "btn_rotate_cw": "↻ 90°",
        "btn_rotate_ccw": "↺ 90°",
        "label_pattern": "Pattern:",
        "label_doctype": "Doc Type:",
        "label_engine": "Engine:",
        "label_rotation": "Rotate:",
        "chk_bbox": "Show BBox",
        "tab_text": "OCR Text",
        "tab_json": "Data (JSON)",
        "tab_stats": "Statistics",
        "tab_compare": "Engine Compare",
        "status_ready": "Ready - Select pattern and click Generate, or open an image file",
        "status_generating": "Generating test image...",
        "status_generated": "Test image generated",
        "status_processing": "Processing OCR...",
        "status_complete": "OCR Complete",
        "status_cleared": "Cleared",
        "status_rotated": "Image rotated",
        "hint_input": "Select pattern above and click Generate\nor click Open Image to load a file",
        "hint_result": "Run OCR to see results\n\nGreen = High conf (>=80%)\nYellow = Medium (60-79%)\nRed = Low (<60%)",
    }
    
    def __init__(self, config: Optional[GUIConfig] = None):
        self.config = config or GUIConfig()
        self.pipeline = OCRPipeline()
        self.test_generator = TestImageGenerator()
        
        # Available engines
        self.available_engines = get_available_engines()
        self.current_engine = EngineType.TESSERACT
        
        # State
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[Image.Image] = None
        self.preprocessed_image: Optional[Image.Image] = None  # Preprocessed image for display
        self.current_result: Optional[PipelineResult] = None
        self.ocr_result: Optional[OCRResult] = None
        self.preprocess_info: Optional[Dict[str, Any]] = None  # For bbox scaling
        self.temp_files: List[str] = []
        self.comparison_results: Dict[str, Any] = {}  # For engine comparison
        
        # Build GUI
        self._build_gui()
    
    def _build_gui(self):
        """Build the GUI"""
        self.root = tk.Tk()
        self.root.title(self.UI_TEXT["window_title"])
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.root.configure(bg="#f0f0f0")
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Build components
        self._build_control_panel()
        self._build_test_image_panel()
        self._build_image_panels()
        self._build_result_panel()
        self._build_status_bar()
        
        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_control_panel(self):
        """Build control panel"""
        control_frame = ttk.LabelFrame(self.main_frame, text=self.UI_TEXT["control_panel"], padding="5")
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Open button
        self.btn_open = ttk.Button(control_frame, text=self.UI_TEXT["btn_open"], 
                                   command=self._open_file)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # Rotation controls
        ttk.Label(control_frame, text=self.UI_TEXT["label_rotation"]).pack(side=tk.LEFT, padx=(10, 2))
        self.btn_rotate_ccw = ttk.Button(control_frame, text=self.UI_TEXT["btn_rotate_ccw"],
                                         command=lambda: self._rotate_image(-90), state=tk.DISABLED, width=5)
        self.btn_rotate_ccw.pack(side=tk.LEFT, padx=2)
        self.btn_rotate_cw = ttk.Button(control_frame, text=self.UI_TEXT["btn_rotate_cw"],
                                        command=lambda: self._rotate_image(90), state=tk.DISABLED, width=5)
        self.btn_rotate_cw.pack(side=tk.LEFT, padx=2)
        
        # Engine selection
        ttk.Label(control_frame, text=self.UI_TEXT["label_engine"]).pack(side=tk.LEFT, padx=(10, 5))
        self.engine_var = tk.StringVar(value=EngineType.TESSERACT.value)
        engine_values = [e.value for e in self.available_engines]
        engine_display = [e.display_name for e in self.available_engines]
        self.engine_combo = ttk.Combobox(control_frame, textvariable=self.engine_var,
                                         values=engine_values, width=18, state="readonly")
        self.engine_combo.pack(side=tk.LEFT, padx=5)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)
        
        # OCR button
        self.btn_ocr = ttk.Button(control_frame, text=self.UI_TEXT["btn_ocr"],
                                  command=self._run_ocr, state=tk.DISABLED)
        self.btn_ocr.pack(side=tk.LEFT, padx=5)
        
        # Compare All Engines button
        self.btn_compare = ttk.Button(control_frame, text=self.UI_TEXT["btn_compare"],
                                      command=self._compare_all_engines, state=tk.DISABLED)
        self.btn_compare.pack(side=tk.LEFT, padx=5)
        
        # Document type
        ttk.Label(control_frame, text=self.UI_TEXT["label_doctype"]).pack(side=tk.LEFT, padx=(20, 5))
        self.doc_type_var = tk.StringVar(value="auto")
        doc_type_combo = ttk.Combobox(control_frame, textvariable=self.doc_type_var,
                                      values=["auto", "receipt", "invoice"], width=10, state="readonly")
        doc_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.btn_save = ttk.Button(control_frame, text=self.UI_TEXT["btn_save"],
                                   command=self._save_result, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.btn_clear = ttk.Button(control_frame, text=self.UI_TEXT["btn_clear"],
                                    command=self._clear_all)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # BBox toggle
        self.show_bbox_var = tk.BooleanVar(value=True)
        self.chk_bbox = ttk.Checkbutton(control_frame, text=self.UI_TEXT["chk_bbox"],
                                        variable=self.show_bbox_var, command=self._update_result_image)
        self.chk_bbox.pack(side=tk.LEFT, padx=20)
    
    def _build_test_image_panel(self):
        """Build test image generation panel"""
        test_frame = ttk.LabelFrame(self.main_frame, text=self.UI_TEXT["test_panel"], padding="5")
        test_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Pattern selection
        ttk.Label(test_frame, text=self.UI_TEXT["label_pattern"]).pack(side=tk.LEFT, padx=5)
        
        self.test_pattern_var = tk.StringVar(value="receipt_normal")
        pattern_keys = list(TestImageGenerator.PATTERNS.keys())
        pattern_display = [f"{k}" for k in pattern_keys]
        
        self.pattern_combo = ttk.Combobox(test_frame, textvariable=self.test_pattern_var,
                                          values=pattern_keys, width=25, state="readonly")
        self.pattern_combo.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        self.btn_generate = ttk.Button(test_frame, text=self.UI_TEXT["btn_generate"],
                                       command=self._generate_test_image)
        self.btn_generate.pack(side=tk.LEFT, padx=10)
        
        # Generate + OCR button
        self.btn_generate_ocr = ttk.Button(test_frame, text=self.UI_TEXT["btn_generate_ocr"],
                                           command=self._generate_and_run_ocr)
        self.btn_generate_ocr.pack(side=tk.LEFT, padx=5)
        
        # Test all button
        self.btn_test_all = ttk.Button(test_frame, text=self.UI_TEXT["btn_test_all"],
                                       command=self._run_all_patterns_test)
        self.btn_test_all.pack(side=tk.LEFT, padx=5)
    
    def _build_image_panels(self):
        """Build image display panels"""
        image_frame = ttk.Frame(self.main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left: Input image
        left_frame = ttk.LabelFrame(image_frame, text=self.UI_TEXT["input_panel"], padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.input_canvas = tk.Canvas(left_frame, width=self.config.image_max_width,
                                      height=self.config.image_max_height, bg="#e0e0e0",
                                      highlightthickness=1, highlightbackground="#999")
        self.input_canvas.pack(fill=tk.BOTH, expand=True)
        self.input_canvas.create_text(self.config.image_max_width // 2, 
                                      self.config.image_max_height // 2,
                                      text=self.UI_TEXT["hint_input"], font=("", 11),
                                      fill="#666", tags="hint")
        
        # Right: OCR result
        right_frame = ttk.LabelFrame(image_frame, text=self.UI_TEXT["result_panel"], padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.result_canvas = tk.Canvas(right_frame, width=self.config.image_max_width,
                                       height=self.config.image_max_height, bg="#e0e0e0",
                                       highlightthickness=1, highlightbackground="#999")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        self.result_canvas.create_text(self.config.image_max_width // 2,
                                       self.config.image_max_height // 2,
                                       text=self.UI_TEXT["hint_result"], font=("", 11),
                                       fill="#666", tags="hint")
    
    def _build_result_panel(self):
        """Build result display panel"""
        result_frame = ttk.LabelFrame(self.main_frame, text=self.UI_TEXT["extraction_panel"], padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: OCR Text
        text_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(text_frame, text=self.UI_TEXT["tab_text"])
        self.ocr_text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD,
                                                       font=("Consolas", 10), height=6)
        self.ocr_text_area.pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: JSON
        json_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(json_frame, text=self.UI_TEXT["tab_json"])
        self.json_text_area = scrolledtext.ScrolledText(json_frame, wrap=tk.WORD,
                                                        font=("Consolas", 10), height=6)
        self.json_text_area.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Statistics
        stats_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(stats_frame, text=self.UI_TEXT["tab_stats"])
        self.stats_text_area = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD,
                                                         font=("Consolas", 10), height=6)
        self.stats_text_area.pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: Engine Comparison
        compare_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(compare_frame, text=self.UI_TEXT["tab_compare"])
        self.compare_text_area = scrolledtext.ScrolledText(compare_frame, wrap=tk.WORD,
                                                           font=("Consolas", 10), height=6)
        self.compare_text_area.pack(fill=tk.BOTH, expand=True)
    
    def _build_status_bar(self):
        """Build status bar"""
        self.status_var = tk.StringVar(value=self.UI_TEXT["status_ready"])
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _generate_test_image(self):
        """Generate selected test image"""
        pattern = self.test_pattern_var.get()
        self.status_var.set(f"{self.UI_TEXT['status_generating']} {pattern}")
        self.root.update()
        
        try:
            img = self.test_generator.generate_pattern(pattern)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, prefix='ocr_test_')
            img.save(temp_file.name)
            self.temp_files.append(temp_file.name)
            
            self._load_image(temp_file.name)
            self.status_var.set(f"{self.UI_TEXT['status_generated']}: {pattern}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate test image:\n{e}")
            self.status_var.set(f"Error: {e}")
    
    def _generate_and_run_ocr(self):
        """Generate test image and run OCR"""
        self._generate_test_image()
        if self.current_image:
            self._run_ocr()
    
    def _run_all_patterns_test(self):
        """Run test on all patterns"""
        results_window = tk.Toplevel(self.root)
        results_window.title("All Patterns Test Results")
        results_window.geometry("800x600")
        
        result_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD, font=("Consolas", 10))
        result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result_text.insert(tk.END, "=" * 60 + "\n")
        result_text.insert(tk.END, "All Patterns Test Results\n")
        result_text.insert(tk.END, "=" * 60 + "\n\n")
        
        def run_tests():
            patterns = self.test_generator.generate_all_patterns()
            
            for name, img in patterns:
                self.root.after(0, lambda n=name: self.status_var.set(f"Testing: {n}"))
                
                try:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, prefix='ocr_test_')
                    img.save(temp_file.name)
                    self.temp_files.append(temp_file.name)
                    
                    result = self.pipeline.process(temp_file.name)
                    
                    def update_result(n=name, r=result):
                        result_text.insert(tk.END, f"\n[Pattern] {n}\n")
                        result_text.insert(tk.END, "-" * 40 + "\n")
                        result_text.insert(tk.END, f"  Doc Type: {r.document_type.value}\n")
                        result_text.insert(tk.END, f"  Class Conf: {r.confidence:.1%}\n")
                        result_text.insert(tk.END, f"  OCR Conf: {r.ocr_confidence:.1f}%\n")
                        
                        if r.data:
                            data = r.data.to_dict()
                            for key in ['date', 'total_amount', 'store_name', 'vendor_name']:
                                if key in data and data[key]:
                                    result_text.insert(tk.END, f"  {key}: {data[key]}\n")
                        
                        if r.warnings:
                            result_text.insert(tk.END, f"  Warnings: {len(r.warnings)}\n")
                        
                        result_text.insert(tk.END, "\n")
                        result_text.see(tk.END)
                    
                    self.root.after(0, update_result)
                    
                except Exception as e:
                    self.root.after(0, lambda n=name, e=e: result_text.insert(tk.END, f"\n[X] {n}: Error - {e}\n"))
            
            self.root.after(0, lambda: self.status_var.set("All patterns test completed"))
            self.root.after(0, lambda: result_text.insert(tk.END, "\n" + "=" * 60 + "\nTest Complete\n"))
        
        thread = threading.Thread(target=run_tests)
        thread.start()
    
    def _open_file(self):
        """Open file dialog"""
        file_types = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=file_types)
        
        if file_path:
            self._load_image(file_path)
    
    def _load_image(self, file_path: str):
        """Load and display image"""
        try:
            self.current_image_path = file_path
            self.current_image = Image.open(file_path)
            
            # Apply EXIF orientation if present
            try:
                self.current_image = ImageOps.exif_transpose(self.current_image)
            except Exception:
                pass
            
            self._display_image(self.input_canvas, self.current_image, "input_image")
            self.input_canvas.delete("hint")
            self.btn_ocr.config(state=tk.NORMAL)
            self.btn_rotate_cw.config(state=tk.NORMAL)
            self.btn_rotate_ccw.config(state=tk.NORMAL)
            self.status_var.set(f"Image loaded: {Path(file_path).name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def _rotate_image(self, angle: int):
        """
        Rotate the current image by the specified angle.
        
        Args:
            angle: Rotation angle in degrees (positive = clockwise)
        """
        if self.current_image is None:
            return
        
        # Rotate the image (PIL uses counter-clockwise, so negate for clockwise)
        # expand=True ensures the entire image is visible after rotation
        self.current_image = self.current_image.rotate(-angle, expand=True, fillcolor='white')
        
        # Save rotated image to a temporary file for OCR processing
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        self.current_image.save(temp_path)
        
        # Update the temp path for OCR
        if hasattr(self, 'temp_files'):
            self.temp_files.append(temp_path)
        self.current_image_path = temp_path
        
        # Update display
        self._display_image(self.input_canvas, self.current_image, "input_image")
        self.status_var.set(f"{self.UI_TEXT['status_rotated']} ({angle}°)")
    
    def _display_image(self, canvas: tk.Canvas, image: Image.Image, tag: str):
        """Display image on canvas"""
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width() or self.config.image_max_width
        canvas_height = canvas.winfo_height() or self.config.image_max_height
        
        # Maintain aspect ratio
        img_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = min(canvas_width, image.width)
            new_height = int(new_width / img_ratio)
        else:
            new_height = min(canvas_height, image.height)
            new_width = int(new_height * img_ratio)
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        
        canvas.delete(tag)
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER, tags=tag)
        
        if tag == "input_image":
            self.input_photo = photo
        else:
            self.result_photo = photo
    
    def _run_ocr(self):
        """Run OCR processing"""
        if not self.current_image_path:
            return
        
        self.btn_ocr.config(state=tk.DISABLED)
        self.btn_compare.config(state=tk.DISABLED)
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_generate_ocr.config(state=tk.DISABLED)
        self.status_var.set(self.UI_TEXT["status_processing"])
        self.root.update()
        
        def ocr_task():
            try:
                # PaddleOCR系の場合、初回実行時に初期化メッセージを表示
                engine_type = self.pipeline._current_engine_type
                if engine_type in (EngineType.PADDLEOCR, EngineType.PADDLE_VL) and not self.pipeline._engine_initialized:
                    def update_status():
                        self.status_var.set(f"{engine_type.display_name}を初期化中... (初回のみ時間がかかります)")
                    self.root.after(0, update_status)
                
                doc_type_str = self.doc_type_var.get()
                doc_type = None
                if doc_type_str == "receipt":
                    doc_type = DocumentType.RECEIPT
                elif doc_type_str == "invoice":
                    doc_type = DocumentType.INVOICE
                
                # Set the selected engine
                engine_value = self.engine_var.get()
                try:
                    engine_type = EngineType.from_string(engine_value)
                    self.pipeline.set_engine(engine_type)
                except ValueError:
                    pass  # Keep current engine
                
                result = self.pipeline.process(self.current_image_path, doc_type)
                
                preprocessed = self.pipeline.preprocessor.process_file(self.current_image_path)
                
                # エンジンが初期化されていることを確認
                self.pipeline._ensure_engine_initialized()
                ocr_result = self.pipeline._ocr_engine.recognize(preprocessed.image)
                
                # Store preprocessing info for bbox coordinate scaling and stats display
                self.preprocess_info = {
                    'original_size': preprocessed.original_size,
                    'processed_size': preprocessed.processed_size,
                    'roi_detected': preprocessed.roi_detected,
                    'perspective_corrected': preprocessed.perspective_corrected,
                    'preprocessing_applied': preprocessed.preprocessing_applied,
                    'deskew_angle': preprocessed.deskew_angle,
                    'multi_scale_count': len(preprocessed.multi_scale_images) if preprocessed.multi_scale_images else 0
                }
                
                # Store preprocessed image for display
                # Convert to RGB if grayscale for display
                if len(preprocessed.image.shape) == 2:
                    # Grayscale -> RGB
                    preprocessed_rgb = np.stack([preprocessed.image] * 3, axis=-1)
                else:
                    preprocessed_rgb = preprocessed.image
                self.preprocessed_image = Image.fromarray(preprocessed_rgb)
                
                self.root.after(0, lambda: self._display_result(result, ocr_result))
                
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))
        
        thread = threading.Thread(target=ocr_task)
        thread.start()
    
    def _on_engine_change(self, event=None):
        """Handle engine selection change"""
        engine_value = self.engine_var.get()
        try:
            engine_type = EngineType.from_string(engine_value)
            self.current_engine = engine_type
            
            # エンジン切り替え（遅延初期化なので即座に完了）
            self.pipeline.set_engine(engine_type)
            
            # PaddleOCR系の場合は初回実行時に初期化される旨を表示
            if engine_type in (EngineType.PADDLEOCR, EngineType.PADDLE_VL):
                self.status_var.set(f"Engine: {engine_type.display_name} (初回実行時にモデルを読み込みます)")
            else:
                self.status_var.set(f"Engine: {engine_type.display_name}")
        except ValueError as e:
            self.status_var.set(f"Engine error: {e}")
    
    def _compare_all_engines(self):
        """Compare all available OCR engines"""
        if not self.current_image_path:
            return
        
        self.btn_ocr.config(state=tk.DISABLED)
        self.btn_compare.config(state=tk.DISABLED)
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_generate_ocr.config(state=tk.DISABLED)
        self.status_var.set("Comparing all engines...")
        self.root.update()
        
        def compare_task():
            try:
                results = {}
                preprocessed = self.pipeline.preprocessor.process_file(self.current_image_path)
                
                # Convert grayscale to BGR for engines that need it
                if len(preprocessed.image.shape) == 2:
                    image_bgr = cv2.cvtColor(preprocessed.image, cv2.COLOR_GRAY2BGR)
                else:
                    image_bgr = preprocessed.image
                
                for engine_type in self.available_engines:
                    try:
                        self.root.after(0, lambda et=engine_type: 
                                       self.status_var.set(f"Testing {et.display_name}..."))
                        
                        engine = self.pipeline._create_ocr_engine(engine_type)
                        ocr_result = engine.recognize(image_bgr)
                        
                        results[engine_type.value] = {
                            'engine_name': engine_type.display_name,
                            'text': ocr_result.full_text,
                            'confidence': ocr_result.average_confidence,
                            'word_count': len(ocr_result.words),
                            'processing_time': getattr(ocr_result, 'processing_time', 0.0),
                            'warnings': ocr_result.warnings[:5] if ocr_result.warnings else []
                        }
                    except Exception as e:
                        results[engine_type.value] = {
                            'engine_name': engine_type.display_name,
                            'error': str(e),
                            'text': '',
                            'confidence': 0.0,
                            'word_count': 0,
                            'processing_time': 0.0,
                            'warnings': []
                        }
                
                self.comparison_results = results
                self.root.after(0, lambda: self._display_comparison_results(results))
                
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))
        
        thread = threading.Thread(target=compare_task)
        thread.start()
    
    def _display_comparison_results(self, results: Dict[str, Any]):
        """Display engine comparison results"""
        self.btn_ocr.config(state=tk.NORMAL)
        self.btn_compare.config(state=tk.NORMAL)
        self.btn_generate.config(state=tk.NORMAL)
        self.btn_generate_ocr.config(state=tk.NORMAL)
        
        # Switch to comparison tab
        self.notebook.select(3)  # Compare tab
        
        # Format comparison results
        lines = ["=" * 60]
        lines.append("OCR ENGINE COMPARISON RESULTS")
        lines.append("=" * 60)
        lines.append("")
        
        for engine_key, data in results.items():
            lines.append(f"--- {data['engine_name']} ---")
            if 'error' in data:
                lines.append(f"  ERROR: {data['error']}")
            else:
                lines.append(f"  Confidence: {data['confidence']:.1f}%")
                lines.append(f"  Words detected: {data['word_count']}")
                lines.append(f"  Processing time: {data['processing_time']:.3f}s")
                lines.append(f"  Text preview: {data['text'][:100]}..." if len(data['text']) > 100 else f"  Text: {data['text']}")
                if data['warnings']:
                    lines.append(f"  Warnings: {len(data['warnings'])} items")
            lines.append("")
        
        # Summary
        lines.append("=" * 60)
        lines.append("SUMMARY")
        lines.append("-" * 60)
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_conf = max(valid_results.items(), key=lambda x: x[1]['confidence'])
            fastest = min(valid_results.items(), key=lambda x: x[1]['processing_time'])
            most_words = max(valid_results.items(), key=lambda x: x[1]['word_count'])
            
            lines.append(f"  Highest Confidence: {best_conf[1]['engine_name']} ({best_conf[1]['confidence']:.1f}%)")
            lines.append(f"  Fastest: {fastest[1]['engine_name']} ({fastest[1]['processing_time']:.3f}s)")
            lines.append(f"  Most Words: {most_words[1]['engine_name']} ({most_words[1]['word_count']} words)")
        
        lines.append("=" * 60)
        
        self.compare_text_area.delete(1.0, tk.END)
        self.compare_text_area.insert(tk.END, "\n".join(lines))
        
        self.status_var.set(f"Comparison complete - {len(results)} engines tested")
    
    def _display_result(self, result: PipelineResult, ocr_result: OCRResult):
        """Display OCR results"""
        self.current_result = result
        self.ocr_result = ocr_result
        
        self._update_result_image()
        
        self.ocr_text_area.delete(1.0, tk.END)
        ocr_text = result.ocr_text
        if isinstance(ocr_text, bytes):
            try:
                ocr_text = ocr_text.decode('utf-8', errors='replace')
            except Exception:
                ocr_text = str(ocr_text)
        self.ocr_text_area.insert(tk.END, ocr_text)
        
        self.json_text_area.delete(1.0, tk.END)
        if result.data:
            json_str = json.dumps(result.data.to_dict(), ensure_ascii=False, indent=2)
            self.json_text_area.insert(tk.END, json_str)
        else:
            self.json_text_area.insert(tk.END, "No data extracted")
        
        self._display_stats(result, ocr_result)
        
        self.btn_ocr.config(state=tk.NORMAL)
        self.btn_compare.config(state=tk.NORMAL)
        self.btn_generate.config(state=tk.NORMAL)
        self.btn_generate_ocr.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED if not result.data else tk.NORMAL)
        
        # Get engine info
        engine_name = getattr(result, 'engine_name', '') or self.pipeline.get_current_engine().display_name
        processing_time = getattr(result, 'processing_time', 0.0)
        
        self.status_var.set(f"{self.UI_TEXT['status_complete']} - Engine: {engine_name}, "
                           f"Type: {result.document_type.value}, "
                           f"Conf: {result.confidence:.1%}, OCR: {result.ocr_confidence:.1f}%, "
                           f"Time: {processing_time:.3f}s")
    
    def _update_result_image(self):
        """Update result image with bounding boxes on preprocessed image"""
        if not hasattr(self, 'preprocessed_image') or not self.preprocessed_image or not self.ocr_result:
            return
        
        # Use preprocessed image as base (instead of original)
        result_image = self.preprocessed_image.copy()
        
        if self.show_bbox_var.get() and self.ocr_result.words:
            draw = ImageDraw.Draw(result_image)
            
            # No coordinate scaling needed - bbox coordinates are already in preprocessed image space
            for word in self.ocr_result.words:
                if word.confidence > 0:
                    bbox = word.bbox
                    
                    # Use bbox coordinates directly (already in preprocessed image coordinates)
                    left = int(bbox.left)
                    top = int(bbox.top)
                    right = int(bbox.right)
                    bottom = int(bbox.bottom)
                    
                    if word.confidence >= self.config.confidence_high_threshold:
                        color = self.config.bbox_color_high
                    elif word.confidence >= self.config.confidence_mid_threshold:
                        color = self.config.bbox_color_mid
                    else:
                        color = self.config.bbox_color_low
                    
                    draw.rectangle([left, top, right, bottom],
                                  outline=color, width=self.config.bbox_width)
        
        self._display_image(self.result_canvas, result_image, "result_image")
        self.result_canvas.delete("hint")
    
    def _display_stats(self, result: PipelineResult, ocr_result: OCRResult):
        """Display statistics"""
        self.stats_text_area.delete(1.0, tk.END)
        
        stats = [
            "=" * 50,
            "OCR Processing Statistics",
            "=" * 50,
            "",
            f"OCR Engine: {result.engine_name}",
            f"Processing Time: {result.processing_time:.2f} seconds",
            "",
            f"Document Type: {result.document_type.value}",
            f"Classification Confidence: {result.confidence:.2%}",
            f"OCR Confidence: {result.ocr_confidence:.1f}%",
            "",
            "--- Recognition Stats ---",
            f"Words Recognized: {len(ocr_result.words)}",
            f"Blocks Recognized: {len(ocr_result.blocks)}",
        ]
        
        if ocr_result.words:
            high = sum(1 for w in ocr_result.words if w.confidence >= 80)
            mid = sum(1 for w in ocr_result.words if 60 <= w.confidence < 80)
            low = sum(1 for w in ocr_result.words if 0 < w.confidence < 60)
            
            stats.extend([
                f"  High (>=80%): {high} words",
                f"  Medium (60-79%): {mid} words", 
                f"  Low (<60%): {low} words",
            ])
        
        # 前処理情報（ROI検出・透視補正を含む）
        if hasattr(self, 'preprocess_info') and self.preprocess_info:
            stats.extend(["", "--- Image Processing ---"])
            orig = self.preprocess_info.get('original_size')
            proc = self.preprocess_info.get('processed_size')
            if orig:
                stats.append(f"  Original Size: {orig[1]}x{orig[0]} px")
            if proc:
                stats.append(f"  Processed Size: {proc[1]}x{proc[0]} px")
            
            # ROI検出と透視補正の状態
            roi = self.preprocess_info.get('roi_detected', False)
            persp = self.preprocess_info.get('perspective_corrected', False)
            applied = self.preprocess_info.get('preprocessing_applied', [])
            
            stats.append(f"  ROI Detection: {'Applied' if roi else 'Not Applied'}")
            stats.append(f"  Perspective Correction: {'Applied' if persp else 'Not Applied'}")
            
            if applied:
                stats.append(f"  Steps: {', '.join(applied)}")
        
        if result.preprocessing_info:
            stats.extend(["", "--- Preprocessing ---"])
            for k, v in result.preprocessing_info.items():
                stats.append(f"  {k}: {v}")
        
        if result.warnings:
            stats.extend(["", "--- Warnings ---"])
            for w in result.warnings[:10]:
                stats.append(f"  [!] {w}")
        
        self.stats_text_area.insert(tk.END, "\n".join(stats))
    
    def _show_error(self, error_msg: str):
        """Show error"""
        self.btn_ocr.config(state=tk.NORMAL)
        self.btn_generate.config(state=tk.NORMAL)
        self.btn_generate_ocr.config(state=tk.NORMAL)
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("OCR Error", error_msg)
    
    def _save_result(self):
        """Save results"""
        if not self.current_result:
            return
        
        file_path = filedialog.asksaveasfilename(title="Save Result", defaultextension=".json",
                                                  filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        
        if file_path:
            try:
                output_data = self.current_result.to_dict()
                output_data.pop("_debug", None)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                self.status_var.set(f"Result saved: {Path(file_path).name}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{e}")
    
    def _clear_all(self):
        """Clear all"""
        self.current_image_path = None
        self.current_image = None
        self.current_result = None
        self.ocr_result = None
        self.preprocess_info = None
        
        self.input_canvas.delete("all")
        self.result_canvas.delete("all")
        
        self.input_canvas.create_text(self.config.image_max_width // 2, self.config.image_max_height // 2,
                                      text=self.UI_TEXT["hint_input"], font=("", 11), fill="#666", tags="hint")
        self.result_canvas.create_text(self.config.image_max_width // 2, self.config.image_max_height // 2,
                                       text=self.UI_TEXT["hint_result"], font=("", 11), fill="#666", tags="hint")
        
        self.ocr_text_area.delete(1.0, tk.END)
        self.json_text_area.delete(1.0, tk.END)
        self.stats_text_area.delete(1.0, tk.END)
        
        self.btn_ocr.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        
        self.status_var.set(self.UI_TEXT["status_cleared"])
    
    def _on_close(self):
        """Cleanup on close"""
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except:
                pass
        self.root.destroy()
    
    def run(self):
        """Run GUI"""
        self.root.mainloop()


def launch_gui():
    """Launch GUI shortcut"""
    app = OCRExperimentGUI()
    app.run()


if __name__ == "__main__":
    launch_gui()
