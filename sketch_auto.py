# （ファイル先頭〜インポートまでは元のまま）
import sys
import time
import math
import random
import argparse
import subprocess
import ctypes
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
from pathlib import Path

# --- 必須ライブラリ ---
try:
    import cv2
    import numpy as np
    import pyautogui
    import keyboard
    from scipy import interpolate
    from pydantic_settings import BaseSettings
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.traceback import install
except ImportError as e:
    print(f"必要なライブラリがありません: {e}")
    print("pip install opencv-python numpy pyautogui keyboard scipy rich pydantic-settings")
    sys.exit(1)

install()
console = Console()

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.005

# --- ブラシ関連座標定義 ---
OPTIONS_COORDS = {
    "Brush Type": (412, 123),   # ブラシメニューを開く
    "Brush Size": (30, 683),    # ブラシサイズ（クリックで1px）
}

BRUSH_TYPES = {
    "Pencil":          (453, 168),   # 1 → 鉛筆
    "Calligraphy Brush": (453, 202), # 2 → カリグラフィブラシ
    "Calligraphy Pen":   (453, 251), # 3 → カリグラフィペン
    "Airbrush":          (465, 307), # 4 → エアブラシ
    "Oil Brush":         (469, 338), # 5 → 油彩ブラシ
    "Crayon":            (451, 390), # 6 → クレヨン
    "Marker":            (457, 432), # 7 → マーカー
    "Natural Pencil":    (478, 482), # 8 → 鉛筆（ナチュラル）
    "Watercolor Brush":  (469, 523), # 9 → 水彩ブラシ
}

# --- キャンバス座標定義 ---
CANVAS_COORDS = {
    "left top": (223, 213),
    "right top": (1699, 213),
    "left bottom": (221, 1200),
    "right bottom": (1699, 1198),
}

# 境界の安全パディング（ブラシの太さやリサイズハンドルを考慮）
CANVAS_PADDING = 10

CANVAS_MIN_X = max(CANVAS_COORDS["left top"][0], CANVAS_COORDS["left bottom"][0]) + CANVAS_PADDING
CANVAS_MAX_X = min(CANVAS_COORDS["right top"][0], CANVAS_COORDS["right bottom"][0]) - CANVAS_PADDING
CANVAS_MIN_Y = max(CANVAS_COORDS["left top"][1], CANVAS_COORDS["right top"][1]) + CANVAS_PADDING
CANVAS_MAX_Y = min(CANVAS_COORDS["left bottom"][1], CANVAS_COORDS["right bottom"][1]) - CANVAS_PADDING
CANVAS_W = CANVAS_MAX_X - CANVAS_MIN_X
CANVAS_H = CANVAS_MAX_Y - CANVAS_MIN_Y

# --- 設定 ---
class FudeType(str, Enum):
    SUMI = "sumi"
    FUDE = "fude"
    PENCIL = "pencil"
    CHARCOAL = "charcoal"

class ArtisanConfig(BaseSettings):
    draw_width: int = 800
    brush_type: FudeType = FudeType.SUMI
    base_speed: float = 1000
    global_speed_val: float = 1000.0  # 全体の速度倍率
    global_quality: int = 2000        # 品質係数（デフォルト1000、500=低品質、2000=超高品質）
    friction: float = 1.12
    jitter_amount: float = 1.2
    preview_mode: bool = False
    canvas_offset_x: int = 0
    canvas_offset_y: int = 0
    brush_tool_type: Optional[str] = "Natural Pencil"  # MS Paintで使用するブラシ（None=変更しない）
    brush_size: int =2  # ブラシサイズ（1px基準からの上矢印キー回数 + 1）
    rapid_mode: bool = False

    # 新規追加（クオリティーに関係なく固定で効く遅延）
    reliable_stroke_delay: float = 0.004       # 各ポイント移動後の固定遅延（秒）。デフォルト 0.004
    reliable_stroke_end_delay: float = 0.005   # 各ストローク終了後の固定遅延（秒）。デフォルト 0.005
    reliable_buffer_reset_interval: int = 30   # 何ストロークごとにマウスバッファをリセットするか デフォルト 30

    class Config:
        env_prefix = "ARTISAN_"
    
    # 品質に基づく動的パラメータ
    def get_spacing(self, layer_level=1):
        """ハッチング間隔（品質が高いほど密）"""
        base = 5 - (self.global_quality - 1000) / 500  # 1000→5, 2000→3, 500→6
        return max(1, int(base))
    
    def get_min_len(self):
        """最小ストローク長（品質が高いほど短い線も描く）"""
        base = 5 - (self.global_quality - 1000) / 500
        return max(1, int(base))
    
    def get_angles(self):
        """ハッチング角度リスト（品質が高いほど多角度）"""
        if self.global_quality >= 1500:
            return [30, 60, -30, -60, 0, 90, 45, -45]  # 8方向
        elif self.global_quality >= 1000:
            return [45, -45, 0, 90]  # 4方向（デフォルト）
        else:
            return [45, -45]  # 2方向（低品質）
    
    def get_density(self):
        """スプライン補間密度（品質が高いほど滑らか）"""
        base = 1.8 + (self.global_quality - 1000) / 500  # 1000→1.8, 2000→3.8
        return min(5.0, max(0.5, base / self.global_speed_val))
    
    def get_num_points_multiplier(self):
        """補間点数の倍率"""
        return 1.0 + (self.global_quality - 1000) / 1000  # 1000→1.0, 2000→2.0
    
    def get_min_dist(self):
        """描画時の最小移動距離（品質が高いほど細かく）"""
        base = 2.0 - (self.global_quality - 1000) / 1000  # 1000→2.0, 2000→1.0
        return max(0.5, base * self.global_speed_val)
    
    def get_thresholds(self):
        """階調閾値リスト（品質が高いほど細分化）"""
        if self.global_quality >= 1500:
            return [225, 200, 170, 150, 130, 110, 80, 60]  # 8段階
        elif self.global_quality >= 1000:
            return [225, 170, 115, 60]  # 4段階（デフォルト）
        else:
            return [200, 100]  # 2段階（低品質）
    
    def get_resolution_multiplier(self):
        """解像度倍率"""
        return self.global_quality / 1000  # 1000→1.0x, 2000→2.0x

# --- 魂のデータ構造 ---
@dataclass
class Point:
    x: float
    y: float
    pressure: float = 0.5
    velocity: float = 1.0

@dataclass
class Stroke:
    points: List[Point]
    layer: str = "本描き"
    importance: float = 1.0

    @property
    def start(self): return self.points[0]
    @property
    def end(self): return self.points[-1]

# （Mekiki, Unpitsu は元のまま — 略さずそのまま使えます）
# --- 目利き・運筆・匠 ---
class Mekiki:
    def __init__(self, config: ArtisanConfig):
        self.config = config

    def kantei(self, path: str):
        try:
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            img = None
            
        if img is None: raise ValueError("画像が読み込めません")

        h, w = img.shape[:2]
        
        # キャンバス領域に合わせてはみ出さないギリギリのサイズに拡大（アスペクト比維持）
        scale_w = CANVAS_W / w
        scale_h = CANVAS_H / h
        fit_scale = min(scale_w, scale_h)
        
        nw = int(w * fit_scale)
        nh = int(h * fit_scale)
        
        console.print(f"[yellow]画像サイズ調整: {w}x{h} -> {nw}x{nh} (キャンバス最大化: {CANVAS_W}x{CANVAS_H})[/yellow]")

        img = cv2.resize(img, (nw, nh))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 局所コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        layers = {}

        # --- 品質適応型ハッチング生成 ---
        thresholds = self.config.get_thresholds()
        angles = self.config.get_angles()
        spacing = self.config.get_spacing()
        min_len = self.config.get_min_len()
        
        for i, thresh in enumerate(thresholds):
            _, mask = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY_INV)
            angle = angles[i % len(angles)]
            base_pressure = 0.3 + i * (0.3 / len(thresholds))  # 徐々に濃く
            
            layer_name = f"Level{i+1}"
            strokes = self._generate_hatching(mask, angle=angle, spacing=spacing, 
                                             min_len=min_len, base_pressure=base_pressure)
            if strokes:
                layers[layer_name] = strokes
        
        # 最濃部 + 輪郭の特別処理
        final_thresh = thresholds[-1]
        _, mask_final = cv2.threshold(enhanced, final_thresh, 255, cv2.THRESH_BINARY_INV)
        
        # エッジ抽出
        lap = cv2.Laplacian(enhanced, cv2.CV_8U, ksize=5)
        _, bin_edges = cv2.threshold(lap, 30, 255, cv2.THRESH_BINARY)
        bin_edges = cv2.morphologyEx(bin_edges, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        sharp = cv2.Canny(enhanced, 80, 200)
        
        combined_deep = cv2.bitwise_or(mask_final, bin_edges)
        combined_deep = cv2.bitwise_or(combined_deep, sharp)
        
        layers["LevelFinal"] = self._generate_hatching(
            combined_deep, 
            angle=90, 
            spacing=max(1, spacing - 1),  # より密に
            min_len=max(1, min_len - 1), 
            base_pressure=0.7
        )
        
        return layers, nw, nh

    def _generate_hatching(self, mask, angle, spacing, min_len, base_pressure=0.4):
        h, w = mask.shape
        
        diag = int(math.hypot(w, h))
        center = (w // 2, h // 2)
        
        # マスクを回転
        M_rot = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, M_rot, (w, h))
        
        strokes = []
        # スキャンライン (spacingピクセルごと)
        for y in range(0, h, spacing):
            row = rotated_mask[y, :]
            indices = np.where(row > 127)[0]
            if len(indices) < 2: continue
            
            diffs = np.diff(indices)
            breaks = np.where(diffs > 1)[0]
            
            starts = [indices[0]]
            ends = []
            
            for b in breaks:
                ends.append(indices[b])
                starts.append(indices[b+1])
            ends.append(indices[-1])
            
            for s, e in zip(starts, ends):
                if (e - s) < min_len: continue
                
                # 回転前の座標に戻す
                ptr_s = np.array([[[s, y]]], dtype=np.float32)
                ptr_e = np.array([[[e, y]]], dtype=np.float32)
                
                M_back = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                pt_s_trans = cv2.transform(ptr_s, M_back)[0][0]
                pt_e_trans = cv2.transform(ptr_e, M_back)[0][0]
                
                # 筆圧の設定
                p1 = Point(float(pt_s_trans[0]), float(pt_s_trans[1]), pressure=base_pressure, velocity=1.2)
                p2 = Point(float(pt_e_trans[0]), float(pt_e_trans[1]), pressure=base_pressure * 0.5, velocity=1.5)
                
                strokes.append(Stroke([p1, p2], layer="Hatching", importance=base_pressure))
                
        return strokes

class Unpitsu:
    def __init__(self, config):
        self.config = config

    def neru(self, strokes: List[Stroke]) -> List[Stroke]:
        if not strokes: return []
        strokes.sort(key=lambda s: s.importance, reverse=True)
        result = [strokes.pop(0)]
        cur = result[0].end

        while strokes:
            best_idx = 0
            best_score = float('inf')
            for i, s in enumerate(strokes):
                dist = math.hypot(s.start.x - cur.x, s.start.y - cur.y)
                score = dist - (s.importance * 40)
                if score < best_score:
                    best_score = score
                    best_idx = i
            best = strokes.pop(best_idx)
            result.append(best)
            cur = best.end
            
        return [self._physics(s) for s in result if self._physics(s)]

    def _physics(self, s: Stroke) -> Optional[Stroke]:
        if len(s.points) < 2: return None
        x = [p.x for p in s.points]
        y = [p.y for p in s.points]
        try:
            k = min(3, len(x)-1)
            tck, u = interpolate.splprep([x, y], s=0, k=k)
            
            # 品質適応型密度
            density = self.config.get_density()
            length = sum(math.hypot(x[i+1]-x[i], y[i+1]-y[i]) for i in range(len(x)-1))
            multiplier = self.config.get_num_points_multiplier()
            num = max(5, int(length * density * multiplier))
            u_new = np.linspace(0, 1, num)
            x2, y2 = interpolate.splev(u_new, tck)
            dx, dy = interpolate.splev(u_new, tck, der=1)
            ddx, ddy = interpolate.splev(u_new, tck, der=2)
        except:
            return s

        points = []
        # 元の画像範囲を確認（クランプ用）
        ref_max_x = max(x) if len(x) > 0 else 0
        ref_max_y = max(y) if len(y) > 0 else 0

        for i in range(len(x2)):
            vel2 = dx[i]**2 + dy[i]**2 + 1e-8
            curv = abs(dx[i]*ddy[i] - dy[i]*ddx[i]) / (vel2**1.5)
            
            # 品質適応型の筆圧モデル
            if self.config.global_quality >= 1500:
                # 超高品質: 非線形の精密モデル
                velocity = 1.0 / (1.0 + (curv * 20) ** 1.2)
                velocity = max(0.1, velocity)
                pressure = max(0.1, 1.2 - velocity * 0.7)
            else:
                # 標準品質
                velocity = max(0.2, 1.0 / (1 + curv*12))
                pressure = 1.0 - velocity*0.35

            if i < len(x2)*0.15:
                t = i/(len(x2)*0.15)
                pressure *= t
                velocity *= 0.6 + 0.4*t
            elif i > len(x2)*0.85:
                t = (i - len(x2)*0.85)/(len(x2)*0.15)
                pressure *= (1-t)
                velocity *= 1 + t*1.2

            # スプラインによる画像外へのはみ出しをクランプ
            cx = max(0, min(x2[i], ref_max_x))
            cy = max(0, min(y2[i], ref_max_y))
            points.append(Point(cx, cy, pressure, velocity))
        return Stroke(points, s.layer, s.importance)

# --- Takumi（ここを修正） ---
class Takumi:
    def __init__(self, config: ArtisanConfig):
        self.config = config
        self.sw, self.sh = pyautogui.size()
        self._is_down = False
        self.current_brush = None
        self.current_size = None

    def _move(self, x, y):
        # 座標変換とキャンバス領域内への厳密なクランプ
        tx = int(x + self.config.canvas_offset_x)
        ty = int(y + self.config.canvas_offset_y)
        tx = max(CANVAS_MIN_X, min(tx, CANVAS_MAX_X))
        ty = max(CANVAS_MIN_Y, min(ty, CANVAS_MAX_Y))
        
        if self.config.preview_mode:
            return

        try:
            # 常に moveTo を使用し、ストロークの連続性を維持する
            # （mouseDown/Up は _draw_stroke が管理するため、dragTo による再 click/release は不要）
            pyautogui.moveTo(tx, ty)
        except pyautogui.FailSafeException:
            console.print("[red]マウスが画面端に移動し、フェイルセーフが発動しました。処理を中断します。[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]マウス移動に失敗しました: {e}[/red]")
            sys.exit(1)

    # --------------------------------------------------------------
    # 追加：ブラシサイズスライダーを確実にアクティブにする
    # --------------------------------------------------------------
    def _ensure_brush_size_slider_active(self):
        """ブラシサイズスライダーをクリックしてフォーカスを当てる（最大3回リトライ）"""
        if self.config.preview_mode:
            return

        bsx, bsy = OPTIONS_COORDS["Brush Size"]
        console.print(f"[bold yellow]ブラシサイズスライダーをアクティブ化中 → ({bsx}, {bsy})[/bold yellow]")

        for attempt in range(3):
            # 少しゆっくり移動して確実に
            pyautogui.moveTo(bsx, bsy, duration=0.25)
            time.sleep(0.15)
            pyautogui.click(clicks=2, interval=0.15)   # ダブルクリックで確実にフォーカス
            time.sleep(0.25)

            # 成功判定（マウスカーソルがスライダー付近に留まっていればOK）
            cur = pyautogui.position()
            if abs(cur[0] - bsx) <= 20 and abs(cur[1] - bsy) <= 20:
                console.print(f"[dim green]スライダーアクティブ化成功 (attempt {attempt+1})[/dim green]")
                return

            console.print(f"[dim yellow]スライダーアクティブ化再試行… (attempt {attempt+2})[/dim yellow]")

        console.print("[red]ブラシサイズスライダーのアクティブ化に失敗しました。手動でクリックしてください。[/red]")
        # 失敗しても続行（描画はできるがサイズが変わらない可能性あり）

    # --------------------------------------------------------------
    # ブラシ種類変更（変更後に必ずスライダーアクティブ化）
    # --------------------------------------------------------------
    def _set_brush_type(self, type_name: str):
        if self.current_brush == type_name:
            return

        if type_name not in BRUSH_TYPES:
            console.print(f"[yellow]警告: ブラシタイプ '{type_name}' が見つかりません[/yellow]")
            return

        if self.config.preview_mode:
            self.current_brush = type_name
            return

        try:
            if self._is_down:
                pyautogui.mouseUp()
                self._is_down = False

            # 1. ブラシメニューを開く
            bx, by = OPTIONS_COORDS["Brush Type"]
            pyautogui.moveTo(bx, by, duration=0.15)
            pyautogui.click()
            time.sleep(0.25)

            # 2. 目的のブラシを選択
            tx, ty = BRUSH_TYPES[type_name]
            pyautogui.moveTo(tx, ty, duration=0.15)
            pyautogui.click()
            time.sleep(0.25)

            self.current_brush = type_name
            console.print(f"[dim green]ブラシを '{type_name}' に変更しました[/dim green]")

            # 3. 変更後すぐにブラシサイズスライダーをアクティブ化（これが肝！）
            self._ensure_brush_size_slider_active()

        except Exception as e:
            console.print(f"[red]ブラシ変更失敗 ({type_name}): {e}[/red]")

    # --------------------------------------------------------------
    # ブラシサイズ変更（最初にスライダーアクティブ化）
    # --------------------------------------------------------------
    def _set_brush_size(self, size: int):
        if self.current_size == size:
            return

        if self.config.preview_mode:
            self.current_size = size
            return

        try:
            if self._is_down:
                pyautogui.mouseUp()
                self._is_down = False
                time.sleep(0.1)

            # ブラシサイズスライダーをクリック（これで1pxになる）
            bsx, bsy = OPTIONS_COORDS["Brush Size"]
            console.print(f"[bold yellow]ブラシサイズスライダーをクリック: ({bsx}, {bsy})[/bold yellow]")
            
            # 確実にクリックするために複数回試行
            for attempt in range(3):
                pyautogui.moveTo(bsx, bsy, duration=0.2)
                time.sleep(0.15)
                pyautogui.click()
                time.sleep(0.1)
                
                # クリックが成功したか確認（マウスの位置をチェック）
                current_pos = pyautogui.position()
                if abs(current_pos[0] - bsx) < 5 and abs(current_pos[1] - bsy) < 5:
                    console.print(f"[dim green]スライダークリック成功 (attempt {attempt + 1})[/dim green]")
                    break
                else:
                    console.print(f"[dim yellow]スライダークリック再試行 (attempt {attempt + 1})[/dim yellow]")
            
            time.sleep(0.2)
            
            # brush_sizeの回数だけ上矢印キーを押す
            up_count = max(0, size)
            if up_count > 0:
                console.print(f"[dim yellow]上矢印キーを {up_count} 回押します (現在のサイズ: 1px → 目標: {up_count + 1}px)[/dim yellow]")
                pyautogui.press('up', presses=up_count, interval=0.05)
                time.sleep(0.2)
                console.print(f"[bold green]ブラシサイズを {up_count + 1}px に設定完了[/bold green]")
            else:
                console.print(f"[bold green]ブラシサイズを 1px に設定完了[/bold green]")

            self.current_size = size

        except Exception as e:
            console.print(f"[red]サイズ変更失敗 ({size}): {e}[/red]")

    # 以下、_check_abort / egaku / _draw_stroke / _finish_ceremony は変更なし
    # （元のコードをそのまま貼り付けてください）

    def _check_abort(self):
        if keyboard.is_pressed("esc"):
            console.print("\n[red]筆を置きました。[/red]")
            sys.exit(0)

    def egaku(self, layers: Dict[str, List[Stroke]]):
        total = sum(len(v) for v in layers.values())
        console.print(Panel(f"[bold white]魂の筆 V2、始まります。[/bold white]\n総{total}筆", style="white on black"))
        
        with Progress(SpinnerColumn("dots"), TextColumn("{task.description}"), BarColumn(), 
                      TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn()) as p:
            
            # レイヤー順序と表示名
            layer_order = ["Level1", "Level2", "Level3", "Level4"]
            display_names = {
                "Level1": "[dim]第一層（45度）…",
                "Level2": "[dim]第二層（-45度）…",
                "Level3": "[bold white]第三層（水平）…",
                "Level4": "[cyan]第四層（垂直・詳細）…"
            }
            
            tasks = {}
            stroke_count = 0
            for name in layer_order:
                if name in layers and len(layers[name]) > 0:
                    tasks[name] = p.add_task(display_names[name], total=len(layers[name]))
            
            for layer_name in layer_order:
                if layer_name not in layers: continue
                # 速度調整: ハッチングは描画量が多いので高速に
                speed_mult = {
                    "Level1": 4.0, 
                    "Level2": 4.0, 
                    "Level3": 3.0, 
                    "Level4": 2.0
                }.get(layer_name, 1.0)
                
                for stroke in layers[layer_name]:
                    self._draw_stroke(stroke, speed_mult)
                    if layer_name in tasks:
                        p.advance(tasks[layer_name])

                    # 定期的にバッファリセット
                    stroke_count += 1
                    if stroke_count % self.config.reliable_buffer_reset_interval == 0:
                        reset_windows_mouse_buffer()
                        time.sleep(0.05)  # リセット後の安定待ち

        self._finish_ceremony()

    def _draw_stroke(self, stroke: Stroke, speed_mult: float):
        self._check_abort()
        start = stroke.start

        # 余計な線を描かないためにまず離す
        if not self.config.preview_mode:
            try:
                pyautogui.mouseUp(button='left')
            except Exception:
                pass
            self._is_down = False

        # 筆を運ぶ（移動のみ）
        self._move(start.x, start.y)
        # 構えウェイトなし

        # 長押し開始
        if not self.config.preview_mode:
            try:
                pyautogui.mouseDown(button='left')
            except pyautogui.FailSafeException:
                console.print("[red]フェイルセーフによる中断。[/red]")
                sys.exit(1)
            except Exception as e:
                console.print(f"[red]mouseDown に失敗しました: {e}[/red]")
                sys.exit(1)
            # time.sleep(0.02) # Removed
            self._is_down = True

        # prev_pressure = stroke.points[0].pressure # Unused

        # 品質適応型の最小移動距離
        min_dist = self.config.get_min_dist()
        last_x, last_y = start.x, start.y

        # 固定の遅延値を使用（クオリティーやspeed_multの影響を受けない）
        point_delay = self.config.reliable_stroke_delay
        end_delay = self.config.reliable_stroke_end_delay

        for i in range(1, len(stroke.points)):
            pt = stroke.points[i]
            dist = math.hypot(pt.x - last_x, pt.y - last_y)
            if dist < min_dist: continue

            self._move(pt.x, pt.y)
            last_x, last_y = pt.x, pt.y

            # 確実な反映のために毎ポイント後に固定遅延
            if not self.config.preview_mode and not self.config.rapid_mode:
                time.sleep(point_delay)

        # 抜き・跳ね・終了
        if not self.config.preview_mode:
            try:
                pyautogui.mouseUp(button='left')
            except Exception as e:
                console.print(f"[red]mouseUp に失敗しました: {e}[/red]")
            finally:
                self._is_down = False
            
            # ストローク終了後に少し長めの遅延を入れて次のストロークを待つ
            if not self.config.rapid_mode:
                time.sleep(end_delay)

    def _finish_ceremony(self):
        if self.config.preview_mode: return
        console.print("[bold green]完成。[/bold green]")
        # time.sleep(1.0) # Removed

        # 落款だけ押す
        pyautogui.click()         
        # time.sleep(0.6) # Removed
        console.print("[bold magenta]筆を置きました。ありがとうございました。[/bold magenta]")

# --- Windowsマウス入力バッファリセット ---
def reset_windows_mouse_buffer():
    """Windowsのマウス入力バッファをクリア（pyautoguiの高速連打による飽和を解消）"""
    try:
        # Windows APIを使用してメッセージキューをクリア
        user32 = ctypes.windll.user32
        
        # PeekMessageでキューを空にする
        msg = ctypes.wintypes.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE = 1
            pass
        
        # マウスイベントをフラッシュ
        user32.GetQueueStatus(0x00FF)  # QS_ALLINPUT
        
        console.print("[dim green]マウス入力バッファをリセットしました。[/dim green]")
        time.sleep(0.1)  # システムの安定化待ち
    except Exception as e:
        console.print(f"[dim yellow]バッファリセット中にエラー（無視可能）: {e}[/dim yellow]")

# --- 起動 ---
def open_paint_safe():
    """ペイントを安全に起動・最大化"""
    console.print("[yellow]キャンバス（MS Paint）を準備しています...[/yellow]")
    try:
        subprocess.Popen('mspaint')
        time.sleep(1.2)
        wins = pyautogui.getWindowsWithTitle('ペイント')
        if not wins:
            # 英語環境などで 'Paint' になる可能性に対応
            wins = pyautogui.getWindowsWithTitle('Paint')
        if wins:
            win = wins[0]
            try:
                if not win.isMaximized:
                    win.maximize()
                win.activate()
                time.sleep(0.35)  # フォーカスが確実に入るよう短い待ち
            except Exception as e:
                console.print(f"[red]ウィンドウ操作に失敗しました: {e}[/red]")
    except Exception as e:
        console.print(f"[red]ペイントの起動に失敗しました: {e}[/red]")
        console.print("手動で起動してください。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--brush", choices=["sumi","fude","pencil","charcoal"], default="sumi")
    parser.add_argument("--speed", type=float, default=10.0, help="描画速度倍率 (デフォルト: 10.0)")
    parser.add_argument("--rapid-mode", action="store_true", help="超高速描画モード (遅延ゼロ)")
    args = parser.parse_args()

    config = ArtisanConfig(
        preview_mode=args.preview, 
        brush_type=args.brush, 
        base_speed=1000, 
        global_speed_val=args.speed,
        rapid_mode=args.rapid_mode
    )

    if args.rapid_mode:
        console.print("[bold red]!!! RAPID MODE ACTIVATED: Delays Eliminated !!![/bold red]")
        pyautogui.PAUSE = 0.0
        config.reliable_stroke_delay = 0.0
        config.reliable_stroke_end_delay = 0.0
        config.reliable_buffer_reset_interval = 999999

    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title="魂を吹き込む画像を選んでください", filetypes=[("画像", "*.jpg *.jpeg *.png *.webp")])
    root.destroy()
    if not path: return

    mekiki = Mekiki(config)
    try:
        layers, w, h = mekiki.kantei(path)
    except Exception as e:
        console.print(f"[red]鑑定中にエラーが発生しました: {e}[/red]")
        return

    unpitsu = Unpitsu(config)
    for k in list(layers.keys()):
        layers[k] = unpitsu.neru(layers[k])

    config.canvas_offset_x = CANVAS_MIN_X + (CANVAS_W - w) // 2
    config.canvas_offset_y = CANVAS_MIN_Y + (CANVAS_H - h) // 2

    if args.preview:
        img = np.ones((h+100, w+100, 3), np.uint8) * 255
        
        # 描画順序と色設定
        # (レイヤー名, (B,G,R), 標準太さ)
        draw_config = [
            ("Level1", (200, 200, 200), 1),
            ("Level2", (150, 150, 150), 1),
            ("Level3", (100, 100, 100), 1),
            ("Level4", (20, 20, 20), 1)
        ]

        for name, color, base_thick in draw_config:
            if name not in layers: continue
            for s in layers[name]:
                pts = np.array([[int(p.x), int(p.y)] for p in s.points])
                if len(pts) < 2: continue
                # 筆圧に応じた描画
                for i in range(len(pts)-1):
                    # プレビュー用太さ計算 (筆圧 * base_thick)
                    thick_real = max(1, int(s.points[i].pressure * 3 * base_thick))
                    cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), color, thick_real, cv2.LINE_AA)
                cv2.imshow("Preview - 魂の筆", img)
                cv2.waitKey(1)
        cv2.waitKey(0); cv2.destroyAllWindows()
        return

    open_paint_safe()
    time.sleep(1)
    
    # キャンバスサイズをウィンドウサイズに合わせる
    pyautogui.click(1587, 1257)
  
    
    takumi = Takumi(config)
    
    # ブラシの設定（config.brush_tool_typeで指定）
    if config.brush_tool_type:
        console.print(f"[cyan]ブラシを '{config.brush_tool_type}' に設定中...[/cyan]")
        takumi._set_brush_type(config.brush_tool_type)
     
    
    # ブラシサイズの設定（config.brush_sizeで指定）
    # brush_size >= 0 で判定（0の場合も1pxに設定）
    if config.brush_size is not None:
        console.print(f"[cyan]ブラシサイズを {config.brush_size + 1}px に設定中...[/cyan]")
        takumi._set_brush_size(config.brush_size)
      

    takumi.egaku(layers)
    
    # プログラム終了時にバッファをリセット
    reset_windows_mouse_buffer()

if __name__ == "__main__":
    main()

