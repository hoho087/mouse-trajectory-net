import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import sys
import numpy as np
import winsound
import io
import contextlib
import Trajectory
import train_model
import test_model

pygame.init()

WIDTH, HEIGHT = 1000, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mouse Trajectory System")

# 暗色調顏色
BG_COLOR = (25, 25, 35)
PANEL_COLOR = (35, 35, 50)
CANVAS_COLOR = (45, 45, 60)
WHITE = (230, 230, 230)
SCROLLBAR_COLOR = (100, 100, 100)

BTN_COLORS = {
    "collect": (52, 152, 219),
    "train":   (39, 174, 96),
    "test":    (192, 57, 43),
    "log":     (155, 89, 182),
    "quit":    (127, 140, 141)
}
BTN_HOVER_OFFSET = 30
INPUT_BG = (60, 60, 80)
INPUT_ACTIVE = (100, 100, 140)

def load_cjk_font(size=20):
    candidates_path = [
        "SourceHanSansTC-Heavy.otf",            
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\mingliu.ttc"
    ]
    for p in candidates_path:
        if os.path.exists(p):
            try:
                return pygame.font.Font(p, size)
            except:
                pass
    return pygame.font.SysFont("Arial Unicode MS", size)

FONT = load_cjk_font(22)
FONT_SMALL = load_cjk_font(18)

# 日誌攔截
class LogBuffer(io.StringIO):
    def __init__(self):
        super().__init__()
        self.lines = []

    def write(self, text):
        sys.__stdout__.write(text)
        for line in text.splitlines():
            if line.strip():
                self.lines.append(line)
        return len(text)

    def flush(self):
        sys.__stdout__.flush()

log_buffer = LogBuffer()
sys.stdout = log_buffer
sys.stderr = log_buffer

def adjust_color(color, offset):
    r = max(0, min(255, color[0] + offset))
    g = max(0, min(255, color[1] + offset))
    b = max(0, min(255, color[2] + offset))
    return (r,g,b)

class Button:
    def __init__(self, text, x, y, w, h, color, action):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.base_color = color
        self.color = color
        self.action = action
        self.pressed = False

    def draw(self, win, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.color = adjust_color(self.base_color, BTN_HOVER_OFFSET)
        else:
            self.color = self.base_color
        if self.pressed:
            self.color = adjust_color(self.base_color, -40)
        pygame.draw.rect(win, self.color, self.rect, border_radius=12)
        txt = FONT.render(self.text, True, WHITE)
        win.blit(txt, (self.rect.centerx - txt.get_width()//2,
                       self.rect.centery - txt.get_height()//2))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# ===================== 日誌面板 =====================
def safe_render(text, font, color):
    filtered = "".join(ch for ch in text if font.metrics(ch)[0] is not None)
    return font.render(filtered, True, color)

def draw_logs(surface, logs, scroll_offset):
    y = 20 - scroll_offset
    max_width = surface.get_width() - 40
    line_height = FONT_SMALL.get_linesize()

    for line in logs:
        while FONT_SMALL.size(line)[0] > max_width:
            cut = len(line)
            while FONT_SMALL.size(line[:cut])[0] > max_width and cut > 0:
                cut -= 1
            txt = safe_render(line[:cut], FONT_SMALL, WHITE)
            surface.blit(txt, (20, y))
            y += line_height
            line = line[cut:]
        txt = safe_render(line, FONT_SMALL, WHITE)
        surface.blit(txt, (20, y))
        y += line_height

def draw_scrollbar(surface, total_lines, scroll_offset):
    visible_height = HEIGHT - 150
    content_height = total_lines * FONT_SMALL.get_linesize()
    if content_height <= visible_height:
        return None
    bar_height = max(40, visible_height * visible_height // content_height)
    bar_y = int(scroll_offset * visible_height / content_height)
    rect = pygame.Rect(surface.get_width()-25, 100+bar_y, 10, bar_height)
    pygame.draw.rect(surface, SCROLLBAR_COLOR, rect)
    return rect

# ===================== 測試模型 =====================
def test_model_main(dx=100, dy=50):
    if not os.path.exists("mouse_traj.onnx"):
        return None
    traj = test_model.run_inference("mouse_traj.onnx", dx, dy)
    return traj

# ===================== 主選單 =====================
def main_menu():
    clock = pygame.time.Clock()
    buttons = [
        Button("收集資料", 50, 100, 180, 50, BTN_COLORS["collect"], "collect"),
        Button("訓練模型", 50, 180, 180, 50, BTN_COLORS["train"], "train"),
        Button("測試模型", 50, 260, 180, 50, BTN_COLORS["test"], "test"),
        Button("查看日誌", 50, 340, 180, 50, BTN_COLORS["log"], "log"),
        Button("退出",     50, 420, 180, 50, BTN_COLORS["quit"], "quit"),
    ]

    run = True
    while run:
        mouse_pos = pygame.mouse.get_pos()
        WIN.fill(BG_COLOR)
        pygame.draw.rect(WIN, PANEL_COLOR, (0,0,280,HEIGHT))

        title = FONT.render("功能選單", True, WHITE)
        WIN.blit(title, (90,40))

        for btn in buttons:
            btn.draw(WIN, mouse_pos)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx,my = event.pos
                for btn in buttons:
                    if btn.is_clicked((mx,my)):
                        btn.pressed = True
                        winsound.Beep(700,100)
                        if btn.action=="collect":
                            Trajectory.collect_data()
                        elif btn.action=="train":
                            train_model.train_model("mouse_dataset.jsonl")
                        elif btn.action=="test":
                            test_page()
                        elif btn.action=="log":
                            log_page()
                        elif btn.action=="quit":
                            run=False
            elif event.type == pygame.MOUSEBUTTONUP:
                for btn in buttons:
                    btn.pressed = False

        clock.tick(60)

    pygame.quit()
    sys.exit()

# ===================== 測試頁面 =====================
def test_page():
    clock = pygame.time.Clock()
    dx_input, dy_input = "", ""
    active_box = None
    test_traj = None
    test_dx, test_dy = 0, 0

    run = True
    while run:
        WIN.fill(BG_COLOR)
        title = FONT.render("測試模型 (ESC 返回)", True, WHITE)
        WIN.blit(title, (50,40))

        dx_rect = pygame.Rect(50, 100, 100, 40)
        dy_rect = pygame.Rect(170, 100, 100, 40)
        pygame.draw.rect(WIN, INPUT_ACTIVE if active_box=="dx" else INPUT_BG, dx_rect, border_radius=8)
        pygame.draw.rect(WIN, INPUT_ACTIVE if active_box=="dy" else INPUT_BG, dy_rect, border_radius=8)
        txt1 = FONT.render(dx_input or "dx", True, WHITE)
        txt2 = FONT.render(dy_input or "dy", True, WHITE)
        WIN.blit(txt1, (dx_rect.centerx - txt1.get_width()//2, dx_rect.centery - txt1.get_height()//2))
        WIN.blit(txt2, (dy_rect.centerx - txt2.get_width()//2, dy_rect.centery - txt2.get_height()//2))

        # 畫布
        canvas = pygame.Surface((WIDTH-300, HEIGHT-150))
        canvas.fill(CANVAS_COLOR)
        if test_traj is not None:
            cx, cy = (WIDTH-300)//2, (HEIGHT-150)//2
            pygame.draw.circle(canvas, (0,200,0), (cx, cy), 8)
            pygame.draw.circle(canvas, (200,0,0), (cx+test_dx, cy+test_dy), 8)
            pts = [(int(cx+x), int(cy+y)) for x,y in test_traj]
            if len(pts)>1:
                pygame.draw.lines(canvas, (100,150,255), False, pts, 3)
        WIN.blit(canvas, (280,150))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:
                    run=False
                elif active_box:
                    if event.key==pygame.K_BACKSPACE:
                        if active_box=="dx": dx_input=dx_input[:-1]
                        else: dy_input=dy_input[:-1]
                    elif event.unicode.isdigit() or (event.unicode=="-" and not (active_box=="dx" and dx_input) and not (active_box=="dy" and dy_input)):
                        if active_box=="dx": dx_input+=event.unicode
                        else: dy_input+=event.unicode
                if event.key==pygame.K_RETURN:
                    try: dx=int(dx_input); dy=int(dy_input)
                    except: dx,dy=100,50
                    test_traj=test_model_main(dx,dy)
                    test_dx,test_dy=dx,dy
            elif event.type==pygame.MOUSEBUTTONDOWN:
                mx,my=event.pos
                if dx_rect.collidepoint((mx,my)): active_box="dx"
                elif dy_rect.collidepoint((mx,my)): active_box="dy"
                else: active_box=None

        clock.tick(60)

# ===================== 日誌頁面 =====================
def log_page():
    clock = pygame.time.Clock()
    scroll_offset = 0
    scroll_dragging = False
    scrollbar_rect = None
    run = True

    while run:
        WIN.fill(BG_COLOR)
        title=FONT.render("日誌面板 (ESC 返回)",True,WHITE)
        WIN.blit(title,(50,40))

        log_area = pygame.Surface((WIDTH-100, HEIGHT-150))
        log_area.fill(CANVAS_COLOR)
        draw_logs(log_area, log_buffer.lines, scroll_offset)
        WIN.blit(log_area, (50,100))

        scrollbar_rect = draw_scrollbar(WIN, len(log_buffer.lines), scroll_offset)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
            elif event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE:
                run=False
            elif event.type==pygame.MOUSEWHEEL:
                scroll_offset -= event.y * 20
                scroll_offset = max(0, scroll_offset)
            elif event.type==pygame.MOUSEBUTTONDOWN:
                if scrollbar_rect and scrollbar_rect.collidepoint(event.pos):
                    scroll_dragging = True
                    drag_offset = event.pos[1] - scrollbar_rect.y
            elif event.type==pygame.MOUSEBUTTONUP:
                scroll_dragging = False
            elif event.type==pygame.MOUSEMOTION and scroll_dragging:
                visible_height = HEIGHT - 150
                content_height = max(1, len(log_buffer.lines) * FONT_SMALL.get_linesize())
                bar_height = max(40, visible_height * visible_height // content_height)
                new_y = event.pos[1] - drag_offset - 100
                new_y = max(0, min(new_y, visible_height - bar_height))
                scroll_offset = int(new_y * content_height / visible_height)

        clock.tick(60)

if __name__=="__main__":
    main_menu()
