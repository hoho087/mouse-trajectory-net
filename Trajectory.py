import pygame
import random
import numpy as np
import winsound
import json
import os

pygame.init()

# 視窗大小
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mouse Trajectory Collector")

# 顏色
WHITE = (230, 230, 230)
BG_COLOR = (25, 25, 35)
COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0), (200, 0, 200)]
TRAJ_COLOR = (100, 100, 255)  # 即時軌跡顏色
POINT_COLOR = (255, 0, 200)   # 點顏色

RADIUS = 20  # 小球參數
POINT_RADIUS = 6  # 點半徑

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
            except Exception:
                pass
    for name in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        try:
            f = pygame.font.SysFont(name, size)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.SysFont(None, size)

font = load_cjk_font(20)

def draw_ball(pos, color, radius=RADIUS):
    pygame.draw.circle(WIN, color, pos, radius)

def interpolate_points(points, num=10):
    if len(points) < 2:
        return np.zeros((num, 2))

    dists = [0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        dists.append(dists[-1] + (dx**2 + dy**2) ** 0.5)

    total_dist = dists[-1]
    if total_dist == 0:
        return np.zeros((num, 2))

    targets = np.linspace(0, total_dist, num)
    new_points = []
    j = 0
    for t in targets:
        while j < len(dists)-1 and dists[j+1] < t:
            j += 1
        if j == len(dists)-1:
            new_points.append(points[-1])
        else:
            ratio = (t - dists[j]) / (dists[j+1] - dists[j] + 1e-8)
            x = points[j][0] + ratio * (points[j+1][0] - points[j][0])
            y = points[j][1] + ratio * (points[j+1][1] - points[j][1])
            new_points.append((x, y))

    return np.array(new_points)

def save_json(dataset, filename="mouse_dataset.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        for (dx_dy, traj_rel) in dataset:
            dx, dy = dx_dy
            record = {
                "relative_move": {"dx": int(round(dx)), "dy": int(round(dy))},
                "trajectory": [[int(round(x)), int(round(y))] for x, y in traj_rel]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(dataset)} samples saved -> {filename}")

def draw_instructions():
    texts = [
        "S：SAVE",
        "Z / D：Undo",
        "ESC：Stop collecting",
        "Click the center ball to start",
        "then the target ball to finish"
    ]
    for i, t in enumerate(texts):
        img = font.render(t, True, WHITE)
        WIN.blit(img, (10, 10 + i*25))

def rel_to_abs_points(traj_rel):
    center = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    arr = np.array(traj_rel, dtype=float) + center
    return arr

def collect_data():
    clock = pygame.time.Clock()
    run = True
    ball_pos = (WIDTH // 2, HEIGHT // 2)
    ball_color = COLORS[0]
    target_pos = None
    target_color = None

    collecting = False
    trajectory = []
    traj_points = None

    dataset = []
    undone = []

    while run:
        WIN.fill(BG_COLOR)

        draw_ball(ball_pos, ball_color)
        if target_pos:
            draw_ball(target_pos, target_color)

        # 即時軌跡
        if len(trajectory) > 1:
            pygame.draw.lines(WIN, TRAJ_COLOR, False, trajectory, 2)

        # 切分點
        if traj_points is not None:
            for x, y in traj_points:
                draw_ball((int(x), int(y)), POINT_COLOR, POINT_RADIUS)

        draw_instructions()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_json(dataset)
                elif event.key == pygame.K_ESCAPE:
                    run = False
                elif event.key in (pygame.K_z, pygame.K_d):  # 撤銷 Undo
                    if dataset:
                        item = dataset.pop()
                        undone.append(item)
                        # 顯示更前一次樣本的切分點（如果還有的話）
                        if dataset:
                            last_rel = dataset[-1][1]
                            traj_points = rel_to_abs_points(last_rel)
                        else:
                            traj_points = None
                        collecting = False
                        trajectory = []
                        target_pos = None
                        target_color = None
                        winsound.Beep(400, 200)
                        print("Undo")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                if (mx - ball_pos[0])**2 + (my - ball_pos[1])**2 <= RADIUS**2 and not target_pos:
                    ball_pos = (mx, my)
                    if random.random() < 0.7:  # 70% 機率落在150內
                        angle = random.uniform(0, 2*np.pi)
                        r = random.uniform(20, 150)
                        tx = int(mx + r * np.cos(angle))
                        ty = int(my + r * np.sin(angle))
                    else:
                        tx = random.randint(50, WIDTH-50)
                        ty = random.randint(50, HEIGHT-50)

                    # 確保不會跑到畫面外
                    target_pos = (max(50, min(WIDTH-50, tx)), max(50, min(HEIGHT-50, ty)))
                    target_color = random.choice(COLORS[1:])
                    collecting = True
                    trajectory = []
                    traj_points = None
                    winsound.Beep(800, 150)

                elif target_pos and (mx - target_pos[0])**2 + (my - target_pos[1])**2 <= RADIUS**2:
                    collecting = False
                    interp = interpolate_points(trajectory, num=10) 
                    traj_points = interp.copy() 
                    traj_rel = (interp - np.array(ball_pos)).tolist() 

                    dx = mx - ball_pos[0]
                    dy = my - ball_pos[1]
                    dataset.append(((dx, dy), traj_rel))
                    undone.clear()

                    winsound.Beep(1200, 200)

                    ball_pos = (WIDTH // 2, HEIGHT // 2)
                    ball_color = COLORS[0]
                    target_pos = None
                    target_color = None

        if collecting:
            mx, my = pygame.mouse.get_pos()
            trajectory.append((mx, my))

        clock.tick(240)

    return dataset

if __name__ == "__main__":
    data = collect_data()
    save_json(data)
    pygame.quit()
