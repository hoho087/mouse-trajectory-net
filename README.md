# mouse-trajectory-net

This project is to train a simple neural network model by collecting mouse movement data, enabling the computer to predict mouse trajectories resembling those of a human based on the starting point and relative displacement `(dx, dy)` of the mouse    
Hope to resolve the mouse trajectory detection issue  
Since it will be used in conjunction with control algorithms such as PID, the trajectory is not segmented based on time but rather divided according to its total length  

本專案的目標是透過蒐集滑鼠移動資料，訓練一個簡單的神經網路模型，讓電腦能夠根據滑鼠的起點與相對位移 `(dx, dy)`，預測擬人滑鼠軌跡  
希望能解決鼠標軌跡檢測的問題  
因為會與pid等控制算法共同使用，所以切分軌跡並不使用時間去切分，而是以軌跡的總長度切分  

![image](https://github.com/hoho087/mouse-trajectory-net/blob/main/figure.png)

---

## Features / 功能
- Interactive tool to **collect mouse trajectories** with Pygame  
- Data stored in JSONL format  
- Neural network (MLP) with PyTorch for trajectory prediction  
- Export trained model to **ONNX** for fast inference  
- GUI system for data collection, training, testing, and logs  

- 使用 Pygame 建立UI介面，蒐集滑鼠移動軌跡  
- 資料以 JSONL 格式儲存  
- 使用 PyTorch 建立 MLP 神經網路，進行軌跡預測  
- 模型匯出為 **ONNX** 格式以加速推理  
- 提供圖形化介面，可執行資料蒐集、模型訓練、測試與日誌檢視  

---

## Installation / 安裝
```bash
git clone https://github.com/hoho087/mouse-trajectory-net.git
cd mouse-trajectory-net
pip install -r requirements.txt
```

---

## Usage / 使用方法

### GUI System / 圖形化系統
```bash
python main.py
```
Interactive menu for data collection, training, testing, and logs.  
You can switch the interface language(between English and Chinese)  
圖形化功能選單，提供資料蒐集、訓練、測試與日誌功能。  
可以在介面中切換中英顯示  

---

## Dataset Format / 資料格式
Each record is stored as JSONL:  
每筆資料以 JSONL 格式儲存：  

```json
{"relative_move": {"dx": -20, "dy": -50}, "trajectory": [[0, 0], [-2, -4]...
```

---

## Model / 模型
- **Input 輸入**: `(dx, dy)`  
- **Output 輸出**: 10 trajectory points (20 values) / 10 個軌跡點 (共 20 個數值)  
- **Architecture 架構**: Multi-Layer Perceptron (MLP) with ReLU  
