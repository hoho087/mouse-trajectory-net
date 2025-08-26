# mouse-trajectory-net

This project collects mouse movement trajectories and trains a neural network to predict cursor paths based on relative displacement `(dx, dy)`  
Hope to resolve the mouse trajectory detection issue  

本專案的目標是透過蒐集滑鼠移動資料，訓練一個神經網路模型，讓電腦能夠根據滑鼠的起點與相對位移 `(dx, dy)`，預測完整的滑鼠軌跡  
希望能解決鼠標軌跡檢測的問題  

![image](https://github.com/hoho087/mouse-trajectory-net/blob/main/image.png)

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
圖形化功能選單，提供資料蒐集、訓練、測試與日誌功能。  

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

---

## Requirements / 環境需求
- Python 3.8+  
- PyTorch  
- onnxruntime  
- pygame  
- numpy  
