import onnxruntime as ort
import numpy as np

def run_inference(model_path="mouse_traj.onnx", dx=100, dy=50):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_data = np.array([[dx, dy]], dtype=np.float32)
    outputs = session.run(None, {"input": input_data})
    traj = outputs[0].reshape(10, 2)
    return traj

if __name__ == "__main__":
    test = run_inference("mouse_traj.onnx", dx=120, dy=80)
    print(test)