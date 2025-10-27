# 🧠 Project: FPGA Hardware Accelerator for Gesture Recognition

## 📍 Step 1 — Environment Setup

This document describes the software and hardware setup, installation process, validation steps, and test procedures for the **Gesture Recognition on FPGA (UltraZed-EG)** project using **FINN** or **HLS4ML**.

---

## 🧰 1. Hardware Environment

| Component               | Description                                    |
| ----------------------- | ---------------------------------------------- |
| **Main board**          | AVNET UltraZed-EG SOM                          |
| **Carrier board**       | UltraZed I/O Carrier Card                      |
| **FPGA**                | Xilinx Zynq UltraScale+ (XCZU7EV)              |
| **Processor**           | ARM Cortex-A53 Quad-core (PS)                  |
| **Camera**              | USB or MIPI module (depending on support)      |
| **Display / Interface** | HDMI or UART console                           |
| **Power Supply**        | 12 V / 4 A minimum                             |
| **Storage**             | MicroSD (≥16 GB) with Linux or PetaLinux image |

---

## 💻 2. Software Environment

| Category                | Tool / Version               | Purpose                          |
| ----------------------- | ---------------------------- | -------------------------------- |
| **Operating System**    | Arch Linux                   | Development host                 |
| **Python**              | 3.10 (conda env `gesture`)   | Model training and scripting     |
| **Docker**              | Latest                       | Containerized FINN environment   |
| **FINN**                | Xilinx official Docker image | ONNX → FPGA flow                 |
| **Vivado**              | 2023.2                       | Hardware synthesis               |
| **PetaLinux**           | 2023.2 (optional)            | Embedded OS on UltraZed          |
| **PyTorch**             | 2.x                          | CNN model training               |
| **OpenCV**              | 4.x                          | Camera capture and preprocessing |
| **ONNX / ONNX Runtime** | 1.15                         | Model export and verification    |

---

## 🗂️ 3. Project Structure

```
gesture_fpga/
├── models/
│   └── simple_cnn.py
├── onnx/
│   └── gesture_float.onnx
├── scripts/
│   ├── cam_test.py
│   └── export_onnx.py
├── hw/
│   └── (Vivado projects)
├── finn_build/
│   └── (FINN-generated artifacts)
├── doc/
│   ├── README_setup.md
│   ├── board_notes.md
│   └── setup_logs/
└── requirements.txt
```

---

## ⚙️ 4. Installation and Configuration

### 4.1 Python Environment

```bash
conda create -n gesture python=3.10 -y
conda activate gesture
pip install torch torchvision torchaudio onnx onnxruntime onnxscript onnx-simplifier numpy opencv-python matplotlib
```

### 4.2 FINN (Docker)

Define Docker Arguments (on Host): Set the environment variable that mounts your project folder into the container.

```bash
export DOCKER_ARGS="-v ~/proj/gesture_fpga:/workspace"
```

Launch Container (on Host): Run the interactive shell from the cloned FINN repository(~finn).

```bash
cd ~/finn
./run-docker.sh bash
```

Verification (Inside Container): Once the command prompt changes, you are inside Docker.

```bash
cd /workspace
python - <<'PY'
import finn.core as f
print('✅ FINN import OK')
PY
```

### 4.3 Vivado / Vitis

The installation is highly dependent on the system environment setup.

Install Tools: Install Vitis Unified Software Platform (2023.2), ensuring the UltraScale+ device family support is selected.
Note: On Arch Linux, resolve dependencies like libtinfo.so.5 (via yay -S ncurses5-compat-libs).
Ensure you have the Board Files (BSP) installed for the UltraZed-EG if needed to activate the "Next" button in the Vivado board selection wizard.

Source Environment: The environment must be sourced to make the binaries available (this should be added to your ~/.bashrc permanently):

```bash
source /opt/Xilinx/Vitis/2023.2/settings64.sh
Verification: Test both the version and the graphical interface:
```

Verification:

```bash
vivado -version
vivado gui
```

### 4.4 System Utilities

```bash
sudo pacman -S git-lfs minicom screen v4l-utils
```

---

## 🔌 5. Hardware Tests

### 5.1 UART Access to UltraZed

1. Connect the board via USB-UART cable.
2. Check the device port: (usually /dev/ttyUSB0)

   ```bash
   dmesg | grep tty
   ```
3. Open serial console:

   ```bash
   screen /dev/ttyUSB0 115200
   ```
4. Power on the board and observe the boot log.
5. Save the output to `doc/setup_logs/console_boot_log.txt`.

---

### 5.2 Camera Test on Host

File: `scripts/cam_test.py`

```python
import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit(1)
ret, frame = cap.read()
print("Frame OK:", ret, "Shape:", None if not ret else frame.shape)
cv2.imshow("test", frame)
cv2.waitKey(2000)
cap.release()
cv2.destroyAllWindows()
```

Run:

```bash
python scripts/cam_test.py
```

---

## 🧩 6. Minimal AI Test

### 6.1 PyTorch Model

File: `models/simple_cnn.py`

```python
import torch, torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,128), nn.ReLU(),
            nn.Linear(128,n_classes)
        )
    def forward(self,x): return self.net(x)
```

### 6.2 ONNX Export

File: `scripts/export_onnx.py`

```python
import torch, onnx
from models.simple_cnn import SimpleCNN
model = SimpleCNN(n_classes=4).eval()
dummy = torch.randn(1,3,64,64)
torch.onnx.export(model, dummy, "onnx/gesture_float.onnx", opset_version=18,
                  input_names=['input'], output_names=['output']) # opset_version=11 provoke runtime errors
onnx.checker.check_model(onnx.load("onnx/gesture_float.onnx"))
print("✅ ONNX export and validation successful.")
```

Run:

```bash
python scripts/export_onnx.py
```

---

## 🧱 7. FINN Verification (Docker)

Inside the container:

```bash
cd /workspace
python - <<'PY'
import onnx
from finn.util.visualization import showInNetron
m = onnx.load("onnx/gesture_float.onnx")
print("ONNX loaded OK. Ready for FINN flow.")
PY
```

---

## 🧾 8. Logs and Documentation

All setup logs should be stored under `doc/setup_logs/`:

* `vivado_version.txt`
* `docker_finn_log.txt`
* `console_boot_log.txt`
* `camera_test_output.txt`
* `onnx_check.txt`

---

## ✅ 9. Setup Validation Checklist

| Test                       | Expected Result                          | Status |
| -------------------------- | ---------------------------------------- | ------ |
| Python installation        | `python --version` → 3.10                | ☐      |
| ONNX export                | “ONNX export and validation successful.” | ☐      |
| UART access                | Boot log visible on console              | ☐      |
| Camera test                | Frame successfully displayed             | ☐      |
| FINN import                | “FINN import OK”                         | ☐      |
| Vivado check               | Valid version displayed                  | ☐      |
| (Optional) PetaLinux build | `petalinux-build --version` works        | ☐      |

---

## 💡 10. Notes and Recommendations

* Docker is the most stable way to run FINN.
* Start with a small gesture dataset (e.g., Sign Language MNIST, ASL Alphabet).
* If MIPI camera support is complex, use a USB webcam first.
* Always document your board setup (BSP version, device tree, SD image, jumpers) in `doc/board_notes.md`.

---

**Author:** Etienne Bertin
**Date:** October 2025
**Project:** FPGA Hardware Accelerator for Gesture Recognition
**Institution:** —
**Supervisor:** —

---
