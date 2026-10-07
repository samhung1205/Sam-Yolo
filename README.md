# Sam-Yolo

以 Ultralytics **8.3.242** 為基礎的客製 YOLO 船舶辨識與模型實驗專案，包含自訂模組、偵測頭、模型設定、訓練腳本和實驗結果。

本專案獨立維護；官方更新由實驗需求決定，不自動同步官方分支。

## 專案內容

- `ultralytics/nn/AddModules/`：自訂神經網路模組。
- `ultralytics/nn/modules/`、`ultralytics/nn/tasks.py`：客製偵測頭與模型解析。
- `ultralytics/cfg/models/11/`：YOLO11 與客製模型設定。
- `ultralytics/cfg/datasets/`：資料集設定。
- `train11.py`、`val11.py`、`detect.py`：訓練、驗證與推論腳本。
- `runs_ship/`、`runs_ship-fix/`、`runs_merge/`：已保存的實驗參數與結果。
- `ultralytics/cutout/`：船舶去背與資料擴增工具。

## 安裝與實驗環境

請在專用虛擬環境安裝這份原始碼，以保留客製模組：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

上述指令只安裝 `pyproject.toml` 已宣告的基本依賴。自訂模組另外需要 `timm`、`einops`、`mmengine` 與具有對應算子支援的 `mmcv`。目前模組初始化會一併匯入 DyHead，因此需要相容的 MMCV 環境，即使這次實驗沒有選用 DyHead。這些套件與 Python、PyTorch、torchvision、CUDA 的相容版本，應以實際可工作的訓練環境為準；本專案尚未提供完整環境鎖定檔。

同一環境請避免執行 `pip install -U ultralytics`。要比較官方新版時，另外 clone 官方儲存庫並使用不同虛擬環境。

## 開始訓練前

1. 取得自己的資料集與模型權重。Git 只保存部分資料標註與實驗結果，沒有保存資料集圖片或訓練權重。
2. 調整資料 YAML 的 `path`，目前部分設定使用原訓練電腦的絕對路徑。
3. 確認訓練腳本的資料、模型、GPU 和輸出設定符合本次實驗。

`train11.py` 的 `yolo11n-asff.yaml` 名稱會由模型載入器解析為已存在的 `yolo11-asff.yaml`，並選擇 `n` 尺度。

`train26.py` 與 `val26.py` 是保留的實驗腳本；目前快照未包含所需的完整 YOLO26 設定，不能視為可直接執行的入口。

## CI

`Sam-Yolo CI` 在主分支 push、PR 和手動觸發時檢查 Python 語法、模型及資料 YAML 與 `pyproject.toml`。沒有每日排程、自動同步、套件發布或完整 GPU 訓練。

CI 成功代表原始碼與設定可解析；不代表自訂模型已通過訓練、推論或 CUDA/MMCV 相容性驗證。

本機執行相同檢查（Python 3.11 以上）：

```bash
python -m pip install "PyYAML==6.0.3"
python scripts/check_repository.py
```

## 分支與來源

- `main`：本專案的客製程式主分支。
- `exp-v8.3.242`：整理前的客製實驗基準，保留供比對。

分支整理說明請見 [維護說明](docs/PROJECT_MAINTENANCE.md)。

本專案源自 [Ultralytics](https://github.com/ultralytics/ultralytics)，保留原始碼中的授權與來源資訊。[原官方 README](docs/UPSTREAM_README.md) 僅供參考；本專案設定以本頁為準。

授權請見 [LICENSE](LICENSE)。
