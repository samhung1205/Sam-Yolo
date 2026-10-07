# 專案維護與分支整理

本專案以 `exp-v8.3.242` 的客製提交 `19f8350950fc6474ef8415f0c4ac244e7d10c4ee` 為基準。舊官方 `main` 快照是 `572d381deb25e48733cedce27073c593af95cb7b`。

2026-10-07 整理：先保存完整 Git bundle 備份，再從客製分支建立 `sam-main`、移除官方自動化並加入專案檢查；確認後讓客製版本成為 `main`。客製模型核心、資料標註與實驗結果保留。

舊官方工作流程會每天重跑官方測試，另有 Dependabot 每日更新；它們不適合本專案。目前不設定自動版本升級，也不自動發布 PyPI 或 Docker 映像。安全性提醒與版本更新是不同功能。

## 既有電腦上的 checkout

GitHub 的整理不會修改你電腦上原有專案夾。請先保存未提交工作及未追蹤資料。原本本機 `main` 可能仍是官方分支，不要直接 pull 並合併它。

最容易避免混用的方法是將整理好的專案 clone 到新目錄，驗證後再處理舊工作目錄：

```bash
git clone https://github.com/samhung1205/Sam-Yolo.git Sam-Yolo-project
```

將資料、權重及可工作的環境設定接到新目錄；不要刪除尚未備份的舊目錄。

若要繼續使用既有 checkout，可先 fetch，再從遠端主分支建立一個新的本機分支，保留原本本機分支：

```bash
git fetch origin --prune
git switch -c sam-project --track origin/main
```

前提是目前工作已保存且 `sam-project` 尚不存在。

## 官方新版實驗

另用不同目錄與虛擬環境 clone 官方專案。需要把新版功能帶回本專案時，建立升級分支、確認客製模型及既有權重相容性後再合併，不自動跟隨官方 main。

## 待補齊的重現資訊

- 實際可工作的 Python、Torch、torchvision、CUDA、MMCV、mmengine、timm、einops 版本與環境鎖定檔。
- 資料圖片與權重的保存位置、取得方式及版本。
- 可攜的資料集路徑與訓練參數。
- YOLO26 實驗入口的完整模型設定。
- 在原訓練環境執行自訂模型初始化、前向推論和小規模訓練驗證。

## 備份還原

本次整理前的 Git 歷史保存於 `Sam-Yolo-before-migration-2026-10-07.bundle`；備份不包含 Git 未追蹤的資料圖片、權重或本機未提交修改。

在一個 Git 儲存庫中可先檢查備份，再將原分支恢復到新名稱：

```bash
git bundle verify /path/to/Sam-Yolo-before-migration-2026-10-07.bundle
git fetch /path/to/Sam-Yolo-before-migration-2026-10-07.bundle refs/heads/main:refs/heads/recover-upstream
git fetch /path/to/Sam-Yolo-before-migration-2026-10-07.bundle refs/remotes/origin/exp-v8.3.242:refs/heads/recover-experiment
```

這些還原指令會建立恢復分支，不覆寫目前主分支。
