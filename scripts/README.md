# 本地 GitHub Workflows 測試指南

## 🚀 快速開始

### 1. 快速驗證 (推薦第一步)
```bash
# 檢查所有 workflow 配置是否正確
./scripts/quick-validate.sh
```

### 2. 完整本地測試
```bash
# 測試所有 workflows
./scripts/test-workflows-locally.sh all

# 或分別測試
./scripts/test-workflows-locally.sh main        # 主要測試流程
./scripts/test-workflows-locally.sh docker      # Docker 相關測試
./scripts/test-workflows-locally.sh performance # 性能測試
./scripts/test-workflows-locally.sh security    # 安全掃描
```

## 🛠️ 其他本地測試工具

### Act - GitHub Actions 本地運行器
```bash
# 安裝 act
brew install act  # macOS
# 或
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# 運行特定 job
act -j test                    # 主要測試
act -j docker-build           # Docker 構建
act -j performance-baseline   # 性能基準測試

# 列出所有可用 jobs
act -l

# 模擬不同事件
act push                       # 模擬 push
act pull_request              # 模擬 PR
```

### Nektos/act 進階用法
```bash
# 使用特定 Docker 映像
act -P ubuntu-latest=catthehacker/ubuntu:act-latest

# 使用本地 secrets
act --secret-file .secrets

# 只運行特定 workflow
act -W .github/workflows/main.yml

# 乾運行 (不實際執行)
act --dry-run
```

## 📋 測試檢查清單

### 推送前必檢項目
- [ ] `./scripts/quick-validate.sh` 通過
- [ ] YAML 語法正確
- [ ] Docker 能正常構建
- [ ] Python 代碼無語法錯誤
- [ ] 服務端口可用

### 深度測試項目
- [ ] 所有測試通過 (`./scripts/test-workflows-locally.sh main`)
- [ ] Docker 映像能正常運行
- [ ] 性能基準測試符合預期
- [ ] 安全掃描無嚴重問題

## ⚡ 常見問題解決

### 問題 1: Docker 服務無法啟動
```bash
# 檢查 Docker 狀態
docker info

# 清理舊容器
docker stop test-redis test-qdrant || true
docker rm test-redis test-qdrant || true

# 重新啟動 Docker Desktop (macOS)
```

### 問題 2: Python 導入錯誤
```bash
# 確保虛擬環境正確設置
source .venv/bin/activate
uv pip install -e .

# 檢查 PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 問題 3: 端口被占用
```bash
# 檢查端口使用情況
lsof -i :6379  # Redis
lsof -i :6333  # Qdrant

# 停止占用端口的進程
kill -9 <PID>
```

### 問題 4: Act 運行錯誤
```bash
# 更新 act 到最新版本
brew upgrade act

# 使用更兼容的 Docker 映像
act -P ubuntu-latest=catthehacker/ubuntu:act-20.04

# 檢查 Docker 資源
docker system df
docker system prune  # 清理空間
```

## 📁 腳本文件說明

- `quick-validate.sh` - 快速驗證 workflows 配置
- `test-workflows-locally.sh` - 完整本地測試套件
- `README.md` - 本使用指南

## 🔧 環境要求

### 必需工具
- Docker & Docker Compose
- Python 3.10+
- uv (Python 包管理器)

### 可選工具
- act (GitHub Actions 本地運行)
- bandit (安全掃描)
- safety (依賴安全檢查)

## 💡 最佳實踐

1. **推送前先驗證**: 總是先運行 `quick-validate.sh`
2. **分步測試**: 從簡單到複雜逐步測試
3. **清理環境**: 測試後清理 Docker 容器和臨時文件
4. **監控資源**: 注意 Docker 和系統資源使用情況
5. **保持更新**: 定期更新測試工具和依賴

## 🚨 注意事項

- 本地測試環境可能與 GitHub Actions 略有差異
- 某些功能 (如 GitHub secrets) 在本地無法完全模擬
- 網絡和權限設置可能導致不同的結果
- 建議重要變更仍在 GitHub 上進行最終驗證
