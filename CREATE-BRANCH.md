# 🌿 创建 backend-worker 分支

如果你想为 backend-worker 创建一个独立的 Git 分支，按照以下步骤：

## 创建并切换到新分支

```bash
# 从当前位置创建新分支
git checkout -b backend-worker

# 或者如果你想从特定分支创建
git checkout -b backend-worker main
```

## 只保留 backend-worker 目录的内容

```bash
# 删除其他目录（保留 backend-worker）
git rm -r backend-docker
git rm -r KunTalkwithAi
git rm -r .kiro
# 删除其他不需要的文件...

# 将 backend-worker 目录的内容移到根目录
git mv backend-worker/* .
git mv backend-worker/.* . 2>/dev/null || true
git rm -r backend-worker

# 提交更改
git add .
git commit -m "Move backend-worker to root and clean up other directories"
```

## 推送分支到远程

```bash
git push -u origin backend-worker
```

## 在 Cloudflare Workers 中配置

1. 在 Cloudflare Dashboard 中：
   - 进入 Workers & Pages
   - 创建新的 Worker
   - 连接到你的 Git 仓库
   - 选择 `backend-worker` 分支
   - 设置构建命令：`npm run build`
   - 设置输出目录：`dist`

2. 设置环境变量：
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`

## 当前配置

- **Worker 名称**: `backend-worker`
- **兼容日期**: `2025-01-30`
- **主文件**: `src/index.js`
- **构建命令**: `npm run build`

现在你的配置已经准备好了！🚀