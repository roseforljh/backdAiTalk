# 🚀 快速部署指南

## 前置要求

1. **安装 Node.js** (版本 18+)
2. **安装 Wrangler CLI**:
   ```bash
   npm install -g wrangler
   ```
3. **Cloudflare 账号** 和 **API Token**

## 5分钟部署步骤

### 1. 登录 Cloudflare
```bash
wrangler login
```

### 2. 修改配置
编辑 `wrangler.toml`，将 `name` 改为你的 Worker 名称：
```toml
name = "backend-worker"  # 已经设置好了
```

### 3. 安装依赖并构建
```bash
npm install
npm run build
```

### 4. 部署
```bash
npm run deploy
```

### 5. 设置环境变量
在 Cloudflare Dashboard 中设置：
- `OPENAI_API_KEY` - 你的 OpenAI API 密钥
- `GEMINI_API_KEY` - 你的 Gemini API 密钥
- `GOOGLE_API_KEY` - Google API 密钥（可选，用于Web搜索）
- `GOOGLE_CSE_ID` - Google 自定义搜索引擎ID（可选，用于Web搜索）

或使用命令行：
```bash
wrangler secret put OPENAI_API_KEY
wrangler secret put GEMINI_API_KEY
wrangler secret put GOOGLE_API_KEY
wrangler secret put GOOGLE_CSE_ID
```

### 6. 测试部署
```bash
curl https://backend-worker.your-subdomain.workers.dev/health
```

## 🎉 完成！

你的 EzTalk Proxy 现在已经运行在 Cloudflare Workers 上了！

### API 端点
- **健康检查**: `GET /health`
- **主聊天接口**: `POST /chat` (与backend-docker兼容)
- **OpenAI 兼容**: `POST /v1/chat/completions`
- **备用聊天接口**: `POST /api/v1/chat`

### 示例请求

**主聊天接口（Form Data，与backend-docker兼容）**:
```bash
curl -X POST https://backend-worker.your-subdomain.workers.dev/chat \
  -F 'chat_request_json={
    "provider": "openai",
    "model": "gpt-3.5-turbo", 
    "api_key": "your-api-key",
    "messages": [{"role": "user", "type": "simple_text_message", "content": "Hello!"}]
  }'
```

**OpenAI兼容接口（JSON）**:
```bash
curl -X POST https://backend-worker.your-subdomain.workers.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 故障排除

### 常见问题
1. **部署失败**: 检查 `wrangler.toml` 中的 Worker 名称是否唯一
2. **API 错误**: 确保在 Cloudflare Dashboard 中正确设置了环境变量
3. **CORS 问题**: Worker 已配置允许所有来源，无需额外设置

### 查看日志
```bash
wrangler tail
```

### 本地开发
```bash
npm run dev
```

需要帮助？查看 `README-WORKER.md` 获取详细文档。