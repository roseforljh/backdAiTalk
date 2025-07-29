# EzTalk Proxy - Cloudflare Workers Version

这是EzTalk Proxy的Cloudflare Workers适配版本，可以部署在Cloudflare的边缘网络上。

## 功能特性

- ✅ OpenAI兼容API
- ✅ Gemini API支持
- ✅ 流式响应
- ✅ 多模态内容支持（图片、音频、视频）
- ✅ 文件上传处理
- ✅ CORS支持
- ✅ 错误处理和日志记录

## 部署步骤

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

在Cloudflare Dashboard中设置以下环境变量，或使用wrangler命令：

```bash
# OpenAI API密钥
wrangler secret put OPENAI_API_KEY

# Gemini API密钥
wrangler secret put GEMINI_API_KEY

# Google服务账号JSON（如果使用Google Cloud AI Platform）
wrangler secret put GOOGLE_APPLICATION_CREDENTIALS
```

### 3. 修改wrangler.toml

编辑`wrangler.toml`文件，设置你的Worker名称：

```toml
name = "your-eztalk-proxy-worker"
```

### 4. 构建项目

```bash
npm run build
```

### 5. 部署到Cloudflare Workers

```bash
# 部署到生产环境
npm run deploy

# 或部署到测试环境
npm run deploy:staging
```

### 6. 本地开发

```bash
npm run dev
```

## API端点

### 主要聊天端点
- `POST /chat` - 主要聊天接口（与backend-docker兼容）
- `POST /api/v1/chat` - 备用聊天接口
- `POST /v1/chat/completions` - OpenAI兼容接口
- `POST /api/gemini/*` - Gemini直接接口

### 健康检查
- `GET /health` - 服务健康状态

## 请求格式

### 标准聊天请求

```json
{
  "provider": "openai",
  "model": "gpt-4",
  "api_key": "your-api-key",
  "api_address": "https://api.openai.com/v1/chat/completions",
  "messages": [
    {
      "role": "user",
      "type": "simple_text_message",
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### 多模态请求

```json
{
  "provider": "gemini",
  "model": "gemini-pro-vision",
  "api_key": "your-gemini-api-key",
  "messages": [
    {
      "role": "user",
      "type": "parts_message",
      "parts": [
        {
          "type": "text_content",
          "text": "What's in this image?"
        },
        {
          "type": "inline_data_content",
          "mime_type": "image/jpeg",
          "base64_data": "base64-encoded-image-data"
        }
      ]
    }
  ]
}
```

## 支持的模型

### OpenAI兼容
- GPT-4系列
- GPT-3.5系列
- 其他OpenAI兼容的模型

### Gemini
- gemini-pro
- gemini-pro-vision
- gemini-1.5-pro
- gemini-1.5-flash

## 文件上传

支持通过multipart/form-data上传文件：

```javascript
const formData = new FormData();
formData.append('chat_request_json', JSON.stringify(chatRequest));
formData.append('file', fileBlob, 'image.jpg');

fetch('/api/v1/chat', {
  method: 'POST',
  body: formData
});
```

## 错误处理

所有错误响应都包含以下格式：

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "request_id": "uuid"
}
```

## 限制

- 文件上传大小限制：100MB（Cloudflare Workers限制）
- 请求超时：30秒
- 并发连接数：100

## 监控和日志

- 使用Cloudflare Dashboard查看Worker日志
- 支持不同日志级别：DEBUG, INFO, WARN, ERROR
- 通过环境变量`LOG_LEVEL`控制日志级别

## 故障排除

### 常见问题

1. **API密钥错误**
   - 确保在Cloudflare Dashboard中正确设置了环境变量
   - 检查API密钥是否有效

2. **CORS错误**
   - Worker已配置允许所有来源的CORS请求
   - 如需限制，修改`src/utils/cors.js`

3. **超时错误**
   - 检查上游API是否响应正常
   - 调整`API_TIMEOUT`环境变量

4. **文件上传失败**
   - 确保文件大小在限制范围内
   - 检查文件MIME类型是否支持

## 开发

### 项目结构

```
src/
├── index.js          # 主入口文件
├── router.js         # 路由处理
├── handlers/         # 请求处理器
│   ├── chat.js       # 聊天处理器
│   ├── openai.js     # OpenAI处理器
│   └── gemini.js     # Gemini处理器
└── utils/            # 工具函数
    ├── cors.js       # CORS处理
    ├── logger.js     # 日志工具
    └── http.js       # HTTP客户端
```

### 添加新功能

1. 在相应的处理器中添加新方法
2. 在`router.js`中添加新路由
3. 更新文档和测试

## 许可证

MIT License