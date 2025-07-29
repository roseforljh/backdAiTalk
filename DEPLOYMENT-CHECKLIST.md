# 🚀 部署检查清单

在部署到 Cloudflare Workers 之前，请确保完成以下步骤：

## ✅ 前置条件检查

- [ ] 已安装 Node.js (版本 18+)
- [ ] 已安装 Wrangler CLI: `npm install -g wrangler`
- [ ] 已有 Cloudflare 账号
- [ ] 已登录 Wrangler: `wrangler login`

## ✅ 配置检查

- [ ] 修改了 `wrangler.toml` 中的 Worker 名称
- [ ] 准备好了 API 密钥:
  - [ ] OpenAI API Key
  - [ ] Gemini API Key (如果使用)

## ✅ 代码检查

- [ ] 运行了基本测试: `node test-local.js`
- [ ] 安装了依赖: `npm install`
- [ ] 构建成功: `npm run build`
- [ ] 检查 `dist/index.js` 文件存在

## ✅ 部署步骤

1. **部署 Worker**:
   ```bash
   npm run deploy
   ```

2. **设置环境变量**:
   ```bash
   wrangler secret put OPENAI_API_KEY
   wrangler secret put GEMINI_API_KEY
   ```

3. **测试部署**:
   ```bash
   curl https://backend-worker.your-subdomain.workers.dev/health
   ```

## ✅ 部署后验证

- [ ] 健康检查端点响应正常
- [ ] OpenAI 兼容端点工作正常
- [ ] 日志显示正常: `wrangler tail`

## 🔧 故障排除

### 常见问题及解决方案

1. **Worker 名称冲突**
   - 修改 `wrangler.toml` 中的 `name` 字段

2. **构建失败**
   - 检查 Node.js 版本
   - 删除 `node_modules` 重新安装: `rm -rf node_modules && npm install`

3. **API 调用失败**
   - 确认环境变量设置正确
   - 检查 API 密钥有效性

4. **CORS 错误**
   - Worker 已配置允许所有来源，通常不会有 CORS 问题

### 有用的命令

```bash
# 查看 Worker 状态
wrangler status

# 查看实时日志
wrangler tail

# 本地开发
npm run dev

# 删除 Worker
wrangler delete

# 查看环境变量
wrangler secret list
```

## 📞 获取帮助

- 查看详细文档: `README-WORKER.md`
- 快速部署指南: `QUICK-DEPLOY.md`
- Cloudflare Workers 文档: https://developers.cloudflare.com/workers/

---

**准备好了吗？运行 `./deploy.sh` 开始部署！** 🚀