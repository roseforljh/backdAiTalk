#!/bin/bash

# EzTalk Proxy Cloudflare Workers 部署脚本

set -e

echo "🚀 开始部署 EzTalk Proxy 到 Cloudflare Workers..."

# 检查是否安装了必要的工具
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: npm 未安装"
    exit 1
fi

if ! command -v wrangler &> /dev/null; then
    echo "❌ 错误: wrangler CLI 未安装"
    echo "请运行: npm install -g wrangler"
    exit 1
fi

# 安装依赖
echo "📦 安装依赖..."
npm install

# 构建项目
echo "🔨 构建项目..."
npm run build

# 检查是否已登录 Cloudflare
if ! wrangler whoami &> /dev/null; then
    echo "🔐 请先登录 Cloudflare:"
    wrangler login
fi

# 部署
echo "🚀 部署到 Cloudflare Workers..."
if [ "$1" = "staging" ]; then
    echo "部署到测试环境..."
    npm run deploy:staging
else
    echo "部署到生产环境..."
    npm run deploy
fi

echo "✅ 部署完成!"
echo ""
echo "📋 接下来的步骤:"
echo "1. 在 Cloudflare Dashboard 中设置环境变量:"
echo "   - OPENAI_API_KEY"
echo "   - GEMINI_API_KEY"
echo "   - GOOGLE_APPLICATION_CREDENTIALS (可选)"
echo ""
echo "2. 测试部署:"
echo "   curl https://your-worker.your-subdomain.workers.dev/health"
echo ""
echo "3. 查看日志:"
echo "   wrangler tail"