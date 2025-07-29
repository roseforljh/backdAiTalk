@echo off
REM EzTalk Proxy Cloudflare Workers 部署脚本 (Windows版本)

echo 🚀 开始部署 EzTalk Proxy 到 Cloudflare Workers...

REM 检查是否安装了 npm
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 错误: npm 未安装
    pause
    exit /b 1
)

REM 检查是否安装了 wrangler
where wrangler >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 错误: wrangler CLI 未安装
    echo 请运行: npm install -g wrangler
    pause
    exit /b 1
)

REM 检查wrangler.toml是否存在
if not exist "wrangler.toml" (
    echo ❌ 错误: wrangler.toml 文件不存在
    pause
    exit /b 1
)

REM 运行基本测试
echo 🧪 运行基本测试...
node test-local.js
if %errorlevel% neq 0 (
    echo ❌ 基本测试失败
    pause
    exit /b 1
)

REM 安装依赖
echo 📦 安装依赖...
npm install
if %errorlevel% neq 0 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

REM 构建项目
echo 🔨 构建项目...
npm run build
if %errorlevel% neq 0 (
    echo ❌ 构建失败
    pause
    exit /b 1
)

REM 检查构建结果
if not exist "dist\index.js" (
    echo ❌ 错误: 构建失败，dist\index.js 不存在
    pause
    exit /b 1
)

echo ✅ 构建成功

REM 检查是否已登录 Cloudflare
wrangler whoami >nul 2>nul
if %errorlevel% neq 0 (
    echo 🔐 请先登录 Cloudflare:
    wrangler login
)

REM 部署
echo 🚀 部署到 Cloudflare Workers...
if "%1"=="staging" (
    echo 部署到测试环境...
    npm run deploy:staging
) else (
    echo 部署到生产环境...
    npm run deploy
)

if %errorlevel% neq 0 (
    echo ❌ 部署失败
    pause
    exit /b 1
)

echo.
echo ✅ 部署完成!
echo.
echo 📋 接下来的步骤:
echo 1. 在 Cloudflare Dashboard 中设置环境变量:
echo    - OPENAI_API_KEY
echo    - GEMINI_API_KEY
echo.
echo 2. 测试部署:
echo    curl https://your-worker.your-subdomain.workers.dev/health
echo.
echo 3. 查看日志:
echo    wrangler tail
echo.
echo 4. 测试聊天功能:
echo    curl -X POST https://your-worker.your-subdomain.workers.dev/v1/chat/completions ^
echo    -H "Content-Type: application/json" ^
echo    -d "{\"model\":\"gpt-3.5-turbo\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"

pause