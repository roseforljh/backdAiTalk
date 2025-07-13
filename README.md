---
<<<<<<< HEAD
title: EzTalk Proxy
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
# 关键：添加以下两行以确保网络访问和密钥可用
network: true
secrets:
  - GOOGLE_API_KEY
---

# EzTalk 代理服务

这是一个 FastAPI 后端，作为代理来处理与各种大型语言模型（如 Gemini）的通信。

## 部署到 Hugging Face Spaces

1.  **确保此 `README.md` 文件位于您的仓库根目录。**
2.  在您的 Space 的 "Settings" 页面中，找到 "Repository secrets" 部分。
3.  点击 "New secret"。
4.  **Secret name**: `GOOGLE_API_KEY`
5.  **Secret value**: 粘贴您的谷歌 AI Studio API 密钥。
6.  保存后，Hugging Face 会自动重新构建您的 Space。新的构建将拥有网络访问权限和 API 密钥。
=======
title: Backend
emoji: 🐨
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> ea649e55cb7a25db11c11edf8ef83bd87766c984
