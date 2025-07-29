/**
 * 本地测试脚本
 * 用于在部署前测试基本功能
 */

// 模拟Cloudflare Workers环境
global.crypto = require('crypto').webcrypto;
global.fetch = require('node-fetch');

// 简单的测试
async function testBasicFunctionality() {
  console.log('🧪 开始基本功能测试...');
  
  try {
    // 测试UUID生成
    const uuid = crypto.randomUUID();
    console.log('✅ UUID生成测试通过:', uuid);
    
    // 测试URL解析
    const url = new URL('https://api.openai.com/v1/chat/completions');
    console.log('✅ URL解析测试通过:', url.hostname);
    
    // 测试JSON处理
    const testData = { test: 'data', number: 123 };
    const jsonString = JSON.stringify(testData);
    const parsed = JSON.parse(jsonString);
    console.log('✅ JSON处理测试通过:', parsed);
    
    console.log('🎉 所有基本功能测试通过！');
    
  } catch (error) {
    console.error('❌ 测试失败:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  testBasicFunctionality();
}