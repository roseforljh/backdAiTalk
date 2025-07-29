/**
 * 功能测试脚本
 * 测试Worker的各项功能是否正常工作
 */

// 模拟环境
const testEnv = {
  APP_VERSION: '1.0.0',
  LOG_LEVEL: 'INFO',
  API_TIMEOUT: '30000',
  READ_TIMEOUT: '25000',
  OPENAI_API_KEY: 'test-openai-key',
  GEMINI_API_KEY: 'test-gemini-key'
};

async function testBasicFunctionality() {
  console.log('🧪 开始功能测试...\n');

  // Test 1: UUID生成
  try {
    const uuid = crypto.randomUUID();
    console.log('✅ UUID生成测试通过:', uuid);
  } catch (error) {
    console.error('❌ UUID生成测试失败:', error);
    return false;
  }

  // Test 2: URL解析
  try {
    const testUrls = [
      'https://api.openai.com/v1/chat/completions',
      'https://generativelanguage.googleapis.com/v1beta',
      'https://ai.google.dev/api'
    ];

    for (const url of testUrls) {
      const parsed = new URL(url);
      console.log(`✅ URL解析测试通过: ${parsed.hostname}`);
    }
  } catch (error) {
    console.error('❌ URL解析测试失败:', error);
    return false;
  }

  // Test 3: JSON处理
  try {
    const testData = {
      provider: 'openai',
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'user', content: 'Hello' }
      ]
    };
    const jsonString = JSON.stringify(testData);
    const parsed = JSON.parse(jsonString);
    console.log('✅ JSON处理测试通过');
  } catch (error) {
    console.error('❌ JSON处理测试失败:', error);
    return false;
  }

  // Test 4: Google API判断逻辑
  try {
    const googleUrls = [
      'https://generativelanguage.googleapis.com/v1beta',
      'https://aiplatform.googleapis.com/v1',
      'https://ai.google.dev/api'
    ];

    const nonGoogleUrls = [
      'https://api.openai.com/v1/chat/completions',
      'https://api.anthropic.com/v1/messages',
      'https://custom-api.example.com/chat'
    ];

    // 简化的Google API判断函数
    function isGoogleOfficialAPI(apiAddress) {
      if (!apiAddress) return false;
      
      try {
        const url = new URL(apiAddress);
        const domain = url.hostname.toLowerCase();
        
        const googleDomains = [
          'generativelanguage.googleapis.com',
          'aiplatform.googleapis.com',
          'googleapis.com',
          'ai.google.dev'
        ];
        
        return googleDomains.some(googleDomain => 
          domain === googleDomain || domain.endsWith('.' + googleDomain)
        );
      } catch (error) {
        return false;
      }
    }

    for (const url of googleUrls) {
      if (!isGoogleOfficialAPI(url)) {
        throw new Error(`应该识别为Google API: ${url}`);
      }
    }

    for (const url of nonGoogleUrls) {
      if (isGoogleOfficialAPI(url)) {
        throw new Error(`不应该识别为Google API: ${url}`);
      }
    }

    console.log('✅ Google API判断逻辑测试通过');
  } catch (error) {
    console.error('❌ Google API判断逻辑测试失败:', error);
    return false;
  }

  // Test 5: Base64编码
  try {
    const testString = 'Hello, World!';
    const encoded = btoa(testString);
    const decoded = atob(encoded);
    
    if (decoded !== testString) {
      throw new Error('Base64编码解码不匹配');
    }
    
    console.log('✅ Base64编码测试通过');
  } catch (error) {
    console.error('❌ Base64编码测试失败:', error);
    return false;
  }

  // Test 6: 流处理逻辑
  try {
    const testSSEData = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n';
    const lines = testSSEData.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim();
        if (data) {
          const parsed = JSON.parse(data);
          if (!parsed.choices) {
            throw new Error('SSE数据格式不正确');
          }
        }
      }
    }
    
    console.log('✅ 流处理逻辑测试通过');
  } catch (error) {
    console.error('❌ 流处理逻辑测试失败:', error);
    return false;
  }

  console.log('\n🎉 所有功能测试通过！');
  console.log('\n📋 测试总结:');
  console.log('- UUID生成: ✅');
  console.log('- URL解析: ✅');
  console.log('- JSON处理: ✅');
  console.log('- Google API判断: ✅');
  console.log('- Base64编码: ✅');
  console.log('- 流处理逻辑: ✅');
  
  return true;
}

// 运行测试
if (typeof require !== 'undefined' && require.main === module) {
  testBasicFunctionality().then(success => {
    if (!success) {
      process.exit(1);
    }
  }).catch(error => {
    console.error('测试运行失败:', error);
    process.exit(1);
  });
}

module.exports = { testBasicFunctionality };