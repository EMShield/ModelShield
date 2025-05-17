const axios = require('axios');

const API_BASE_URL = 'http://localhost:3111/api';
const TEST_UUID = 'test-uuid-123';
const TEST_SYSTEM = 'windows';

async function testEndpoints() {
    try {
        // 测试服务器状态
        const rootResponse = await axios.get('http://localhost:3111/');
        console.log('根路由测试:', rootResponse.data);

        // 测试系统选择
        const systemResponse = await axios.post(`${API_BASE_URL}/main-model/system-select`, {
            system: TEST_SYSTEM
        });
        console.log('系统选择测试:', systemResponse.data);

        // 测试UUID上传
        const uuidResponse = await axios.post(`${API_BASE_URL}/main-model/upload-uuid`, {
            system: TEST_SYSTEM,
            uuid: TEST_UUID
        });
        console.log('UUID上传测试:', uuidResponse.data);

        // 测试密钥生成
        const keyResponse = await axios.post(`${API_BASE_URL}/main-model/generate-key`, {
            uuid: TEST_UUID
        });
        console.log('密钥生成测试:', keyResponse.data);

        // 测试身份验证
        const identityResponse = await axios.post(`${API_BASE_URL}/main-model/verify-identity`, {
            uuid: TEST_UUID,
            key: 'test-key'
        });
        console.log('身份验证测试:', identityResponse.data);

        // 测试模型选择
        const modelResponse = await axios.post(`${API_BASE_URL}/main-model/select-model`, {
            uuid: TEST_UUID,
            modelIndex: 0
        });
        console.log('模型选择测试:', modelResponse.data);

    } catch (error) {
        console.error('测试失败:', error.response ? error.response.data : error.message);
    }
}

testEndpoints(); 