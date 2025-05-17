<template>
  <div class="chat-container">
    <!-- 状态栏 -->
    <div class="status-bar">
      <div class="status-left">
        <button class="back-btn" @click="goBack">
          <i class="el-icon-arrow-left"></i>
          返回选择
        </button>
        <div class="model-info">
          <i class="el-icon-chat-line-round"></i>
          <span>{{ modelName }}</span>
        </div>
      </div>
      <div class="status-right">
        <span class="user-info">
          <i class="el-icon-user"></i>
          {{ username }}
        </span>
        <button class="logout-btn" @click="logout">
          <i class="el-icon-switch-button"></i>
          退出登录
        </button>
      </div>
    </div>

    <!-- 聊天区域 -->
    <div class="chat-area" ref="chatArea">
      <div class="welcome-message" v-if="messages.length === 0">
        <div class="welcome-icon">
          <i class="el-icon-chat-line-round"></i>
        </div>
        <h2>开始对话</h2>
        <p>您可以向AI助手询问任何问题，我会尽力为您解答</p>
      </div>

      <div v-for="(message, index) in messages" :key="index" 
           :class="['message', message.role]">
        <div class="message-header">
          <div class="avatar">
            <i :class="message.role === 'user' ? 'el-icon-user' : 'el-icon-chat-line-round'"></i>
          </div>
          <div class="sender">
            {{ message.role === 'user' ? username : modelName }}
          </div>
        </div>
        <div class="content">
          <div class="text" v-html="formatMessage(message.content)"></div>
          <div v-if="message.code" class="code-block">
            <div class="code-header">
              <span class="language">{{ message.code.language || 'Code' }}</span>
              <button class="copy-btn" @click="copyCode(message.code.content)">
                <i class="el-icon-document-copy"></i>
                复制代码
              </button>
            </div>
            <pre><code>{{ message.code.content }}</code></pre>
          </div>
        </div>
      </div>

      <div class="typing-indicator" v-if="loading">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
      </div>
    </div>

    <!-- 输入区域 -->
    <div class="input-area">
      <div class="input-wrapper">
        <el-input
          v-model="userInput"
          type="textarea"
          :rows="3"
          placeholder="输入您的问题..."
          resize="none"
          @keydown.enter.prevent="sendMessage"
        />
        <button 
          class="send-btn"
          :class="{ 'active': userInput.trim() }"
          @click="sendMessage"
          :disabled="loading || !userInput.trim()"
        >
          <i class="el-icon-position"></i>
        </button>
      </div>
      <div class="input-footer">
        <span class="hint">按 Enter 发送消息，Shift + Enter 换行</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useStore } from 'vuex'
import { ElMessage } from 'element-plus'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'

const router = useRouter()
const route = useRoute()
const store = useStore()
const chatArea = ref(null)
const userInput = ref('')
const loading = ref(false)
const messages = ref([])
const username = ref(store.state.user?.username || '用户')
const modelName = ref('')

const API_URL = '/api/spark-chat'
const MODEL = 'x1' // 讯飞星火X1模型
const uuid = localStorage.getItem('uuid') || '' // 假设uuid已存储

// 返回上一页
const goBack = () => {
  router.back()
}

// 退出登录
const logout = async () => {
  try {
    await store.dispatch('logout')
    router.push('/login')
  } catch (error) {
    ElMessage.error('退出失败')
  }
}

// 发送消息
const sendMessage = async () => {
  if (!userInput.value.trim()) return
  
  const userMessage = {
    role: 'user',
    content: userInput.value
  }
  messages.value.push(userMessage)
  
  loading.value = true
  try {
    // 组装历史消息，支持多轮对话
    const history = messages.value.map(m => ({ role: m.role, content: m.content }))
    // 构造请求体，适配X1新参数
    const requestBody = {
      model: MODEL,
      user: uuid,
      messages: history,
      temperature: 1.2, // X1推荐默认值
      top_p: 0.95,
      top_k: 6,
      presence_penalty: 2.01,
      frequency_penalty: 0.001,
      stream: false, // 如需流式可改true
      max_tokens: 2048,
      tools: [
        {
          type: 'web_search',
          web_search: {
            enable: true,
            search_mode: 'normal' // 可选 deep/normal
          }
        }
      ]
    }
    // 请求讯飞星火API
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    })
    const data = await response.json()
    if (data.code === 0 && data.choices && data.choices.length > 0) {
      messages.value.push({
        role: 'assistant',
        content: data.choices[0].message.content
      })
    } else {
      ElMessage.error(data.message || 'AI回复失败')
    }
  } catch (error) {
    ElMessage.error('发送消息失败')
  } finally {
    loading.value = false
    userInput.value = ''
    await nextTick()
    scrollToBottom()
  }
}

// 格式化消息内容（支持代码高亮）
const formatMessage = (content) => {
  // 检测是否包含代码块
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g
  return content.replace(codeBlockRegex, (match, lang, code) => {
    const language = lang || 'plaintext'
    const highlightedCode = hljs.highlight(code.trim(), { language }).value
    return `<pre><code class="hljs ${language}">${highlightedCode}</code></pre>`
  })
}

// 滚动到底部
const scrollToBottom = () => {
  if (chatArea.value) {
    chatArea.value.scrollTop = chatArea.value.scrollHeight
  }
}

// 复制代码到剪贴板
const copyCode = async (code) => {
  try {
    await navigator.clipboard.writeText(code)
    ElMessage.success('代码已复制到剪贴板')
  } catch (err) {
    ElMessage.error('复制失败')
  }
}

onMounted(() => {
  const modelId = route.params.modelId
  modelName.value = `模型 ${String.fromCharCode(64 + parseInt(modelId.replace('model', '')))}` // model1 -> A
  scrollToBottom()
})
</script>

<style scoped>
/* ====== Chat Pro 美化风格，参考 deepseek/ChatGPT ====== */
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(135deg, #f8fafc 0%, #e6f0fa 100%);
  font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;
}
.status-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 64px;
  background: rgba(255, 255, 255, 0.98);
  backdrop-filter: blur(10px);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 40px;
  box-shadow: 0 2px 8px rgba(22, 119, 255, 0.06);
  z-index: 100;
  border-bottom: 1.5px solid #e6eaf0;
}
.status-left {
  display: flex;
  align-items: center;
  gap: 24px;
}
.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: none;
  border-radius: 8px;
  font-size: 15px;
  cursor: pointer;
  background: #f3f7fd;
  color: #1677ff;
  font-weight: 500;
  transition: all 0.2s;
  box-shadow: 0 1px 2px rgba(22,119,255,0.03);
}
.back-btn:hover {
  background: #e6f4ff;
  color: #0958d9;
}
.model-info {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #262626;
  font-size: 16px;
  font-weight: 600;
}
.model-info i {
  color: #1677ff;
  font-size: 20px;
}
.status-right {
  display: flex;
  align-items: center;
  gap: 20px;
}
.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #595959;
  font-size: 15px;
}
.logout-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: none;
  border-radius: 8px;
  font-size: 15px;
  cursor: pointer;
  background: #fff1f0;
  color: #ff4d4f;
  font-weight: 500;
  transition: all 0.2s;
}
.logout-btn:hover {
  background: #ffeaea;
  color: #d4380d;
}
.chat-area {
  flex: 1;
  overflow-y: auto;
  padding: 84px 0 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 720px;
  margin: 0 auto;
}
.welcome-message {
  text-align: center;
  padding: 60px 0;
  color: #8c8c8c;
}
.welcome-icon {
  width: 72px;
  height: 72px;
  margin: 0 auto 24px;
  border-radius: 36px;
  background: linear-gradient(135deg, #e6f4ff 0%, #bae0ff 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(22,119,255,0.08);
}
.welcome-icon i {
  font-size: 36px;
  color: #1677ff;
}
.welcome-message h2 {
  font-size: 26px;
  font-weight: 700;
  color: #262626;
  margin: 0 0 12px;
}
.welcome-message p {
  font-size: 16px;
  margin: 0;
}
.message {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: 680px;
  margin: 0 auto;
  width: 100%;
  border-radius: 14px;
  box-shadow: 0 2px 8px rgba(22,119,255,0.04);
  background: #fff;
  padding: 18px 22px;
  transition: box-shadow 0.2s;
}
.message.user {
  background: linear-gradient(90deg, #e6f4ff 0%, #fff 100%);
  align-self: flex-end;
  box-shadow: 0 2px 8px rgba(22,119,255,0.08);
}
.message.assistant {
  background: linear-gradient(90deg, #fff 0%, #f8fafc 100%);
  align-self: flex-start;
}
.message-header {
  display: flex;
  align-items: center;
  gap: 12px;
}
.avatar {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e6f4ff;
  box-shadow: 0 1px 2px rgba(22,119,255,0.06);
}
.message.user .avatar {
  background: #1677ff;
}
.avatar i {
  font-size: 20px;
  color: #1677ff;
}
.message.user .avatar i {
  color: #fff;
}
.sender {
  font-size: 15px;
  color: #8c8c8c;
  font-weight: 500;
}
.content {
  font-size: 16px;
  line-height: 1.7;
  color: #262626;
  word-break: break-word;
  background: none;
  padding: 0;
  box-shadow: none;
}
.message.user .content {
  color: #0958d9;
}
.code-block {
  margin-top: 14px;
  background: #18181c;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(22,119,255,0.08);
}
.code-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.language {
  font-size: 13px;
  color: #8c8c8c;
}
.copy-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  background: transparent;
  color: #8c8c8c;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}
.copy-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: white;
}
.code-block pre {
  margin: 0;
  padding: 16px;
  background: none;
  color: #fff;
}
.code-block code {
  font-family: 'Fira Code', monospace;
  font-size: 14px;
  line-height: 1.5;
}
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 12px 16px;
  background: #fff;
  border-radius: 12px;
  width: fit-content;
  margin: 0 auto;
  box-shadow: 0 1px 2px rgba(22,119,255,0.04);
}
.dot {
  width: 8px;
  height: 8px;
  background: #1677ff;
  border-radius: 50%;
  animation: typing 1.4s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-4px); }
}
.input-area {
  padding: 20px 0 32px;
  background: transparent;
  border-top: none;
  box-shadow: none;
}
.input-wrapper {
  position: relative;
  max-width: 720px;
  margin: 0 auto;
}
.input-wrapper :deep(.el-textarea__inner) {
  padding: 14px 60px 14px 18px;
  border: 2px solid #e6eaf0;
  border-radius: 14px;
  font-size: 16px;
  line-height: 1.7;
  resize: none;
  transition: all 0.2s;
  background: #fff;
  box-shadow: 0 1px 2px rgba(22,119,255,0.03);
}
.input-wrapper :deep(.el-textarea__inner:focus) {
  border-color: #1677ff;
  box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.1);
}
.send-btn {
  position: absolute;
  right: 16px;
  bottom: 16px;
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 10px;
  background: #1677ff;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 20px;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(22,119,255,0.08);
}
.send-btn.active {
  background: #0958d9;
  color: #fff;
}
.send-btn.active:hover {
  background: #4096ff;
  transform: translateY(-1px);
}
.send-btn:disabled {
  background: #e6eaf0;
  color: #bfbfbf;
  cursor: not-allowed;
  transform: none;
}
.input-footer {
  max-width: 720px;
  margin: 8px auto 0;
  display: flex;
  justify-content: flex-end;
}
.hint {
  font-size: 13px;
  color: #8c8c8c;
}

/* 响应式适配 */
@media (max-width: 900px) {
  .chat-area, .input-wrapper, .input-footer {
    max-width: 100vw;
    padding-left: 8px;
    padding-right: 8px;
  }
  .status-bar {
    padding: 0 12px;
  }
}
</style> 