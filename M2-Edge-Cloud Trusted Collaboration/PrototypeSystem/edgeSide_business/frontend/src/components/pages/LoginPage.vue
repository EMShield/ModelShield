<template>
  <div class="login-container">
    <div class="login-box">
      <div class="login-header">
        <h1>Cloud Server</h1>
        <p class="subtitle">登录到您的账户</p>
      </div>

      <div class="login-tabs">
        <button 
          :class="['tab-btn', { active: activeTab === 'login' }]"
          @click="activeTab = 'login'"
        >
          登录
        </button>
        <button 
          :class="['tab-btn', { active: activeTab === 'register' }]"
          @click="activeTab = 'register'"
        >
          注册
        </button>
      </div>

      <!-- 登录表单 -->
      <form v-if="activeTab === 'login'" @submit.prevent="handleLogin" class="login-form">
        <div class="form-group">
          <label for="uuid">UUID</label>
          <div class="input-group">
            <input
              id="uuid"
              v-model="loginForm.uuid"
              type="text"
              class="form-control"
              placeholder="请输入您的UUID"
              required
            />
            <button type="button" class="btn btn-secondary" @click="handleUploadUUID">
              上传UUID
            </button>
          </div>
        </div>

        <div class="form-group">
          <label for="key">密钥</label>
          <div class="input-group">
            <input
              id="key"
              v-model="loginForm.key"
              type="password"
              class="form-control"
              placeholder="请输入您的密钥"
              required
            />
            <button type="button" class="btn btn-secondary" @click="triggerKeyFileInput">
              上传密钥
            </button>
            <input
              ref="keyFileInput"
              type="file"
              accept=".key,.txt"
              style="display: none"
              @change="handleKeyFileChange"
            />
          </div>
        </div>

        <div class="form-group">
          <label>操作系统</label>
          <div class="os-selector">
            <div class="current-os">
              <span>当前系统: {{ currentOS }}</span>
              <button type="button" class="change-os-btn" @click="showOSSelector = true">
                更改
              </button>
            </div>
            <div v-if="showOSSelector" class="os-options">
              <button
                v-for="os in availableOS"
                :key="os"
                type="button"
                :class="['os-option', { active: selectedOS === os }]"
                @click="selectOS(os)"
              >
                {{ os }}
              </button>
            </div>
          </div>
        </div>

        <button type="submit" class="btn btn-primary btn-block">登录</button>
      </form>

      <!-- 注册表单 -->
      <form v-else @submit.prevent="handleRegister" class="login-form">
        <div class="form-group">
          <label for="register-uuid">UUID</label>
          <div class="input-group">
            <input
              id="register-uuid"
              v-model="registerForm.uuid"
              type="text"
              class="form-control"
              placeholder="请输入您的UUID"
              required
            />
            <button type="button" class="btn btn-secondary" @click="handleUploadUUID">
              上传UUID
            </button>
          </div>
        </div>

        <div class="form-group">
          <label>操作系统</label>
          <div class="os-selector">
            <div class="current-os">
              <span>当前系统: {{ currentOS }}</span>
              <button type="button" class="change-os-btn" @click="showOSSelector = true">
                更改
              </button>
            </div>
            <div v-if="showOSSelector" class="os-options">
              <button
                v-for="os in availableOS"
                :key="os"
                type="button"
                :class="['os-option', { active: selectedOS === os }]"
                @click="selectOS(os)"
              >
                {{ os }}
              </button>
            </div>
          </div>
        </div>

        <button type="submit" class="btn btn-primary btn-block">获取密钥</button>
      </form>

      <div v-if="error" class="error-message">
        {{ error }}
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useStore } from 'vuex'
import { ElMessage } from 'element-plus'

export default {
  name: 'LoginPage',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useStore()
    const activeTab = ref('login')
    const showOSSelector = ref(false)
    const error = ref('')
    const currentOS = ref('')
    const selectedOS = ref('')
    const availableOS = ['Windows', 'Linux', 'MacOS']
    const keyFileInput = ref(null)

    const loginForm = ref({
      uuid: '',
      key: ''
    })

    const registerForm = ref({
      uuid: ''
    })

    // 检测当前操作系统
    const detectOS = () => {
      const platform = navigator.platform.toLowerCase()
      if (platform.includes('win')) {
        currentOS.value = 'Windows'
      } else if (platform.includes('linux')) {
        currentOS.value = 'Linux'
      } else if (platform.includes('mac')) {
        currentOS.value = 'MacOS'
      } else {
        currentOS.value = 'Unknown'
      }
      selectedOS.value = currentOS.value
    }

    const selectOS = (os) => {
      selectedOS.value = os
      showOSSelector.value = false
    }

    const handleLogin = async () => {
      try {
        // 首先尝试登录
        const loginResult = await store.dispatch('login', {
          uuid: loginForm.value.uuid,
          key: loginForm.value.key,
          system: currentOS.value
        })

        if (loginResult.success) {
          ElMessage.success('登录成功')
          router.push('/model-selection')
          return
        }

        // 如果登录失败，尝试注册
        const registerResult = await store.dispatch('register', {
          uuid: loginForm.value.uuid,
          system: currentOS.value
        })

        if (registerResult.success) {
          // 使用新生成的 key 重试登录
          const retryLoginResult = await store.dispatch('login', {
            uuid: loginForm.value.uuid,
            key: registerResult.key,
            system: currentOS.value
          })

          if (retryLoginResult.success) {
            ElMessage.success('注册并登录成功')
            router.push('/model-selection')
            return
          }
        }

        // 如果都失败了，显示错误信息
        ElMessage.error(registerResult.message || '登录失败，请重试')
      } catch (error) {
        console.error('Login error:', error)
        ElMessage.error('登录过程中发生错误，请重试')
      }
    }

    // 处理上传UUID
    const handleUploadUUID = async () => {
      try {
        error.value = ''
        
        // 不再检查chrome.runtime，因为content.js已经在运行
        console.log('发送GET_UUID请求:', selectedOS.value)
        // 使用 window.postMessage 发送消息
        window.postMessage({
          type: 'GET_UUID',
          system: selectedOS.value
        }, '*')

        // 添加超时检查
        setTimeout(() => {
          if (!loginForm.value.uuid && !registerForm.value.uuid) {
            error.value = '获取UUID超时，请确保插件正常运行'
          }
        }, 5000) // 5秒超时

      } catch (err) {
        console.error('获取UUID错误:', err)
        error.value = err.message || '获取UUID失败'
      }
    }

    // 触发文件选择
    const triggerKeyFileInput = () => {
      keyFileInput.value && keyFileInput.value.click()
    }

    // 处理文件选择
    const handleKeyFileChange = async (event) => {
      const file = event.target.files[0]
      if (!file) return
      try {
        const text = await file.text()
        loginForm.value.key = text.trim()
      } catch (err) {
        error.value = err.message || '读取密钥文件失败'
      }
    }

    // 添加插件消息处理
    const handlePluginMessage = (event) => {
      // 只处理来自插件的消息
      if (event.data && event.data.type === 'UUID_RESULT') {
        console.log('收到UUID响应:', event.data)
        
        if (event.data.error) {
          error.value = event.data.error
        } else if (event.data.uuid) {
          // 根据当前标签页填充UUID
          if (activeTab.value === 'login') {
            loginForm.value.uuid = event.data.uuid
          } else {
            registerForm.value.uuid = event.data.uuid
          }
          error.value = '' // 清除任何错误消息
        }
      }
    }

    // 修改注册处理函数，添加下载密钥功能
    const handleRegister = async () => {
      try {
        error.value = ''
        const response = await store.dispatch('register', {
          uuid: registerForm.value.uuid,
          system: selectedOS.value
        })
        
        if (response.success && response.key) {
          // 创建并下载密钥文件
          const blob = new Blob([response.key], { type: 'text/plain' })
          const url = window.URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `${registerForm.value.uuid}.key`
          document.body.appendChild(a)
          a.click()
          window.URL.revokeObjectURL(url)
          document.body.removeChild(a)

          // 切换到登录页面并填充UUID
          activeTab.value = 'login'
          loginForm.value.uuid = registerForm.value.uuid
          
          // 显示成功消息
          ElMessage.success('注册成功！请使用下载的密钥文件登录。')
        } else {
          ElMessage.error(response.message || '注册失败，请稍后重试')
        }
      } catch (err) {
        console.error('注册错误:', err)
        ElMessage.error(err.response?.data?.message || '注册失败，请稍后重试')
      }
    }

    onMounted(() => {
      detectOS()
      // 添加插件消息监听
      window.addEventListener('message', handlePluginMessage)
      
      // 添加页面可见性变化监听
      document.addEventListener('visibilitychange', handleVisibilityChange)
    })

    onUnmounted(() => {
      // 移除插件消息监听
      window.removeEventListener('message', handlePluginMessage)
      // 移除页面可见性变化监听
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    })

    // 处理页面可见性变化
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        // 页面变为可见时，重新检测操作系统
        detectOS()
      }
    }

    return {
      activeTab,
      loginForm,
      registerForm,
      error,
      currentOS,
      selectedOS,
      availableOS,
      showOSSelector,
      selectOS,
      handleLogin,
      handleRegister,
      handleUploadUUID,
      keyFileInput,
      triggerKeyFileInput,
      handleKeyFileChange
    }
  }
}
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--color-canvas-subtle);
}

.login-box {
  width: 100%;
  max-width: 400px;
  padding: 32px;
  background-color: var(--color-canvas-default);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.login-header {
  text-align: center;
  margin-bottom: 24px;
}

.login-header h1 {
  font-size: 24px;
  color: var(--color-neutral-emphasis);
  margin-bottom: 8px;
}

.subtitle {
  color: var(--color-neutral-muted);
  font-size: 14px;
}

.login-tabs {
  display: flex;
  margin-bottom: 24px;
  border-bottom: 1px solid var(--color-border-default);
}

.tab-btn {
  flex: 1;
  padding: 12px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--color-neutral-muted);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tab-btn.active {
  color: var(--color-neutral-emphasis);
  border-bottom-color: #0366d6;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-neutral-emphasis);
}

.form-control {
  width: 100%;
  padding: 8px 12px;
  font-size: 14px;
  line-height: 20px;
  color: var(--color-neutral-emphasis);
  background-color: var(--color-canvas-default);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  transition: border-color 0.2s ease;
}

.form-control:focus {
  outline: none;
  border-color: #0366d6;
  box-shadow: 0 0 0 3px rgba(3, 102, 214, 0.3);
}

.btn-block {
  width: 100%;
  margin-top: 24px;
}

.error-message {
  margin-top: 16px;
  padding: 8px 12px;
  background-color: #ffebe9;
  border: 1px solid #ff8182;
  border-radius: 6px;
  color: #cf222e;
  font-size: 14px;
}

.os-selector {
  position: relative;
}

.current-os {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background-color: var(--color-canvas-default);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
}

.change-os-btn {
  padding: 4px 8px;
  font-size: 12px;
  color: #0366d6;
  background: none;
  border: 1px solid #0366d6;
  border-radius: 6px;
  cursor: pointer;
}

.os-options {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  margin-top: 4px;
  background-color: var(--color-canvas-default);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  z-index: 1;
}

.os-option {
  display: block;
  width: 100%;
  padding: 8px 12px;
  text-align: left;
  background: none;
  border: none;
  border-bottom: 1px solid var(--color-border-default);
  color: var(--color-neutral-emphasis);
  cursor: pointer;
}

.os-option:last-child {
  border-bottom: none;
}

.os-option:hover {
  background-color: var(--color-canvas-subtle);
}

.os-option.active {
  background-color: #0366d6;
  color: white;
}

.input-group {
  display: flex;
  gap: 8px;
}

.btn-secondary {
  background-color: var(--color-canvas-subtle);
  border: 1px solid var(--color-border-default);
  color: var(--color-neutral-emphasis);
  padding: 8px 12px;
  font-size: 14px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background-color: var(--color-border-default);
}
</style> 