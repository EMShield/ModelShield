<template>
  <div class="model-download">
    <div class="status-bar">
      <div class="status-left">
        <button class="back-btn" @click="goBack">
          <i class="el-icon-arrow-left"></i>
          返回选择
        </button>
      </div>
      <div class="status-right">
        <span class="user-info">
          <i class="el-icon-user"></i>
          {{ username }}
        </span>
        <button class="logout-btn" @click="handleLogout">
          <i class="el-icon-switch-button"></i>
          退出登录
        </button>
      </div>
    </div>

    <div class="content-wrapper">
      <div class="page-header">
        <h1>下载模型文件</h1>
        <p class="subtitle">{{ modelName }} - 准备开始使用</p>
      </div>

      <div class="download-container">
        <div class="info-section">
          <div class="info-card">
            <div class="card-header">
              <i class="el-icon-info"></i>
              <h3>模型信息</h3>
            </div>
            <div class="info-grid">
              <div class="info-item">
                <div class="info-icon">
                  <i class="el-icon-version"></i>
                </div>
                <div class="info-content">
                  <span class="label">版本</span>
                  <span class="value">{{ modelInfo.version }}</span>
                </div>
              </div>
              <div class="info-item">
                <div class="info-icon">
                  <i class="el-icon-folder"></i>
                </div>
                <div class="info-content">
                  <span class="label">文件大小</span>
                  <span class="value">{{ modelInfo.totalSize }}</span>
                </div>
              </div>
              <div class="info-item">
                <div class="info-icon">
                  <i class="el-icon-time"></i>
                </div>
                <div class="info-content">
                  <span class="label">更新时间</span>
                  <span class="value">{{ modelInfo.updateTime }}</span>
                </div>
              </div>
            </div>
            <div class="features-list">
              <div class="feature-item" v-for="(feature, index) in modelInfo.features" :key="index">
                <i class="el-icon-check"></i>
                <span>{{ feature }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="files-section">
          <div class="section-header">
            <i class="el-icon-folder"></i>
            <h3>模型文件</h3>
          </div>
          <div class="files-list">
            <div v-for="file in coreFiles" 
                 :key="file.id" 
                 class="file-card"
                 :class="{ 'downloading': file.downloading, 'completed': file.downloaded }">
              <div class="file-header">
                <div class="file-info">
                  <div class="file-icon">
                    <i class="el-icon-document"></i>
                  </div>
                  <div class="file-details">
                    <h4>{{ file.name }}</h4>
                    <span class="file-type">{{ file.type }}</span>
                  </div>
                </div>
                <span class="file-size">{{ file.size }}</span>
              </div>
              <p class="file-description">{{ file.description }}</p>
              <div class="download-status">
                <template v-if="file.downloading">
                  <div class="progress-bar">
                    <div class="progress-track">
                      <div class="progress-fill" :style="{ width: file.progress + '%' }"></div>
                    </div>
                    <span class="progress-text">{{ file.progress }}%</span>
                  </div>
                </template>
                <button 
                  class="download-btn"
                  :class="{
                    'downloading': file.downloading,
                    'completed': file.downloaded
                  }"
                  @click="downloadFile(file)"
                  :disabled="file.downloading || file.downloaded || file.disabled"
                >
                  <i :class="file.downloaded ? 'el-icon-check' : 'el-icon-download'"></i>
                  {{ file.downloaded ? '已完成' : (file.downloading ? '下载中...' : '下载文件') }}
                </button>
              </div>
            </div>
          </div>
        </div>

        <div class="action-buttons">
          <button 
            class="btn btn-primary"
            :disabled="!allFilesDownloaded"
            @click="proceedToNext"
          >
            <i class="el-icon-chat-line-round"></i>
            开始对话
          </button>
          <p class="hint-text" v-if="!allFilesDownloaded">
            请下载所有必需的模型文件后继续
          </p>
        </div>
      </div>
    </div>

    <div v-if="showMountModal" class="mount-modal">
      <div class="mount-content">
        <h2>模型挂载中...</h2>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: mountProgress + '%' }"></div>
        </div>
        <span class="progress-text">{{ mountProgress }}%</span>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useStore } from 'vuex'
import axios from 'axios'

function base64Decode(str) {
  try {
    return decodeURIComponent(escape(window.atob(str)))
  } catch (e) {
    // 兼容非utf8
    return window.atob(str)
  }
}

export default {
  name: 'CoreDownloadPage',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useStore()
    const username = computed(() => store.state.user?.username || '用户')
    const modelName = ref('')
    const modelId = route.params.modelId
    const uuid = computed(() => store.state.user?.uuid || '')
    const modelIndex = ref(0)
    const realModelUrl = ref('')
    // 挂载进度弹窗相关
    const showMountModal = ref(true)
    const mountProgress = ref(0)
    let mountTimer = null
    
    const modelInfo = ref({
      version: 'v2.0.1',
      totalSize: '4.5GB',
      updateTime: '2024-03-15',
      features: [
        '支持离线运行',
        '自动增量更新',
        '数据安全加密',
        '兼容多种硬件'
      ]
    })

    const coreFilesRef = ref([
      {
        id: 'core1',
        name: '核心模型文件',
        description: '包含模型的基础架构和核心功能',
        size: '1.2GB',
        type: 'Model Core',
        downloading: false,
        downloaded: false,
        progress: 0,
        disabled: false,
        downloadUrl: '',
        filename: ''
      },
      {
        id: 'core2',
        name: '权重参数文件',
        description: '包含模型的训练权重和参数数据',
        size: '1.5GB',
        type: 'Weights',
        downloading: false,
        downloaded: false,
        progress: 0,
      disabled: false,
        downloadUrl: '',
        filename: ''
      }
    ])

    const allFilesDownloaded = computed(() => {
      return coreFilesRef.value.every(file => file.downloaded)
    })

    // 挂载进度轮询
    const pollMountStatus = async () => {
      mountProgress.value = 10
      mountTimer = setInterval(async () => {
        try {
          // 模拟进度条缓慢增长
          if (mountProgress.value < 90) {
            mountProgress.value += Math.floor(Math.random() * 10) + 5
            if (mountProgress.value > 90) mountProgress.value = 90
          }
          const res = await axios.get('/main-model/mount-status')
          if (res.data.success && res.data.mounted) {
            mountProgress.value = 100
            clearInterval(mountTimer)
            setTimeout(() => {
              showMountModal.value = false
            }, 500)
          }
        } catch (e) {
          // 失败时也继续轮询
        }
      }, 500)
    }

    // 下载核心文件（只调用一次后端接口，获取所有下载链接）
    const downloadAllFiles = async () => {
      try {
        // 调用后端统一接口
        const res = await axios.post('/main-model/download-all-core-files', {
          uuid: uuid.value,
          modelIndex: modelIndex.value
        })
        if (!res.data.success) throw new Error(res.data.message)
        // 拿到所有文件的下载信息
        const { coreFiles, secondPart } = res.data.files
        // 更新 coreFiles.value 的下载链接和文件名
        coreFilesRef.value[0].downloadUrl = coreFiles.pt.url
        coreFilesRef.value[0].filename = coreFiles.pt.filename
        coreFilesRef.value[1].downloadUrl = secondPart.pt.url
        coreFilesRef.value[1].filename = secondPart.pt.filename
        // 启用下载按钮
        coreFilesRef.value[0].disabled = false
        coreFilesRef.value[1].disabled = false
      } catch (e) {
        coreFilesRef.value[0].disabled = true
        coreFilesRef.value[1].disabled = true
        alert('获取下载链接失败：' + (e.response?.data?.message || e.message))
      }
    }

    // 下载单个文件
    const downloadFile = async (file) => {
      if (file.downloading || file.downloaded || file.disabled) return
      // 如果没有下载链接，先获取
      if (!file.downloadUrl) {
        await downloadAllFiles()
        if (!file.downloadUrl) return // 获取失败
      }
      file.downloading = true
      file.progress = 0
      try {
        await downloadBlob(file.downloadUrl, file.filename, file)
        file.downloaded = true
      } catch (error) {
        file.downloading = false
        alert('下载失败：' + (error.response?.data?.message || error.message))
      } finally {
        file.downloading = false
        file.progress = 100
      }
    }

    // 下载辅助函数
    const downloadBlob = async (url, filename, file) => {
        const response = await axios({
        url,
          baseURL: '', // 强制用绝对路径，防止拼接 /api/
          method: 'GET',
          responseType: 'blob',
          timeout: 0, // 不限制超时，适配大文件下载
          onDownloadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            file.progress = percentCompleted
          }
          }
        })
      const blobUrl = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
      link.href = blobUrl
      link.setAttribute('download', filename)
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      window.URL.revokeObjectURL(blobUrl)
    }

    // 跳转到对话页面前先卸载
    const proceedToNext = async () => {
      if (allFilesDownloaded.value) {
        try {
          await axios.post('/main-model/unmount')
        } catch (e) {
          // 卸载失败也允许跳转
        }
        router.push(`/chat/${modelId}`)
      }
    }

    const goBack = () => {
      router.push('/model-selection')
    }

    const handleLogout = () => {
      store.dispatch('logout')
      router.push('/login')
    }

    onMounted(async () => {
      const modelId = route.params.modelId
      modelName.value = `模型 ${String.fromCharCode(64 + parseInt(modelId.replace('model', '')) )}`
      modelIndex.value = parseInt(modelId.replace('model', ''))
      // 获取url.json并解码
      try {
        const urlRes = await axios.get('/data/url.json', { baseURL: '' })
        const base64url = urlRes.data[modelIndex.value]
        realModelUrl.value = base64Decode(base64url)
      } catch (e) {
        alert('获取模型URL失败：' + (e.response?.data?.message || e.message))
      }
      // 进入页面即弹出挂载进度条
      showMountModal.value = true
      mountProgress.value = 10
      pollMountStatus()
    })

    return {
      username,
      modelName,
      modelInfo,
      coreFiles: coreFilesRef,
      allFilesDownloaded,
      downloadFile,
      proceedToNext,
      goBack,
      handleLogout,
      showMountModal,
      mountProgress
    }
  }
}
</script>

<style scoped>
.model-download {
  min-height: 100vh;
  background: #f0f2f5;
  position: relative;
}

.model-download::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 280px;
  background: linear-gradient(135deg, #1677ff 0%, #0958d9 100%);
  z-index: 0;
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
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  z-index: 100;
}

.back-btn, .logout-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.back-btn {
  background: #f5f5f5;
  color: #595959;
}

.back-btn:hover {
  background: #e6f4ff;
  color: #1677ff;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #595959;
  font-size: 14px;
}

.logout-btn {
  background: transparent;
  color: #595959;
}

.logout-btn:hover {
  background: #fff1f0;
  color: #ff4d4f;
}

.content-wrapper {
  position: relative;
  max-width: 1200px;
  margin: 0 auto;
  padding: 104px 40px 40px;
  z-index: 1;
}

.page-header {
  text-align: center;
  margin-bottom: 64px;
  color: white;
}

.page-header h1 {
  font-size: 40px;
  font-weight: 600;
  margin-bottom: 16px;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.subtitle {
  font-size: 18px;
  opacity: 0.9;
}

.download-container {
  background: white;
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
  overflow: hidden;
}

.info-section {
  padding: 32px;
  border-bottom: 1px solid #f0f0f0;
}

.info-card {
  background: #fafafa;
  border-radius: 12px;
  padding: 28px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 28px;
}

.card-header i {
  font-size: 24px;
  color: #1677ff;
}

.card-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: #262626;
  margin: 0;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 28px;
  margin-bottom: 28px;
}

.info-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.info-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  background: linear-gradient(135deg, #e6f4ff 0%, #bae0ff 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #1677ff;
  font-size: 20px;
}

.info-content {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.info-item .label {
  color: #8c8c8c;
  font-size: 14px;
}

.info-item .value {
  color: #262626;
  font-size: 16px;
  font-weight: 500;
}

.features-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  padding-top: 24px;
  border-top: 1px solid #f0f0f0;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #595959;
  font-size: 14px;
}

.feature-item i {
  color: #52c41a;
  font-size: 16px;
}

.files-section {
  padding: 32px;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 28px;
}

.section-header i {
  font-size: 24px;
  color: #1677ff;
}

.section-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: #262626;
  margin: 0;
}

.files-list {
  display: grid;
  gap: 20px;
}

.file-card {
  background: #fafafa;
  border-radius: 12px;
  padding: 28px;
  border: 2px solid transparent;
  transition: all 0.3s ease;
}

.file-card:hover {
  border-color: #d9d9d9;
}

.file-card.downloading {
  border-color: #1677ff;
  background: #f0f7ff;
}

.file-card.completed {
  border-color: #52c41a;
  background: #f6ffed;
}

.file-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
}

.file-info {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.file-icon {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  background: linear-gradient(135deg, #e6f4ff 0%, #bae0ff 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #1677ff;
  font-size: 24px;
}

.file-card.downloading .file-icon {
  background: linear-gradient(135deg, #1677ff 0%, #4096ff 100%);
  color: white;
}

.file-card.completed .file-icon {
  background: linear-gradient(135deg, #52c41a 0%, #73d13d 100%);
  color: white;
}

.file-details h4 {
  font-size: 16px;
  font-weight: 500;
  color: #262626;
  margin: 0 0 6px;
}

.file-type {
  font-size: 13px;
  color: #8c8c8c;
  background: rgba(0, 0, 0, 0.04);
  padding: 2px 8px;
  border-radius: 4px;
}

.file-size {
  font-size: 14px;
  color: #595959;
  background: rgba(0, 0, 0, 0.04);
  padding: 4px 12px;
  border-radius: 6px;
}

.file-description {
  color: #595959;
  font-size: 14px;
  line-height: 1.6;
  margin: 0 0 24px;
}

.download-status {
  display: flex;
  align-items: center;
  gap: 16px;
}

.progress-bar {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress-track {
  flex: 1;
  height: 8px;
  background: #f0f0f0;
  border-radius: 6px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #1677ff 0%, #4096ff 100%);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 14px;
  color: #595959;
  min-width: 48px;
}

.download-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  border: 2px solid #d9d9d9;
  border-radius: 8px;
  background: white;
  color: #595959;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  min-width: 120px;
  justify-content: center;
}

.download-btn:hover:not(:disabled) {
  border-color: #1677ff;
  color: #1677ff;
  background: #f0f7ff;
}

.download-btn.downloading {
  background: #f0f7ff;
  border-color: #1677ff;
  color: #1677ff;
  cursor: not-allowed;
}

.download-btn.completed {
  background: #f6ffed;
  border-color: #52c41a;
  color: #52c41a;
}

.download-btn:disabled {
  background: #f5f5f5;
  border-color: #d9d9d9;
  color: #8c8c8c;
  cursor: not-allowed;
}

.action-buttons {
  padding: 32px;
  text-align: center;
  background: #fafafa;
  border-top: 1px solid #f0f0f0;
}

.btn {
  padding: 12px 32px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.btn-primary {
  background: linear-gradient(135deg, #1677ff 0%, #4096ff 100%);
  color: white;
  height: 52px;
  padding: 0 48px;
  font-weight: 500;
  font-size: 17px;
}

.btn-primary:hover:not(:disabled) {
  background: linear-gradient(135deg, #4096ff 0%, #1677ff 100%);
  transform: translateY(-1px);
}

.btn-primary:disabled {
  background: #bfbfbf;
  cursor: not-allowed;
  transform: none;
}

.btn i {
  font-size: 20px;
}

.hint-text {
  margin: 16px 0 0;
  color: #8c8c8c;
  font-size: 14px;
}

.mount-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.mount-content {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  max-width: 400px;
  text-align: center;
}

.mount-content h2 {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 20px;
}

.progress-bar {
  height: 20px;
  background: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 20px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #1677ff 0%, #4096ff 100%);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 18px;
  font-weight: 500;
  color: #595959;
}
</style> 