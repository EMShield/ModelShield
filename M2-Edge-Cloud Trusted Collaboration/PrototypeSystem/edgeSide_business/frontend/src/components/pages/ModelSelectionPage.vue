<template>
  <div class="model-selection">
    <div class="status-bar">
      <div class="status-left">
        <button class="back-btn" @click="goBack">
          <i class="el-icon-arrow-left"></i>
          返回
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
        <h1>选择模型</h1>
        <p class="subtitle">选择最适合您需求的AI模型</p>
      </div>

      <div class="model-grid">
        <div 
          v-for="model in models" 
          :key="model.id"
          class="model-card"
          :class="{ 'selected': selectedModel === model.id }"
          @click="selectModel(model.id)"
        >
          <div class="model-header">
            <div class="model-icon">
              <i :class="model.icon"></i>
            </div>
            <div class="model-title">
              <h3>{{ model.name }}</h3>
              <span class="model-badge" v-if="model.badge">{{ model.badge }}</span>
            </div>
          </div>
          
          <p class="description">{{ model.description }}</p>
          
          <div class="model-features">
            <div class="feature-item" v-for="(feature, index) in model.features" :key="index">
              <i class="el-icon-check"></i>
              <span>{{ feature }}</span>
            </div>
          </div>
          
          <div class="model-metrics">
            <div class="metric-item">
              <i class="el-icon-data-line"></i>
              <div class="metric-info">
                <span class="metric-label">精度</span>
                <span class="metric-value">{{ model.accuracy }}</span>
              </div>
            </div>
            <div class="metric-item">
              <i class="el-icon-coin"></i>
              <div class="metric-info">
                <span class="metric-label">大小</span>
                <span class="metric-value">{{ model.size }}</span>
              </div>
            </div>
          </div>

          <div class="model-footer">
            <button 
              class="select-btn"
              :class="{ 'selected': selectedModel === model.id }"
            >
              <i :class="selectedModel === model.id ? 'el-icon-check' : 'el-icon-right'"></i>
              {{ selectedModel === model.id ? '已选择' : '选择此模型' }}
            </button>
          </div>
        </div>
      </div>

      <div class="action-buttons">
        <button 
          class="btn btn-primary" 
          :disabled="!selectedModel"
          @click="proceedToDownload"
        >
          <i class="el-icon-download"></i>
          下一步：下载模型
        </button>
        <p class="hint-text" v-if="!selectedModel">请选择一个模型以继续</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useStore } from 'vuex'
import axios from 'axios'

export default {
  name: 'ModelSelectionPage',
  setup() {
    const router = useRouter()
    const store = useStore()
    const selectedModel = ref(null)
    const username = computed(() => store.state.user?.username || '用户')

    const models = [
      {
        id: 'model1',
        name: '通用对话模型',
        description: '适用于日常对话、问答等通用场景，支持中英双语，具有广泛的知识储备',
        accuracy: '95%',
        size: '2.5GB',
        icon: 'el-icon-chat-line-round',
        badge: '推荐',
        features: ['多语言支持', '通用知识问答', '上下文理解', '实时响应']
      },
      {
        id: 'model2',
        name: '专业知识模型',
        description: '专注于专业领域的问答和咨询，包含金融、医疗、法律等专业知识库',
        accuracy: '98%',
        size: '3.2GB',
        icon: 'el-icon-medal',
        badge: '高精度',
        features: ['专业知识库', '准确度高', '多领域支持', '持续更新']
      },
      {
        id: 'model3',
        name: '轻量级模型',
        description: '针对移动设备和边缘计算优化，低延迟，适合实时交互场景',
        accuracy: '92%',
        size: '1.8GB',
        icon: 'el-icon-lightning',
        features: ['快速响应', '低资源占用', '边缘计算支持', '实时交互']
      },
      {
        id: 'model4',
        name: '企业定制模型',
        description: '根据企业需求定制的专属模型，支持私有化部署和数据安全隔离',
        accuracy: '97%',
        size: '4.0GB',
        icon: 'el-icon-office-building',
        features: ['私有部署', '数据隔离', '定制化训练', '企业级支持']
      }
    ]

    const selectModel = async (modelId) => {
      selectedModel.value = modelId
      store.commit('SET_SELECTED_MODEL', modelId)
      try {
        const modelIndex = parseInt(modelId.replace('model', '')) - 1
        await axios.post('/main-model/select-model', {
          uuid: store.state.user?.uuid,
          modelIndex
        })
      } catch (e) {
        alert('模型选择失败：' + (e.response?.data?.message || e.message))
      }
    }

    const proceedToDownload = () => {
      if (selectedModel.value) {
        router.push(`/download/${selectedModel.value}`)
      }
    }

    const goBack = () => {
      router.push('/login')
    }

    const handleLogout = () => {
      store.dispatch('logout')
      router.push('/login')
    }

    return {
      models,
      selectedModel,
      username,
      selectModel,
      proceedToDownload,
      goBack,
      handleLogout
    }
  }
}
</script>

<style scoped>
.model-selection {
  min-height: 100vh;
  background: #f0f2f5;
  position: relative;
}

.model-selection::before {
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

.status-left {
  display: flex;
  align-items: center;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  background: #f5f5f5;
  color: #595959;
}

.back-btn:hover {
  background: #e6f4ff;
  color: #1677ff;
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
  font-size: 14px;
}

.logout-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
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

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 24px;
  margin-bottom: 40px;
}

.model-card {
  background: white;
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
  cursor: pointer;
  transition: all 0.3s ease;
  border: 2px solid transparent;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.model-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.model-card.selected {
  border-color: #1677ff;
  background: #f0f7ff;
}

.model-header {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.model-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  background: linear-gradient(135deg, #e6f4ff 0%, #bae0ff 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #1677ff;
  font-size: 28px;
  flex-shrink: 0;
}

.model-card.selected .model-icon {
  background: linear-gradient(135deg, #1677ff 0%, #4096ff 100%);
  color: white;
}

.model-title {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-title h3 {
  font-size: 20px;
  font-weight: 600;
  color: #262626;
  margin: 0;
}

.model-badge {
  display: inline-block;
  padding: 4px 12px;
  background: #f0f7ff;
  color: #1677ff;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 500;
}

.description {
  color: #595959;
  font-size: 15px;
  line-height: 1.6;
  margin: 0;
}

.model-features {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  padding: 20px 0;
  border-top: 1px solid #f0f0f0;
  border-bottom: 1px solid #f0f0f0;
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

.model-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.metric-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.metric-item i {
  font-size: 20px;
  color: #1677ff;
}

.metric-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-label {
  color: #8c8c8c;
  font-size: 13px;
}

.metric-value {
  color: #262626;
  font-size: 16px;
  font-weight: 500;
}

.model-footer {
  margin-top: auto;
}

.select-btn {
  width: 100%;
  padding: 12px;
  border: 2px solid #d9d9d9;
  border-radius: 8px;
  background: white;
  color: #595959;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.select-btn:hover {
  border-color: #1677ff;
  color: #1677ff;
  background: #f0f7ff;
}

.select-btn.selected {
  background: #1677ff;
  border-color: #1677ff;
  color: white;
}

.action-buttons {
  text-align: center;
  margin-top: 48px;
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
</style> 