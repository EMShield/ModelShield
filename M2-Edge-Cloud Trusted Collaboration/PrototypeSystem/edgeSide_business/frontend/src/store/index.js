import { createStore } from 'vuex'
import axios from 'axios'

// 配置 axios 默认值
axios.defaults.baseURL = '/api'  // 通过Nginx代理转发
axios.defaults.timeout = 30000 // 30秒超时
axios.defaults.headers.common['Content-Type'] = 'application/json'

// 添加请求拦截器
axios.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => {
    console.error('请求拦截器错误:', error)
    return Promise.reject(error)
  }
)

// 添加响应拦截器
axios.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      switch (error.response.status) {
        case 401:
          // 未授权，清除 token 并跳转到登录页面
          localStorage.removeItem('token')
          localStorage.removeItem('user')
          window.location.href = '/login'
          break
        case 403:
          console.error('权限不足')
          break
        case 500:
          console.error('服务器错误:', error.response.data)
          break
        default:
          console.error('请求失败:', error.response.data)
      }
    } else if (error.request) {
      console.error('无法连接到服务器，请检查网络连接')
    } else {
      console.error('请求配置错误:', error.message)
    }
    return Promise.reject(error)
  }
)

const store = createStore({
  state: {
    user: null,
    token: localStorage.getItem('token') || null,
    isAuthenticated: false,
    currentOS: null,
    selectedModel: null,
    downloadedFiles: [],
  },
  
  getters: {
    isAuthenticated: state => state.isAuthenticated,
    currentUser: state => state.user,
    currentOS: state => state.currentOS,
    selectedModel: state => state.selectedModel,
    downloadedFiles: state => state.downloadedFiles,
    token: state => state.token
  },
  
  mutations: {
    SET_USER(state, user) {
      state.user = user
      state.isAuthenticated = !!user
    },
    SET_TOKEN(state, token) {
      state.token = token
      if (token) {
        localStorage.setItem('token', token)
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      } else {
        localStorage.removeItem('token')
        delete axios.defaults.headers.common['Authorization']
      }
    },
    CLEAR_AUTH(state) {
      state.user = null
      state.token = null
      state.isAuthenticated = false
      localStorage.removeItem('token')
      delete axios.defaults.headers.common['Authorization']
    },
    SET_OS(state, os) {
      state.currentOS = os
    },
    SET_SELECTED_MODEL(state, modelId) {
      state.selectedModel = modelId
    },
    ADD_DOWNLOADED_FILE(state, fileId) {
      if (!state.downloadedFiles.includes(fileId)) {
        state.downloadedFiles.push(fileId)
      }
    }
  },
  
  actions: {
    async login({ commit }, credentials) {
      try {
        const response = await axios.post('/main-model/login', credentials)
        if (response.data.success) {
          commit('SET_TOKEN', response.data.token)
          commit('SET_USER', response.data.user)
          return { success: true }
        }
        return { success: false, message: response.data.message }
      } catch (error) {
        console.error('Login error:', error)
        return { 
          success: false, 
          message: error.response?.data?.message || '登录失败，请重试'
        }
      }
    },
    
    async register({ commit }, credentials) {
      try {
        const response = await axios.post('/main-model/register', credentials)
        if (response.data.success) {
          return { 
            success: true, 
            key: response.data.key,
            message: '注册成功'
          }
        }
        return { success: false, message: response.data.message }
      } catch (error) {
        console.error('Register error:', error)
        return { 
          success: false, 
          message: error.response?.data?.message || '注册失败，请重试'
        }
      }
    },
    
    async checkAuth({ commit, state }) {
      if (!state.token) {
        commit('CLEAR_AUTH')
        return false
      }

      try {
        const response = await axios.get('/check-auth')
        if (response.data.success) {
          commit('SET_USER', response.data.user)
          return true
        }
        commit('CLEAR_AUTH')
        return false
      } catch (error) {
        console.error('Auth check error:', error)
        commit('CLEAR_AUTH')
        return false
      }
    },
    
    logout({ commit }) {
      commit('CLEAR_AUTH')
    }
  }
})

export default store 