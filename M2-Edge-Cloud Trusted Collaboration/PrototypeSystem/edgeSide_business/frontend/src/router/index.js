import { createRouter, createWebHistory } from 'vue-router'
import LoginPage from '@/components/pages/LoginPage.vue'
import ModelSelectionPage from '@/components/pages/ModelSelectionPage.vue'
import CoreDownloadPage from '@/components/pages/CoreDownloadPage.vue'
import ModelChatPage from '@/components/pages/ModelChatPage.vue'

const routes = [
  {
    path: '/',
    redirect: '/login'
  },
  {
    path: '/login',
    name: 'Login',
    component: LoginPage
  },
  {
    path: '/model-selection',
    name: 'ModelSelection',
    component: ModelSelectionPage,
    meta: { requiresAuth: true }
  },
  {
    path: '/download/:modelId',
    name: 'CoreDownload',
    component: CoreDownloadPage,
    meta: { requiresAuth: true }
  },
  {
    path: '/chat/:modelId',
    name: 'ModelChat',
    component: ModelChatPage,
    meta: { requiresAuth: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 导航守卫
router.beforeEach((to, from, next) => {
  const isAuthenticated = localStorage.getItem('token')
  
  if (to.matched.some(record => record.meta.requiresAuth)) {
    if (!isAuthenticated) {
      next({
        path: '/login',
        query: { redirect: to.fullPath }
      })
    } else {
      next()
    }
  } else {
    next()
  }
})

export default router 