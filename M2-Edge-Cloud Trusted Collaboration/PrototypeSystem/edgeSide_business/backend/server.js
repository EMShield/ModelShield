const express = require('express');
const cors = require('cors');
require('dotenv').config();
const axios = require('axios');

const mainModelRoutes = require('./routes/mainModel');
const errorHandler = require('./middleware/errorHandler');
const downloadRoutes = require('./routes/download');
const app = express();
const port = 3111;

// 配置CORS
const corsOptions = {
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'Range'],
    exposedHeaders: ['Content-Disposition', 'Content-Length', 'Content-Range'],
    credentials: true,
    maxAge: 86400,
    preflightContinue: false,
    optionsSuccessStatus: 204
};

// 添加请求日志中间件
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} ${req.method} ${req.url}`);
    next();
});

app.use(cors(corsOptions));
app.use(express.json());

// 添加根路由处理
app.get('/', (req, res) => {
    console.log('访问根路由');
    res.json({ 
        message: '服务器运行正常',
        status: 'ok'
    });
});

// API 路由
app.use('/api/main-model', mainModelRoutes);
app.use('/api/download', downloadRoutes);

// 讯飞星火API代理接口
app.post('/api/spark-chat', async (req, res) => {
  try {
    const response = await axios.post(
      'https://spark-api-open.xf-yun.com/v2/chat/completions',
      req.body,
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer rDqCHykzynbOhatJkRqT:evQLGhnXwCJwboVOkcao'
        }
      }
    );
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: '代理请求失败', detail: err.message });
  }
});

// 错误处理
app.use(errorHandler);

// 处理 404 错误
app.use((req, res) => {
    console.log(`404 错误: ${req.path}`);
    res.status(404).json({
        error: '未找到请求的资源',
        path: req.path
    });
});

// 全局错误处理
app.use((err, req, res, next) => {
    console.error('服务器错误:', err);
    res.status(500).json({
        error: '服务器内部错误',
        message: err.message
    });
});

const server = app.listen(port, '0.0.0.0', () => {
    console.log(`服务器运行在 http://0.0.0.0:${port}`);
});

server.on('error', (error) => {
    console.error('服务器启动错误:', error);
    if (error.code === 'EADDRINUSE') {
        console.error(`端口 ${port} 已被占用`);
    }
});

process.on('uncaughtException', (error) => {
    console.error('未捕获的异常:', error);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('未处理的Promise拒绝:', reason);
});
