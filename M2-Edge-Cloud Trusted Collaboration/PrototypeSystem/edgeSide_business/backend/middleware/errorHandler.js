const { logger } = require('../utils/scripts');

module.exports = (err, req, res, next) => {
    logger.error('Error:', err);
    
    // 确保错误响应格式统一
    res.status(err.status || 500).json({
        success: false,
        error: err.message || '服务器内部错误',
        path: req.path
    });
};