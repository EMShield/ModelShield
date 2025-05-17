const mainModelService = require('../services/mainModel');
const { logger } = require('../utils/scripts');
const crypto = require('crypto');
const bloomFilter = require('../utils/BloomFilter');

exports.selectSystem = async (req, res) => {
    try {
        const { system } = req.body;
        if (!system) {
            return res.status(400).json({
                success: false,
                error: '系统类型不能为空'
            });
        }
        const result = await mainModelService.selectSystem(system);
        res.json(result);
    } catch (error) {
        logger.error('Controller Error:', error);
        res.status(500).json({
            success: false,
            error: error.message || '系统选择失败，请重试'
        });
    }
};

exports.uploadUUID = async (req, res) => {
    try {
        console.log('Received uploadUUID request:', req.body);
        const { system, uuid } = req.body;
        console.log('Processing uploadUUID with system:', system, 'and uuid:', uuid);
        const result = await mainModelService.uploadUUID(system, uuid);
        console.log('uploadUUID result:', result);
        res.json(result);
    } catch (error) {
        console.error('uploadUUID error:', error);
        logger.error('Controller Error:', error);
        res.status(500).json({ error: error.message });
    }
};

exports.generateKey = async (req, res) => {
    try {
        const { uuid } = req.body;
        const result = await mainModelService.generateKey(uuid);
        res.json(result);
    } catch (error) {
        logger.error('Controller Error:', error);
        res.status(500).json({ error: error.message });
    }
};

exports.verifyIdentity = async (req, res) => {
    try {
        const { uuid, key } = req.body;
        const result = await mainModelService.verifyIdentity(uuid, key);
        res.json(result);
    } catch (error) {
        logger.error('Controller Error:', error);
        res.status(500).json({ error: error.message });
    }
};

exports.selectModel = async (req, res) => {
    try {
      const { uuid, modelIndex } = req.body;
      const result = await mainModelService.selectModel(uuid, modelIndex);
      res.json(result);
    } catch (error) {
      logger.error('Controller Error:', error);
      res.status(500).json({ error: error.message });
    }
};

exports.verifyUrlHash = async (req, res) => {
    try {
        const { uuid, url } = req.body;
        const result = await mainModelService.verifyUrlHash(uuid, url);
        res.json(result);
    } catch (error) {
        logger.error('Controller Error:', error);
        res.status(500).json({ error: error.message });
    }
};

exports.downloadCoreFiles = async (req, res) => {
    try {
        const { uuid, modelIndex } = req.body;
        const result = await mainModelService.downloadCoreFiles(uuid, modelIndex);
        res.json(result);
    } catch (error) {
        logger.error('Controller Error:', error);
        res.status(500).json({ error: error.message });
    }
};

exports.downloadSecondPart = async (req, res) => {
    try {
        const { uuid, modelIndex } = req.body;
        const result = await mainModelService.downloadSecondPart(uuid, modelIndex);
        res.json(result);
    } catch (error) {
        logger.error('Controller Error:', error);
        res.status(500).json({ error: error.message });
    }
};

exports.downloadAllCoreFiles = async (req, res) => {
    try {
        const { uuid, modelIndex } = req.body;
        const result = await mainModelService.downloadAllCoreFiles(uuid, modelIndex);
        res.json(result);
    } catch (error) {
        logger.error('Controller Error:', error);
        res.status(500).json({ error: error.message });
    }
};

exports.register = async (req, res) => {
    try {
        const { system, uuid } = req.body;
        if (!system || !uuid) {
            return res.status(400).json({
                success: false,
                message: '系统类型和UUID不能为空'
            });
        }

        // 验证系统类型
        const normalizedSystem = system.charAt(0).toUpperCase() + system.slice(1).toLowerCase();
        if (!['Windows', 'Linux', 'MacOS'].includes(normalizedSystem)) {
            return res.status(400).json({
                success: false,
                message: '不支持的系统类型'
            });
        }

        // 生成密钥
        const key = crypto.randomBytes(32).toString('hex');
        
        // 添加到布隆过滤器
        await bloomFilter.addUUID(uuid, key);

        return res.json({
            success: true,
            key: key,
            message: '注册成功'
        });
    } catch (error) {
        console.error('Registration error:', error);
        return res.status(500).json({
            success: false,
            message: error.message || '注册失败，请重试'
        });
    }
};

exports.login = async (req, res) => {
    try {
        const { uuid, key, system } = req.body;
        if (!uuid || !key || !system) {
            return res.status(400).json({
                success: false,
                message: 'UUID、密钥和系统类型不能为空'
            });
        }
        // 使用服务层的verifyIdentity方法
        const result = await mainModelService.verifyIdentity(uuid, key);
        if (!result.success) {
            return res.status(401).json({
                success: false,
                message: result.message
            });
        }
        // 登录成功，生成token等
        return res.json({
            success: true,
            token: 'mock-token', // 这里应生成真实token
            user: { uuid, system }
        });
    } catch (error) {
        return res.status(500).json({
            success: false,
            message: error.message || '登录失败，请重试'
        });
    }
};

exports.mountStatus = async (req, res) => {
    try {
        // 建议后续用更精细的挂载状态标志
        const isMounted = !!mainModelService.mountScriptPromise;
        res.json({ success: true, mounted: isMounted });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
};

exports.unmount = async (req, res) => {
    try {
        await mainModelService.unmountModelArea();
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
};
