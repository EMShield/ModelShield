const path = require('path');
const fs = require('fs');
const mainModelService = require('../services/mainModel');

let downloadCount = 0;
const TOTAL_FILES = 2;

async function checkAndUnmount() {
    downloadCount++;
    if (downloadCount === TOTAL_FILES) {
        await mainModelService.unmountModelArea();
        downloadCount = 0;
    }
}

exports.downloadPt = async (req, res) => {
    try {
        const modelIndex = req.params.index;
        const filePath = `/home/cloud_server/nxg/modelSide_business/precision_area_mount/model_${modelIndex}.pt`;

        console.log('下载请求 modelIndex:', modelIndex, 'filePath:', filePath);

        // 检查文件是否存在
        try {
            await fs.promises.access(filePath, fs.constants.R_OK);
        } catch (error) {
            console.error('文件不存在或无法访问:', filePath);
            res.status(404).json({ error: '文件不存在' });
            await mainModelService.unmountModelArea();
            return;
        }

        // 获取文件信息
        const stats = await fs.promises.stat(filePath);
        const fileSize = stats.size;

        // 设置响应头
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Disposition', `attachment; filename=model_${modelIndex}.pt`);
        res.setHeader('Content-Length', fileSize);
        res.setHeader('Access-Control-Expose-Headers', 'Content-Disposition, Content-Length');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Pragma', 'no-cache');

        // 创建可读流
        const fileStream = fs.createReadStream(filePath);

        // 处理错误
        fileStream.on('error', async (error) => {
            console.error('文件流错误:', error);
            res.status(500).json({ error: '文件流错误' });
            await mainModelService.unmountModelArea();
        });

        // 处理完成
        fileStream.on('end', async () => {
            await checkAndUnmount();
        });

        // 管道传输
        fileStream.pipe(res);

    } catch (error) {
        console.error('下载处理失败:', error);
        res.status(500).json({ error: '下载失败' });
        await mainModelService.unmountModelArea();
    }
};

exports.downloadMd = async (req, res) => {
    try {
        const modelIndex = req.params.index;
        const filePath = `/home/cloud_server/nxg/modelSide_business/model_${modelIndex}.md`;

        // 检查文件是否存在
        try {
            await fs.promises.access(filePath, fs.constants.R_OK);
        } catch (error) {
            console.error('文件不存在或无法访问:', filePath);
            res.status(404).json({ error: '文件不存在' });
            await mainModelService.unmountModelArea();
            return;
        }

        // 获取文件信息
        const stats = await fs.promises.stat(filePath);
        const fileSize = stats.size;

        // 设置响应头
        res.setHeader('Content-Type', 'text/markdown');
        res.setHeader('Content-Disposition', `attachment; filename=guide_${modelIndex}.md`);
        res.setHeader('Content-Length', fileSize);
        res.setHeader('Access-Control-Expose-Headers', 'Content-Disposition, Content-Length');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Pragma', 'no-cache');

        // 创建可读流
        const fileStream = fs.createReadStream(filePath);

        // 处理错误
        fileStream.on('error', async (error) => {
            console.error('文件流错误:', error);
            res.status(500).json({ error: '文件流错误' });
            await mainModelService.unmountModelArea();
        });

        // 处理完成
        fileStream.on('end', async () => {
            await checkAndUnmount();
        });

        // 管道传输
        fileStream.pipe(res);

    } catch (error) {
        console.error('下载处理失败:', error);
        res.status(500).json({ error: '下载失败' });
        await mainModelService.unmountModelArea();
    }
};
