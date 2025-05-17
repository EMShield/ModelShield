const { v4: uuidv4 } = require('uuid');
const { exec } = require('child_process');
const iconv = require('iconv-lite');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class MainModelService {
    constructor() {
        this.systemInfo = new Map();
        this.userKeys = new Map();  // 存储 UUID 和 key_hash
        this.mountScriptPromise = null;     // 挂载脚本是否成功执行
        this.bloomFilter = require('../utils/BloomFilter');  // 添加这行
        this.allowedSystems = ['Windows', 'Linux', 'MacOS'];
    }

    async selectSystem(system) {
        try {
            // 转换为首字母大写，其余小写的格式
            system = system.charAt(0).toUpperCase() + system.slice(1).toLowerCase();
            
            if (!this.allowedSystems.includes(system)) {
                throw new Error('不支持的系统类型');
            }

            const systemId = uuidv4();
            
            // 根据不同系统返回对应的 UUID 获取命令
            const commands = {
                'Windows': 'wmic csproduct get uuid',
                'Linux': "dmidecode -s system-uuid | tr 'A-Z' 'a-z'",
                'MacOS': "ioreg -d2 -c IOPlatformExpertDevice | awk -F\\\" '/IOPlatformUUID/{print $(NF-1)}'"
            };

            const systemData = {
                system,
                timestamp: Date.now(),
                command: commands[system],
                registeredUUIDs: []  // 初始化空数组
            };

            this.systemInfo.set(systemId, systemData);

            return {
                success: true,
                systemId,
                message: '系统选择成功',
                showIdentityButtons: true
            };
        } catch (error) {
            console.error('Error selecting system:', error);
            throw new Error('系统选择失败：' + (error.message || '请重试'));
        }
    }

    async uploadUUID(system, uuid) {
        try {
        // 验证 UUID 格式
        if (!uuid || typeof uuid !== 'string') {
            throw new Error('无效的 UUID');
        }

        // 验证 UUID 是否符合预期格式
        const uuidRegex = /^[a-f0-9-]+$/i;
        if (!uuidRegex.test(uuid)) {
            throw new Error('UUID 格式不正确');
        }

            // 转换系统类型为标准格式
            system = system.charAt(0).toUpperCase() + system.slice(1).toLowerCase();

        // 检查系统类型是否有效
            if (!this.allowedSystems.includes(system)) {
            throw new Error('不支持的系统类型');
        }

	return {
            success: true,
            result: uuid
        };
        } catch (error) {
            console.error('Error uploading UUID:', error);
            throw error;
        }
    }  

    async generateKey(uuid) {
        try {
          const key = crypto.randomBytes(32).toString('hex');
          const exists = await this.bloomFilter.checkUUID(uuid);
          
          if (exists) {
            return {
              success: false,
              message: '您已注册，请重新登录',
              showIdentityButtons: true
            };
          }
    
          await this.bloomFilter.addUUID(uuid, key);
          
          return {
            success: true,
            message: '您的密钥文件已下载，请妥善保管',
            keyFile: {
              content: key,
              filename: `key_${uuid}.key`
            }
          };
        } catch (error) {
          throw error;
        }
    }

    async verifyIdentity(uuid, key) {
        try {
          const result = await this.bloomFilter.verifyUUIDAndKey(uuid, key);
          return {
            success: result.success,
            message: result.message,
            showModelButtons: result.success,
            showIdentityButtons: !result.success,
            showAuthButtons: result.message.includes('密钥错误')
          };
        } catch (error) {
          throw error;
        }
    } 

    async selectModel(uuid, modelIndex) {
        try {
          const urls = require('../data/url.json');
          const url = urls[modelIndex];
          if (!url) {
            throw new Error('无效的模型索引');
          }
          // 计算url_hash
          const url_hash = crypto.createHash('sha256').update(url).digest('hex');

          // 读取现有数据（如果文件存在）
          let userModelData = {};
          const dataPath = path.join(__dirname, '../data/user_model.json');
          try {
            const fileContent = await fs.readFile(dataPath, 'utf8');
            userModelData = JSON.parse(fileContent);
          } catch (error) {
            userModelData = {};
          }
      
          // 更新数据
          userModelData[uuid] = {
            url_hash,
            timestamp: Date.now()
          };

          // 保存到文件:只覆盖uuid一样的数据
          const jsonString = JSON.stringify(userModelData, null, 2)
          await fs.writeFile(dataPath, jsonString)

          // 选择模型后立即挂载分区
          this.mountScriptPromise = new Promise((resolve, reject) => {
            exec(`/usr/bin/sudo /home/cloud_server/nxg/mountModelArea.sh`, (error, stdout, stderr) => {
              if (error || stderr) {
                console.error('模型业务区挂载失败:', error || stderr);
                reject(error || new Error(stderr));
                return;
              }
              console.log('模型业务区挂载成功:', stdout);
              resolve(stdout);
            });
          });
          await this.mountScriptPromise;

          return {
            success: true,
            message: url,
            nextStage: true,
            countdown: 5
          };
        } catch (error) {
          throw error;
        }
    }

    // 验证URL是否正确
    // CoreWeightPage.vue与MainModelPage.vue共用一个后端服务文件：减少代码冗余
    async verifyUrlHash(uuid, url) {
        try {
            // 直接执行挂载分区脚本，无需校验 url_hash
                this.mountScriptPromise = new Promise((resolve, reject) => {
                    exec(`/usr/bin/sudo /home/cloud_server/nxg/mountModelArea.sh`, (error, stdout, stderr) => {
                        if (error || stderr) {
                            console.error('模型业务区挂载失败:', error || stderr);
                            reject(error || new Error(stderr));
                            return;
                        }
                        console.log('模型业务区挂载成功:', stdout);
                        resolve(stdout);
                    });
                });

                // 等待挂载完成
                await this.mountScriptPromise;
                
                return {
                    success: true,
                    message: 'URL 验证成功'
                };
        } catch (error) {
            console.error('URL验证失败:', error);
            throw error;
        }
    }
    

    async downloadCoreFiles(uuid, modelIndex) {
        try {
            const index = parseInt(modelIndex);
            if (isNaN(index)) {
                throw new Error('未找到对应的模型');
            }
            // 确保挂载已完成
            if (!this.mountScriptPromise) {
                throw new Error('挂载脚本未执行');
            }
            await this.mountScriptPromise;
            const mountPoint = '/home/cloud_server/nxg/modelSide_business';
            const ptFile = path.join(mountPoint, 'precision_area_mount', `model_${index}.pt`);
            const mdFile = path.join(mountPoint, `model_${index}.md`);
            await Promise.all([
                fs.access(ptFile, fs.constants.R_OK),
                fs.access(mdFile, fs.constants.R_OK)
            ]);
            // 不再自动卸载
            return {
                success: true,
                message: '核心权重文件已准备就绪，开始下载',
                modelIndex: index,
                files: {
                    pt: {
                        url: `/download/precision_area_mount/model_${index}.pt`,
                        filename: `model_${index}.pt`
                    },
                    md: {
                        url: `/download/model_${index}.md`,
                        filename: `model_${index}.md`
                    }
                }
            };
        } catch (error) {
            console.error('下载核心文件失败:', error);
            throw error;
        }
    }

    async downloadSecondPart(uuid, modelIndex) {
        try {
            const urls = JSON.parse(await fs.readFile(path.join(__dirname, '../data/url.json'), 'utf8'));
            // 确保挂载已完成
            if (!this.mountScriptPromise) {
                throw new Error('挂载脚本未执行');
            }
            await this.mountScriptPromise;
            const mountPoint = '/home/cloud_server/nxg/modelSide_business';
            const ptFile = path.join(mountPoint, 'precision_area_mount', `model_${modelIndex}.pt`);
            await fs.access(ptFile, fs.constants.R_OK);
            // 不再自动卸载
            return {
                success: true,
                message: '核心权重文件2已准备就绪，开始下载',
                files: {
                    pt: {
                        url: `/download/precision_area_mount/model_${modelIndex}.pt`,
                        filename: `model_${modelIndex}.pt`
                    }
                }
            };
        } catch (error) {
            throw error;
        }
    }

    // downloadAllCoreFiles 保证挂载完成后并行下载，不再自动卸载
    async downloadAllCoreFiles(uuid, modelIndex) {
        try {
            // 挂载分区（如果未挂载）
            if (!this.mountScriptPromise) {
                throw new Error('挂载脚本未执行');
            }
            await this.mountScriptPromise;
            // 并行下载两个核心文件
            const [coreResult, secondResult] = await Promise.all([
                this.downloadCoreFiles(uuid, modelIndex),
                this.downloadSecondPart(uuid, modelIndex)
            ]);
            // 不再自动卸载
            return {
                success: coreResult.success && secondResult.success,
                message: '两个核心文件已准备就绪',
                files: {
                    coreFiles: coreResult.files,
                    secondPart: secondResult.files
                }
            };
        } catch (error) {
            throw error;
        }
    }
}

module.exports = new MainModelService();
