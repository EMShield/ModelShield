const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class BloomFilter {
  /*
  UUID 和 key 的验证通过布隆过滤器完成
  URL 的验证需要查询 user_model.json 文件：因为URL匹配成功后需要执行挂载脚本，所以最好割裂开
  */
  constructor() {
    this.filterSize = 256;
    this.hashFunctions = 3;
    this.filters = new Map();  // 存储普通布隆过滤器
    this.counterFilter = new Array(this.filterSize).fill(0);  // 计数布隆过滤器
    this.hasFilter = false;  // 标志位
    this.currentFilterIndex = 0;
  }

  // 计算多个哈希值
  getHashIndexes(uuid) {
    const indexes = [];
    for (let i = 0; i < this.hashFunctions; i++) {
      const hash = crypto
        .createHash('sha256')
        .update(uuid + i.toString())
        .digest('hex');
      indexes.push(parseInt(hash, 16) % this.filterSize);
    }
    return indexes;
  }

  // 创建新的布隆过滤器
  createNewFilter() {
    const filter = new Array(this.filterSize).fill(0);
    this.filters.set(this.currentFilterIndex, filter);
    this.hasFilter = true;
    return this.currentFilterIndex++;
  }

  // 生成密钥哈希（Windows环境下模拟Argon2+cityHash的效果）
  async generateKeyHash(key) {
    return new Promise((resolve, reject) => {
      try {
        // 第一步：使用scrypt生成128bit(16字节)的摘要，模拟Argon2
        crypto.scrypt(key, 'fixed_salt', 16, (err, derivedKey1) => {
          if (err) {
            reject(err);
            return;
          }
          
          // 第二步：使用另一个scrypt配置生成32bit(4字节)的摘要，模拟cityHash
          crypto.scrypt(derivedKey1.toString('hex'), 'city_salt', 4, (err, derivedKey2) => {
            if (err) {
              reject(err);
              return;
            }
            
            // 转换为16进制字符串，保持与cityHash.hash32的输出一致
            resolve(derivedKey2.toString('hex'));
          });
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  // 保存键值对到文件
  async saveKeyValuePairs(keyHash, filterIndex, uuid) {
    // 保存键值对文件1 (cityhash:bloom_index)
    const keyValuePath1 = path.join(__dirname, '../data/key_bloom_map.json');
    let keyMap = {};
    try {
      const content = await fs.readFile(keyValuePath1, 'utf8');
      keyMap = JSON.parse(content);
    } catch (error) {
      keyMap = {};
    }
    keyMap[keyHash] = filterIndex;
    await fs.writeFile(keyValuePath1, JSON.stringify(keyMap, null, 2));

    // 保存键值对文件2 (bloom_index:location:uuid)
    const keyValuePath2 = path.join(__dirname, '../data/bloom_uuid_map.json');
    let uuidMap = {};
    try {
      const content = await fs.readFile(keyValuePath2, 'utf8');
      uuidMap = JSON.parse(content);
    } catch (error) {
      uuidMap = {};
    }
    
    if (!uuidMap[filterIndex]) {
      uuidMap[filterIndex] = {};
    }
    
    const indexes = this.getHashIndexes(uuid);
    indexes.forEach(index => {
      uuidMap[filterIndex][index] = uuid;
    });
    
    await fs.writeFile(keyValuePath2, JSON.stringify(uuidMap, null, 2));
  }

  // 加载键值对文件
  async loadKeyValuePairs() {
    const keyValuePath1 = path.join(__dirname, '../data/key_bloom_map.json');
    const keyValuePath2 = path.join(__dirname, '../data/bloom_uuid_map.json');
    
    let keyMap = {};
    let uuidMap = {};
    
    try {
      const content1 = await fs.readFile(keyValuePath1, 'utf8');
      keyMap = JSON.parse(content1);
    } catch (error) {
      keyMap = {};
    }
    
    try {
      const content2 = await fs.readFile(keyValuePath2, 'utf8');
      uuidMap = JSON.parse(content2);
    } catch (error) {
      uuidMap = {};
    }
    
    return { keyMap, uuidMap };
  }

  // 查找已存在的UUID
  async findExistingUUID(uuid, indexes) {
    const { uuidMap } = await this.loadKeyValuePairs();
    
    for (const filterIndex in uuidMap) {
      const filterLocations = uuidMap[filterIndex];
      for (const index of indexes) {
        if (filterLocations[index] === uuid) {
          return true;
        }
      }
    }
    return false;
  }

  // 检查UUID是否存在
  async checkUUID(uuid) {
    if (!this.hasFilter) {
      return false;
    }
    
    const indexes = this.getHashIndexes(uuid);
    return await this.findExistingUUID(uuid, indexes);
  }

  // 添加新的UUID
  async addUUID(uuid, key) {
    try {
      const keyHash = await this.generateKeyHash(key);
      const filterIndex = this.createNewFilter();
      const indexes = this.getHashIndexes(uuid);
      
      // 更新布隆过滤器和计数器
      const filter = this.filters.get(filterIndex);
      indexes.forEach(index => {
        filter[index] = 1;
        this.counterFilter[index]++;
      });

      // 保存键值对文件
      await this.saveKeyValuePairs(keyHash, filterIndex, uuid);
      
      return true;
    } catch (error) {
      console.error('Error in addUUID:', error);
      throw error;
    }
  }

  // 验证UUID和密钥
  async verifyUUIDAndKey(uuid, key) {
    try {
      if (!this.hasFilter) {
        return { success: false, message: '未检索到您的身份，请先注册' };
      }

      const keyHash = await this.generateKeyHash(key);
      const filterData = await this.loadKeyValuePairs();
      
      if (!(keyHash in filterData.keyMap)) {
        return { success: false, message: '未检索到您的身份，请先注册' };
      }

      const filterIndex = filterData.keyMap[keyHash];
      const filter = this.filters.get(filterIndex);
      
      if (!filter) {
        return { success: false, message: '系统错误，请重试' };
      }

      const indexes = this.getHashIndexes(uuid);
      
      if (indexes.every(index => filter[index] === 1)) {
        return { success: true, message: '身份验证成功' };
      }

      if (indexes.every(index => this.counterFilter[index] === 0)) {
        return { success: false, message: '未检索到您的身份，请先注册' };
      }

      const existingUUID = await this.findExistingUUID(uuid, indexes);
      return existingUUID 
        ? { success: false, message: '您的密钥错误，请重新上传' }
        : { success: false, message: '未检索到您的身份，请先注册' };
    } catch (error) {
      console.error('Error in verifyUUIDAndKey:', error);
      return { success: false, message: '系统错误，请重试' };
    }
  }
}

module.exports = new BloomFilter();