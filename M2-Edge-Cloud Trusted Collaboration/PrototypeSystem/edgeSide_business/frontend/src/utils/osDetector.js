/**
 * 操作系统检测工具
 * 用于检测当前用户的操作系统类型
 */

export function detectOS() {
  const userAgent = navigator.userAgent || navigator.vendor || window.opera;
  
  // 检测Windows
  if (/windows|win32/i.test(userAgent)) {
    return {
      name: 'Windows',
      version: getWindowsVersion(userAgent),
      isWindows: true
    };
  }
  
  // 检测macOS
  if (/macintosh|mac os x/i.test(userAgent)) {
    return {
      name: 'macOS',
      version: getMacVersion(userAgent),
      isMac: true
    };
  }
  
  // 检测Linux
  if (/linux/i.test(userAgent)) {
    return {
      name: 'Linux',
      version: 'Unknown',
      isLinux: true
    };
  }
  
  // 检测Android
  if (/android/i.test(userAgent)) {
    return {
      name: 'Android',
      version: getAndroidVersion(userAgent),
      isAndroid: true
    };
  }
  
  // 检测iOS
  if (/iPad|iPhone|iPod/.test(userAgent) && !window.MSStream) {
    return {
      name: 'iOS',
      version: getIOSVersion(userAgent),
      isIOS: true
    };
  }
  
  // 默认返回未知
  return {
    name: 'Unknown',
    version: 'Unknown'
  };
}

// 获取Windows版本
function getWindowsVersion(userAgent) {
  const match = userAgent.match(/Windows NT (\d+\.\d+)/);
  if (match) {
    const version = parseFloat(match[1]);
    switch (version) {
      case 10:
        return '10';
      case 6.3:
        return '8.1';
      case 6.2:
        return '8';
      case 6.1:
        return '7';
      case 6.0:
        return 'Vista';
      case 5.2:
        return 'XP 64-bit';
      case 5.1:
        return 'XP';
      default:
        return `${version}`;
    }
  }
  return 'Unknown';
}

// 获取macOS版本
function getMacVersion(userAgent) {
  const match = userAgent.match(/Mac OS X (\d+[._]\d+[._]\d+)/);
  if (match) {
    return match[1].replace(/_/g, '.');
  }
  return 'Unknown';
}

// 获取Android版本
function getAndroidVersion(userAgent) {
  const match = userAgent.match(/Android (\d+(\.\d+)*)/);
  if (match) {
    return match[1];
  }
  return 'Unknown';
}

// 获取iOS版本
function getIOSVersion(userAgent) {
  const match = userAgent.match(/OS (\d+[._]\d+[._]\d+)/);
  if (match) {
    return match[1].replace(/_/g, '.');
  }
  return 'Unknown';
} 