module.exports = {
    system: {
        allowedSystems: ['Windows', 'Linux', 'MacOS'],
        commands: {
            'Windows': 'systeminfo',
            'Linux': 'uname -a',
            'MacOS': 'system_profiler SPHardwareDataType'
        }
    }
};