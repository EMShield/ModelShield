const express = require('express');
const router = express.Router();
const mainModelController = require('../controllers/mainModel');

router.post('/register', mainModelController.register);
router.post('/system-select', mainModelController.selectSystem);
router.post('/upload-uuid', mainModelController.uploadUUID);
router.post('/generate-key', mainModelController.generateKey);
router.post('/verify-identity', mainModelController.verifyIdentity);
router.post('/select-model', mainModelController.selectModel);
router.post('/verify-url-hash', mainModelController.verifyUrlHash);
router.post('/download-core-files', mainModelController.downloadCoreFiles);
router.post('/download-second-part', mainModelController.downloadSecondPart);
router.post('/login', mainModelController.login);
router.post('/download-all-core-files', mainModelController.downloadAllCoreFiles);
router.get('/mount-status', mainModelController.mountStatus);
router.post('/unmount', mainModelController.unmount);
module.exports = router;
