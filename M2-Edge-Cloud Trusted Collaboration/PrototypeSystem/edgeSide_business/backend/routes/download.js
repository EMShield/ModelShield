const express = require('express');
const router = express.Router();
const downloadController = require('../controllers/download');

router.get('/precision_area_mount/model_:index.pt', downloadController.downloadPt);
router.get('/model_:index.md', downloadController.downloadMd);

module.exports = router;
