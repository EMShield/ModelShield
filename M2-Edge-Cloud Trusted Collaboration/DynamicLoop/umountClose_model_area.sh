#!/bin/bash
# 卸载顺序：子加密分区优先

# 卸载两个子加密分区
sudo umount /home/cloud_server/modelSide_business/precision_core_mount
sudo umount /home/cloud_server/modelSide_business/not_precision_core_mount
# 关闭两个loop设备的映射
sudo cryptsetup luksClose precision_core_mapper
sudo cryptsetup luksClose not_precision_core_mapper
# 还得接触两个子加密分区的loop设备绑定，否则父加密分区无法卸载
sudo losetup -d /dev/loop2
sudo losetup -d /dev/loop3

# 卸载父加密分区
sudo umount /home/cloud_server/modelSide_business
sudo cryptsetup luksClose modelside_area_mapper
