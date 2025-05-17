#!/bin/bash
# 挂载顺序：父加密分区优先
# 卸载的时候是：卸载loop--关闭loop--断开loop(父加密分区没断开)
# 解密的时候是：关联loop--打开loop--挂载loop

# 先从内核密钥环中读取出密钥
key_id=$(keyctl search @s user my_persistent_key)
key_value=$(keyctl pipe $key_id)

# 恢复父加密分区挂载
echo $key_value | sudo -S cryptsetup luksOpen /dev/loop1 modelside_area_mapper
sudo mount /dev/mapper/modelside_area_mapper /home/cloud_server/modelSide_business

# 恢复子加密分区的挂载
sudo losetup /dev/loop2 /home/cloud_server/modelSide_business/precision_core_vfs
sudo losetup /dev/loop3 /home/cloud_server/modelSide_business/not_precision_core_vfs
echo $key_value | sudo -S cryptsetup luksOpen /dev/loop2 precision_core_mapper
echo $key_value | sudo -S cryptsetup luksOpen /dev/loop3 not_precision_core_mapper
sudo mount /dev/mapper/precision_core_mapper /home/cloud_server/modelSide_business/precision_core_mount
sudo mount /dev/mapper/not_precision_core_mapper /home/cloud_server/modelSide_business/not_precision_core_mount
