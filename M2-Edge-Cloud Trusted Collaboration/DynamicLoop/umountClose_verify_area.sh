#!/bin/bash

sudo umount /home/cloud_server/verify_business/verify_area_mount
sudo cryptsetup luksClose verify_area_mapper

