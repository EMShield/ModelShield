#!/bin/bash

key_id=$(keyctl search @s user my_persistent_key)
key_value=$(keyctl pipe $key_id)


echo $key_value | sudo -S cryptsetup luksOpen /dev/loop4 verify_area_mapper
sudo mount /dev/mapper/verify_area_mapper /home/cloud_server/verify_business/verify_area_mount
