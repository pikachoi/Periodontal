#!/bin/sh

sudo ip link delete dev docker0
sudo ip link delete dev virbr0
sudo ip link delete dev virbr0-nic

echo "Finish"
