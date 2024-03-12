#!/bin/sh

today=$(date "+%Y-%m-%d")
echo ${today}

nohup python manage.py runserver 0.0.0.0:22335 > lt4/log/${today}_django.out &