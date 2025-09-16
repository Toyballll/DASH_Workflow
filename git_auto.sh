#!/bin/bash

# 1. 把所有修改添加到暂存区
git add .

# 2. 提交，带时间戳（如果你没输入参数）
if [ -z "$1" ]; then
    git commit -m "Auto commit on $(date '+%Y-%m-%d %H:%M:%S')"
else
    git commit -m "$1"
fi

# 3. 拉取远程最新代码并 rebase
git pull --rebase origin main

# 4. 推送到远程
git push origin main
