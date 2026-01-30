# 创建 record 目录 (如果它还不存在)
mkdir -p mystle/record

# 运行脚本并将所有输出重定向到带时间戳的文件中
python mystle/eval_pictures.py > mystle/record/speed_test_$(date +%Y.%m.%d-%H.%M).log 2>&1