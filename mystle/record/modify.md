1. 因为默认 multimodel 时，一张图片产生的 token 数太多(2000+)，所以我将 1600*900 的图片裁剪成了 800*450 的图片，同时还传入 image_aspect_ratio='square' 参数，这样可以大幅减少图片 token 数量 (500+)，从而减少显存占用，避免 OOM。

2. 在 llava_llama.py 的 generate 函数中，修复了 refresh=True 且 current_sequences 为空时，input_id 初始化失败的问题，强制将 input_ids 初始化为 input 的最后一个 token。