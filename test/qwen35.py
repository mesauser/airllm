# from transformers import AutoModel, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# model = AutoModel.from_pretrained(model_path, local_files_only=True)
import torch
torch.cuda.memory._record_memory_history()
orig = torch.empty_strided
def traced(*a, **k):
    import traceback
    tb = "".join(traceback.format_stack()[-6:-1])
    if "cuda" in str(k.get("device", "")):
        print("empty_strided", a[:2], "bytes~",
              abs(__import__("operator").mul(*a[0])) * 2, "\n", tb)
    return orig(*a, **k)
torch.empty_strided = traced

from transformers.quantizers import quantizers_utils as _qu
def _patched_should_convert(full_name, keys):
    for key in keys:
        if  full_name.endswith(key):
        # if full_name.startswith(key + '.') or full_name.startswith(key) or full_name.endswith(key):
            return False
    return True
_qu.should_convert_module = _patched_should_convert


def _replace_fp8_linears(model, use_strict=True):
    import torch
    import torch.nn.functional as F

    base_model = model.model if hasattr(model, 'model') else model
    for name, mod in base_model.named_modules():
        if type(mod).__name__ != 'FP8Linear':
            continue

        if use_strict:
            def strict_forward(self, x):
                w_fp8 = self.weight
                scale_inv = self.weight_scale_inv
                O, I = w_fp8.shape
                num_blocks_out, num_blocks_in = scale_inv.shape
                bs_out = O // num_blocks_out
                bs_in = I // num_blocks_in
                bs = max(bs_out, bs_in)  # 通常相等
                scale_expanded = scale_inv.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)
                scale_expanded = scale_expanded[:O, :I]
                w_deq = w_fp8.to(torch.float32) * scale_expanded
                bias = getattr(self, 'bias', None)
                return F.linear(x.to(torch.float32), w_deq, bias).to(x.dtype)

            mod.forward = lambda x, self=mod: strict_forward(self, x)
        else:
            raise NotImplementedError("仅支持 use_strict=True")
    return model
    """
    将模型中所有 FP8Linear 模块的 forward 替换为自定义实现。
    use_strict=True：反量化后做普通线性（精度最高，速度较慢）
    use_strict=False：暂不支持简化版，请保持 True
    """
    import torch
    import torch.nn.functional as F

    for name, mod in model.named_modules():
        if type(mod).__name__ != 'FP8Linear':
            continue

        # 获取 block_size（默认 128）
        block_size = getattr(mod, 'block_size', 128)

        if use_strict:
            def strict_forward(self, x):
                # 获取 FP8 权重和 scale_inv
                w_fp8 = self.weight                     # shape (O, I), dtype fp8
                scale_inv = self.weight_scale_inv       # shape (num_blocks_out, num_blocks_in)

                O, I = w_fp8.shape
                bs = block_size

                # 将 scale_inv 展开到完整权重形状
                scale_expanded = scale_inv.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)
                scale_expanded = scale_expanded[:O, :I]   # 裁剪到实际尺寸

                # 反量化：w_fp8 * scale_inv → float32
                w_deq = w_fp8.to(torch.float32) * scale_expanded

                bias = getattr(self, 'bias', None)
                return F.linear(x.to(torch.float32), w_deq, bias).to(x.dtype)

            # 绑定 forward（注意闭包捕获 self）
            mod.forward = lambda x, self=mod: strict_forward(self, x)

        else:
            # 快速版暂不实现，可自行接入 ScaledMMFP8Linear
            raise NotImplementedError("简化版仅支持 use_strict=True")

    return model




from airllm import AutoModel
model_path = "/data/guowei/model27bfp8/"
# model_path = "/data/guowei/qwen06/"
model = AutoModel.from_pretrained(model_path)
_replace_fp8_linears(model.model.model, use_strict=True)

# print(model)
# print(model.model)
# print(model.model.model)
# model = AutoModel.from_pretrained(model_path,
# profiling_mode=True,
# # prefetching=True,
# )
'''
model.model.get_submodule('model.language_model.layers.33.linear_attn.out_proj')


'''
# input_text = [
#         'What is the capital of United States?',
#         #'I like',
#     ]

# input_text = ['用一句话解释什么是向量数据库']
input_text = ['4*128结果 /no_think']

MAX_LENGTH = 4096

# 关键：enable_thinking=False
messages = [{"role": "user", "content": "4 * 128 只输出结果 /no_think"}]
input_prompt = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,        # ← 关思考
)

# input_tokens = model.tokenizer(input_text,
input_tokens = model.tokenizer(input_prompt,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=4096, 
    padding=False)

model.model.model._tok = model.tokenizer
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=1024,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

torch.cuda.memory._dump_snapshot("startup.pickle")
torch.cuda.memory._record_memory_history(enabled=None)