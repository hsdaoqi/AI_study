"""
inspect_qwen.py —— 解剖 Qwen3-0.6B 的每个文件,亲眼验证知乎文章里的数字
=====================================================================
目的:把"模型的各个文件"从文字变成你亲手翻过的东西。
     这也是 SFT 训练脚本的第一步(加载模型),等于给后面铺路。

怎么跑(AutoDL):
  source /etc/network_turbo            # 学术加速(下模型)
  export HF_ENDPOINT=https://hf-mirror.com   # 若上面还慢,用镜像
  pip install transformers safetensors    # 若没装
  python inspect_qwen.py

它会:①下载 Qwen3-0.6B ②逐个打开 6 类文件 ③打印关键信息,标注该对上文章的哪个数字
"""

import os
import json
from pathlib import Path

# ============================================================
# 第 0 步:下载模型(只下一次,之后本地缓存)
# ============================================================
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-0.6B"
print(f"\n{'='*60}\n第0步:下载 {MODEL_ID}(首次约 1.4GB,之后走缓存)\n{'='*60}")
model_dir = snapshot_download(repo_id=MODEL_ID)
print(f"模型下载到:{model_dir}")

# 先看一眼目录里到底有哪些文件(对照文章"一、模型目录结构")
print(f"\n目录内容:")
for f in sorted(os.listdir(model_dir)):
    size = os.path.getsize(os.path.join(model_dir, f)) / 1e6
    print(f"  {f:35s}  {size:8.2f} MB")


# ============================================================
# 第 1 步:config.json —— 模型的"身份证/形状"
# ============================================================
print(f"\n{'='*60}\n第1步:config.json(模型形状,建空壳用)\n{'='*60}")
with open(Path(model_dir) / "config.json") as f:
    cfg = json.load(f)

# 打印文章里点名的关键参数,你逐条对照文章"关键参数说明"
print(f"  architectures      = {cfg['architectures']}       # 用哪个模型类")
print(f"  num_hidden_layers  = {cfg['num_hidden_layers']}    # 文章说 28 层")
print(f"  hidden_size        = {cfg['hidden_size']}          # 文章说 1024")
print(f"  num_attention_heads= {cfg['num_attention_heads']}  # 文章说 16 (Q头)")
print(f"  num_key_value_heads= {cfg['num_key_value_heads']}  # 文章说 8  (KV头, GQA)")
print(f"  intermediate_size  = {cfg['intermediate_size']}    # 文章说 3072 (MLP中间层)")
print(f"  vocab_size         = {cfg['vocab_size']}           # 文章说 151936")
print(f"  torch_dtype        = {cfg['torch_dtype']}          # bfloat16")
print(f"  tie_word_embeddings= {cfg.get('tie_word_embeddings')}  # 是否共享 embedding 和 lm_head")
# 自己验证一条公式:hidden_size 应 = ? (文章说 Q总维度=16*128=2048,注意这与hidden不同)
print(f"  → 验证 GQA:{cfg['num_attention_heads']}个Q头 / {cfg['num_key_value_heads']}个KV头 = {cfg['num_attention_heads']//cfg['num_key_value_heads']} (每组共享)")


# ============================================================
# 第 2 步:model.safetensors —— 权重(大脑)
# ============================================================
print(f"\n{'='*60}\n第2步:model.safetensors(权重,填进空壳)\n{'='*60}")
from safetensors.torch import load_file

# 找到权重文件(小模型是单个 model.safetensors)
st_path = Path(model_dir) / "model.safetensors"
weights = load_file(str(st_path))

total_params = sum(t.numel() for t in weights.values())
print(f"  张量总数 = {len(weights)}          # 文章说 311 个")
print(f"  参数总量 = {total_params:,}   # 文章说 751,632,384 (0.75B)")

# 打印几个关键张量的名字和形状,对照文章"权重字典结构"
print(f"\n  几个关键张量(名字 = HF 命名规范):")
for name in ["model.embed_tokens.weight",
             "model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.self_attn.k_proj.weight",
             "model.layers.0.mlp.gate_proj.weight",
             "model.norm.weight"]:
    if name in weights:
        print(f"    {name:45s} {tuple(weights[name].shape)}  {weights[name].dtype}")
# 你会看到 q_proj 是 [2048,1024],k_proj 是 [1024,1024](GQA,KV头少一半)—— 对上文章


# ============================================================
# 第 3 步:tokenizer —— 文字 <-> token id 的翻译器
# ============================================================
print(f"\n{'='*60}\n第3步:tokenizer(文字<->token,你VQVAE里'图->token'的文字版)\n{'='*60}")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(model_dir)
print(f"  tokenizer 类型 = {type(tok).__name__}")
print(f"  词表大小       = {tok.vocab_size}")
print(f"  eos_token      = {tok.eos_token!r} (id={tok.eos_token_id})   # SFT关键:模型学会答完停下")
print(f"  pad_token      = {tok.pad_token!r} (id={tok.pad_token_id})")

# 亲手把一句话变成 token,再变回来
text = "你好,世界"
ids = tok(text)["input_ids"]
print(f"\n  '{text}'  ->  token ids: {ids}")
print(f"  再解码回来: {tok.decode(ids)!r}")
# 逐个 token 看看切成了啥
print(f"  逐token:{[tok.decode([i]) for i in ids]}")


# ============================================================
# 第 4 步:chat template —— SFT 数据要用它拼对话(下节课重点)
# ============================================================
print(f"\n{'='*60}\n第4步:chat template(把'对话'拼成模型能读的字符串,SFT必用)\n{'='*60}")
messages = [
    {"role": "user", "content": "1+1等于几?"},
    {"role": "assistant", "content": "等于2。"},
]
# apply_chat_template:把对话按 Qwen 的格式拼起来(带 <|im_start|> <|im_end|> 等特殊token)
chat_str = tok.apply_chat_template(messages, tokenize=False)
print("  一段对话拼出来长这样(注意特殊token的位置):")
print("  " + chat_str.replace("\n", "\n  "))
print("\n  → 下节课讲 SFT 数据格式时,你会看到训练数据就是这么组织的")


# ============================================================
# 第 5 步:用高层接口加载整个模型(SFT脚本的第一行)
# ============================================================
print(f"\n{'='*60}\n第5步:一行加载完整模型(这就是SFT脚本第一步)\n{'='*60}")
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
n_params = sum(p.numel() for p in model.parameters())
print(f"  模型加载成功,参数量 = {n_params:,}")
print(f"  模型类 = {type(model).__name__}")
print(f"  → from_pretrained 干的事 = 读config建空壳 + 读safetensors填权重 + 读tokenizer配置")

print(f"\n{'='*60}\n✅ 解剖完成。你已经亲手翻过 Qwen3-0.6B 的每个文件。\n"
      f"   下一步:SFT 数据集格式 + Loss 公式\n{'='*60}")
