// 生成只含元数据 + 张量表的合成 GGUF，用来验证 loader 的"签名校验 + 变种告警"。
// CLI: node tools/make_synthetic_gguf.mjs <7b|1p5b|llama8b> <out.gguf> [mutate] [name]
//   mutate: none | variant | layers | q6k | extra
// 也可被 import：buildGguf({ model, mutate, name }) -> { buf, total, tensors }
import fs from "node:fs";
import { pathToFileURL } from "node:url";

export const MODELS = {
  "7b":      { arch:"qwen2", L:28, D:3584, NH:28, NKV:4, HD:128, FFN:18944, V:151936, tie:false, bias:true,  theta:1e6, eps:1e-6,
               name:"Qwen2.5-7B-Instruct" },
  "1p5b":    { arch:"qwen2", L:28, D:1536, NH:12, NKV:2, HD:128, FFN:8960,  V:151936, tie:true,  bias:true,  theta:1e6, eps:1e-6,
               name:"Qwen2.5-1.5B-Instruct" },
  "llama8b": { arch:"llama", L:32, D:4096, NH:32, NKV:8, HD:128, FFN:14336, V:128256, tie:false, bias:false, theta:5e5, eps:1e-5,
               name:"DeepSeek-R1-Distill-Llama-8B" },
};

const Q4_K = 12, Q6_K = 14, F32 = 0;

export function buildGguf(modelArg, mutate = "none", nameOverride = null) {
  const m = { ...MODELS[modelArg] };
  if (!m) throw new Error("unknown model: " + modelArg);
  if (mutate === "layers") m.L += 1;
  const mname = nameOverride ?? (mutate === "variant" ? m.name.replace(/-Instruct$/, "") + "-Uncensored" : m.name);

  const tensors = [];
  const push = (n, shape, type, q6 = false) => tensors.push({ name: n, shape, type: q6 ? Q6_K : type });

  push("token_embd.weight", [m.D, m.V], Q4_K);
  for (let i = 0; i < m.L; i++) {
    const p = "blk." + i + ".";
    push(p + "attn_norm.weight", [m.D], F32);
    push(p + "attn_q.weight", [m.D, m.NH * m.HD], Q4_K);
    if (m.bias) push(p + "attn_q.bias", [m.NH * m.HD], F32);
    push(p + "attn_k.weight", [m.D, m.NKV * m.HD], Q4_K);
    if (m.bias) push(p + "attn_k.bias", [m.NKV * m.HD], F32);
    push(p + "attn_v.weight", [m.D, m.NKV * m.HD], Q4_K, mutate === "q6k" && i === 0);
    if (m.bias) push(p + "attn_v.bias", [m.NKV * m.HD], F32);
    push(p + "attn_output.weight", [m.NH * m.HD, m.D], Q4_K);
    push(p + "ffn_norm.weight", [m.D], F32);
    push(p + "ffn_gate.weight", [m.D, m.FFN], Q4_K);
    push(p + "ffn_up.weight", [m.D, m.FFN], Q4_K);
    push(p + "ffn_down.weight", [m.FFN, m.D], Q4_K);
  }
  push("output_norm.weight", [m.D], F32);
  if (!m.tie) push("output.weight", [m.D, m.V], Q4_K);
  if (mutate === "extra") push("blk.0.attn_extra.weight", [m.D, m.D], Q4_K);

  const tb = (t, s) => {
    const n = s.reduce((a, b) => a * b, 1);
    return t === F32 ? n * 4 : t === Q4_K ? (n / 256) * 144 : (n / 256) * 210;
  };
  const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v >>> 0); return b; };
  const u64 = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b; };
  const f32 = (v) => { const b = Buffer.alloc(4); b.writeFloatLE(v); return b; };
  const str = (s) => { const b = Buffer.from(s, "utf8"); return Buffer.concat([u64(b.length), b]); };

  const kv = [
    ["general.architecture", str(m.arch), 8],
    ["general.name", str(mname), 8],
    [m.arch + ".block_count", u32(m.L), 4],
    [m.arch + ".embedding_length", u32(m.D), 4],
    [m.arch + ".attention.head_count", u32(m.NH), 4],
    [m.arch + ".attention.head_count_kv", u32(m.NKV), 4],
    [m.arch + ".attention.key_length", u32(m.HD), 4],
    [m.arch + ".feed_forward_length", u32(m.FFN), 4],
    [m.arch + ".context_length", u32(32768), 4],
    [m.arch + ".rope.freq_base", f32(m.theta), 6],
    [m.arch + ".attention.layer_norm_rms_epsilon", f32(m.eps), 6],
    [m.arch + ".tie_word_embeddings", Buffer.from([m.tie ? 1 : 0]), 7],
  ];

  const parts = [Buffer.from("GGUF"), u32(3), u64(tensors.length), u64(kv.length + 1)];
  for (const [k, v, t] of kv) parts.push(str(k), u32(t), v);
  parts.push(str("tokenizer.ggml.tokens"), u32(9), u32(8), u64(m.V));
  for (let i = 0; i < m.V; i++) parts.push(str("t" + i));

  let off = 0;
  for (const t of tensors) {
    parts.push(str(t.name), u32(t.shape.length));
    for (const d of t.shape) parts.push(u64(d));
    parts.push(u32(t.type), u64(off));
    off = Math.ceil((off + tb(t.type, t.shape)) / 32) * 32;
  }

  const head = Buffer.concat(parts);
  const dataOff = Math.ceil(head.length / 32) * 32;
  return { buf: Buffer.concat([head, Buffer.alloc(dataOff - head.length)]), total: dataOff + off, tensors: tensors.length, name: mname };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [model, out, mutate = "none", name] = process.argv.slice(2);
  const g = buildGguf(model, mutate, name || null);
  fs.writeFileSync(out, g.buf);
  fs.truncateSync(out, g.total);
  console.log(out + "  tensors=" + g.tensors + "  name=" + g.name + "  data=" + (g.total / 2 ** 30).toFixed(3) + " GiB");
}
