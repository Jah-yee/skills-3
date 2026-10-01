# Propose a plan; accept a veto

Before any code, write a short paragraph stating what you'd do by default,
then wait for the user to confirm or veto. You know more than the user
told you at this point (you read the config), so do not ask blank
questions; state a default and let them push back.

The paragraph should cover each of these, derived from what you've
already read:

- **API.** ModuleV3, always, for a new port. Name any piece ModuleV3
  lacks that the model needs (a layer, a distributed primitive, a
  quantized path) and whether you will add it or need the user to decide;
  do not propose V2 as the workaround. The
  [migrate-max-v2-to-v3](../../migrate-max-v2-to-v3/SKILL.md) skill holds
  the concept map.
- **Distribution shape.** Single-GPU dense, multi-GPU tensor-parallel,
  multi-GPU DP+EP (MoE), or text-only port of a multimodal wrapper. The
  rough rule: estimate ``num_params × bytes_per_param`` (with the MoE
  factor when applicable) and compare to one GPU's HBM minus cushion
  for KV cache + activations. If it doesn't fit single-GPU, you are
  multi-GPU and your donor will be from the distributed-transformer
  family; see [distributed-transformer.md](distributed-transformer.md).
- **Quantization variants in scope.** BF16 only, or also FP8 / NVFP4 /
  GPTQ. Check the HF org for sibling repos. For any non-BF16, pre-flight
  the MAX kernel walls in [recognize-walls.md](recognize-walls.md);
  flag a wall up front rather than after building the plumbing.
- **Validation depth.** Generation smoke only, GSM8K-style generation
  eval, lm-eval-harness loglikelihood (hellaswag/mmlu), or formal HF
  logit parity. See [validation-tiers.md](validation-tiers.md) for the
  tiers.
- **Hardware target.** Which machine(s) you'll serve on. Required for
  multi-GPU; confirm capacity before downloading large checkpoints.

The paragraph reads like a competent colleague proposing a plan, not
an interviewer collecting requirements. One example shape:

> ``org/LargeMoE-70B-bf16``, a MoE causal LM, text-only (vision wrapper
> dropped). I'll scaffold from the kimik2_5_modulev3 TP + EP pattern on 4 GPUs,
> register BF16 only for now (FP8 sibling repo exists but MoE quant has a
> known MAX gap), and validate with greedy generation + GSM8K + HellaSwag.
> Go?

The user can say "go" and they're done. Or veto one axis ("just BF16
for now"); three words, no form. Proceed only after they confirm.

If the user already specified any of these in their initial request,
take it as given. Don't re-ask.

After they confirm, state the routing decision in one sentence
("I'll start from the kimik2_5_modulev3 TP + EP pattern, not the olmo3
single-GPU donor, because the BF16 weights need ~400 GB HBM"). That
gives the user one more chance to catch wrong routing before files
get rewritten.
