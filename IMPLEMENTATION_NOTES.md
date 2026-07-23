# Task-Conditioned Patch Tokens: implementation notes

## Controlled comparison

This branch differs from the plain DreamerV3 + MineCLIP reward baseline only in
representation learning. It does not use the global MineCLIP video embedding.

- Baseline visual stream: trainable Dreamer CNN.
- Task condition: frozen 512-D MineCLIP text embedding, computed once per task.
- Spatial tokens: final 4x4 CNN feature map (16 tokens, 768 dimensions with the
  default `cnn_depth: 96`).
- Selection: cosine cross-attention between projected task and patch tokens.
- Top-K: four patches by default.
- Fusion: gated residual applied only to selected tokens.
- RSSM input dimension: unchanged.
- Reward: unchanged `environment + MineCLIP`.

## Equations

Let `p_i` be the i-th RGB patch token and `g` the frozen task text embedding:

    q = LN(W_g stopgrad(g))
    k_i = LN(W_p p_i + pos_i)
    alpha_i = softmax(cos(q, k_i) / tau)

Only the top-K `alpha_i` remain active and are renormalized. The token update is:

    p_i' = p_i + rho * K * alpha_i * W_o[
        sigmoid(G([k_i, q])) * V(k_i)
    ]

All refined and untouched tokens are flattened exactly as in the Dreamer CNN.

## Initial conservative behavior

`output_init: 0.001`, `residual_scale: 0.05`, and gate bias `-2` make the
initial encoder close to the RGB baseline. This is deliberate to reduce the risk
that untrained patch selection destabilizes RSSM learning.
