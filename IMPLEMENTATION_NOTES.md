# Competitive Task-Relevant Object Tokens V2

## Why V2 exists

The first task-object-token experiment learned a strong policy late in training,
but its internal object representation collapsed:

```text
one dominant task-relevant patch
  -> all object queries attend to the same patch
  -> nearly identical object tokens
```

The measured attention overlap stayed almost exactly 1. The cause was
structural: each object independently softmaxed over patches, a strong
`beta log relevance` term dominated query differences, and the mean of all
objects was aligned to one global MineCLIP embedding.

V2 removes those three collapse mechanisms.

## Candidate prior

For RGB patches `p_i`, frozen task embedding `g`, and frozen video semantic
`m_t`:

```text
u_t = LN(W_g stopgrad(g) + W_m stopgrad(m_t))
r_i = softmax(cos(LN(W_p p_i + pos_i), u_t) / tau_r)
```

The top six patches form the candidate set. Their normalized relevance prior is
softened:

```text
prior_i = (1 - mu) * relevant_i + mu * uniform_i
mu = 0.5
```

This prevents a single patch with relevance near 1 from monopolizing every
object.

## Competitive balanced assignment

For object query `q_k` and candidate patch `k_i`:

```text
L_{k,i} = cos(q_k, k_i) / tau_o + beta log(prior_i)
beta = 0.25
```

A log-space Sinkhorn procedure converts these logits to an assignment matrix
`X` with approximate marginals:

```text
sum_i X_{k,i} = 1 / K
sum_k X_{k,i} = prior_i
```

Object attention is the row-normalized matrix:

```text
A_{k,i} = X_{k,i} / sum_j X_{k,j}
```

Patch-to-object competition is the column-normalized matrix:

```text
C_{k,i} = X_{k,i} / sum_j X_{j,i}
```

Thus patches compete across objects before each object aggregates its own
candidate distribution.

## Anti-collapse objectives

V2 uses three low-weight terms:

1. Competition entropy: minimize the object-distribution entropy for each
   candidate patch, so a patch chooses a specific object.
2. Attention diversity: discourage the two object attention maps from being
   identical.
3. Feature diversity: discourage the resulting object embeddings from becoming
   identical even when their attention differs.

Sinkhorn row marginals already balance total object usage, preventing the
low-entropy objective from assigning every patch to one object.

## Global-only semantic alignment

V1 aligned the mean of all object tokens to the same MineCLIP video embedding,
which encouraged identical objects. V2 instead forms a separate task-relevant
global scene token:

```text
global_t = sum_i prior_i * value(p_i)
```

Only this token is aligned with the detached MineCLIP video semantic:

```text
L_global-sem = 1 - cosine(W_global global_t, stopgrad(m_t))
```

Individual objects are not pulled toward a shared global target.

## Conservative first diagnostic

V2 starts with two objects, six candidates, and a smaller residual scale. This
is intentional. The first requirement is to demonstrate stable object
specialization. Expanding to four objects is justified only after overlap and
attention visualizations confirm genuine separation.
