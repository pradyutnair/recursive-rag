# Thesis Results Snapshot

## Best-vs-baseline summary

| dataset | best ours | EM/tokens | best external | EM/tokens | result |
|---|---|---:|---|---:|---|
| musique | force_hard | 0.171/7624 | ma-rag | 0.124/9667 | beats at fewer/equal tokens |
| 2wikimultihop | force_hard | 0.305/8852 | ma-rag | 0.218/9416 | beats at fewer/equal tokens |
| hotpotqa | no_critic | 0.332/5097 | naive | 0.323/885 | not dominant |
| bamboogle | force_hard | 0.312/5273 | ma-rag | 0.384/6828 | not dominant |

## musique

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.053 | 877.1 | true |
| force_easy | 0.062 | 2379.6 | true |
| opera | 0.106 | 3445.4 | true |
| force_hard_no_critic | 0.170 | 6989.1 | true |
| no_critic | 0.159 | 7383.9 | false |
| force_hard | 0.171 | 7623.5 | true |
| ircot | 0.075 | 7895.7 | false |
| adaptive_default | 0.158 | 8003.7 | false |
| ma-rag | 0.124 | 9667.3 | false |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 66 | 0.136 | 0.136 | 0.303 | 2228.6 |
| hard | 934 | 0.160 | 0.057 | 0.162 | 8411.8 |

Scaling slope log(EM)~log(tokens): `0.4498`

## 2wikimultihop

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.209 | 922.0 | true |
| force_easy | 0.239 | 2299.8 | true |
| opera | 0.082 | 3672.8 | false |
| no_critic | 0.268 | 6846.3 | true |
| adaptive_default | 0.268 | 7312.8 | false |
| ircot | 0.159 | 7579.4 | false |
| force_hard | 0.305 | 8852.2 | true |
| ma-rag | 0.218 | 9416.4 | false |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 305 | 0.347 | 0.347 | 0.466 | 2792.2 |
| hard | 695 | 0.233 | 0.191 | 0.234 | 9296.6 |

Scaling slope log(EM)~log(tokens): `0.102`

## hotpotqa

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.323 | 885.0 | true |
| force_easy | 0.297 | 1809.6 | false |
| opera | 0.143 | 3069.9 | false |
| no_critic | 0.332 | 5097.0 | true |
| adaptive_default | 0.332 | 5481.0 | false |
| ircot | 0.311 | 6456.6 | false |
| force_hard | 0.328 | 6662.7 | false |
| ma-rag | 0.244 | 8829.5 | false |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 373 | 0.402 | 0.402 | 0.391 | 2290.5 |
| hard | 627 | 0.290 | 0.234 | 0.290 | 7379.0 |

Scaling slope log(EM)~log(tokens): `0.0076`

## bamboogle

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.160 | 854.4 | true |
| force_easy | 0.112 | 1742.6 | false |
| opera | 0.264 | 2720.4 | true |
| no_critic | 0.264 | 4746.2 | false |
| adaptive_default | 0.264 | 5163.3 | false |
| force_hard | 0.312 | 5273.1 | true |
| ircot | 0.208 | 6774.5 | false |
| ma-rag | 0.384 | 6827.5 | true |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 22 | 0.227 | 0.227 | 0.500 | 2519.1 |
| hard | 103 | 0.272 | 0.087 | 0.272 | 5728.1 |

Scaling slope log(EM)~log(tokens): `0.3849`
