# Thesis Results Snapshot

## Best-vs-baseline summary

| dataset | best ours | EM/tokens | best external | EM/tokens | result |
|---|---|---:|---|---:|---|
| musique | force_hard | 0.171/7624 | ma-rag | 0.124/9667 | beats at fewer/equal tokens |
| 2wikimultihop | force_hard_no_critic | 0.308/8138 | ma-rag | 0.218/9416 | beats at fewer/equal tokens |
| hotpotqa | no_critic | 0.332/5097 | naive | 0.323/885 | not dominant |
| bamboogle | force_hard_no_critic | 0.312/4701 | ma-rag | 0.384/6828 | not dominant |

## musique

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.053 | 877.1 | true |
| force_easy | 0.062 | 2379.6 | true |
| opera | 0.106 | 3445.4 | true |
| force_hard_no_critic_ms3 | 0.172 | 6322.0 | true |
| force_hard_no_critic | 0.170 | 6989.1 | false |
| random_route | 0.162 | 7293.8 | false |
| no_critic | 0.159 | 7383.9 | false |
| force_hard | 0.171 | 7623.5 | false |
| ircot | 0.075 | 7895.7 | false |
| adaptive_default | 0.158 | 8003.7 | false |
| ma-rag | 0.124 | 9667.3 | false |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 66 | 0.136 | 0.136 | 0.303 | 2228.6 |
| hard | 934 | 0.160 | 0.057 | 0.162 | 8411.8 |

Scaling slope log(EM)~log(tokens): `0.4756`

| profile contrast | profile | n | delta EM | 95% CI |
|---|---|---:|---:|---|
| adaptive_minus_always_mas | bridge_2hop | 379 | -0.005 | [-0.016, 0.005] |
| adaptive_minus_always_mas | bridge_3hop_plus | 212 | -0.009 | [-0.028, 0.009] |
| adaptive_minus_always_mas | numeric | 68 | 0.000 | [-0.044, 0.044] |
| adaptive_minus_always_mas | one_hop | 86 | -0.046 | [-0.093, -0.012] |
| adaptive_minus_always_mas | parallel_compare | 12 | -0.083 | [-0.250, 0.000] |
| adaptive_minus_always_mas | temporal | 242 | -0.012 | [-0.029, 0.000] |
| adaptive_minus_always_mas | yes_no | 1 | 0.000 | [0.000, 0.000] |
| adaptive_minus_sas | bridge_2hop | 379 | 0.127 | [0.084, 0.169] |
| adaptive_minus_sas | bridge_3hop_plus | 212 | 0.075 | [0.033, 0.123] |
| adaptive_minus_sas | numeric | 68 | 0.118 | [0.029, 0.206] |
| adaptive_minus_sas | one_hop | 86 | 0.000 | [-0.046, 0.046] |
| adaptive_minus_sas | parallel_compare | 12 | 0.083 | [0.000, 0.250] |
| adaptive_minus_sas | temporal | 242 | 0.095 | [0.054, 0.141] |
| adaptive_minus_sas | yes_no | 1 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | bridge_2hop | 379 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | bridge_3hop_plus | 212 | -0.005 | [-0.024, 0.009] |
| critic_effect_default_minus_nocritic | numeric | 68 | 0.000 | [-0.044, 0.044] |
| critic_effect_default_minus_nocritic | one_hop | 86 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | parallel_compare | 12 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | temporal | 242 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | yes_no | 1 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | bridge_2hop | 379 | 0.011 | [0.000, 0.024] |
| ms3_minus_ms5_no_critic | bridge_3hop_plus | 212 | -0.005 | [-0.014, 0.000] |
| ms3_minus_ms5_no_critic | numeric | 68 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | one_hop | 86 | 0.012 | [0.000, 0.035] |
| ms3_minus_ms5_no_critic | parallel_compare | 12 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | temporal | 242 | -0.008 | [-0.021, 0.000] |
| ms3_minus_ms5_no_critic | yes_no | 1 | 0.000 | [0.000, 0.000] |
| parallel_mas_minus_sas | bridge_2hop | 379 | 0.132 | [0.090, 0.174] |
| parallel_mas_minus_sas | bridge_3hop_plus | 212 | 0.085 | [0.038, 0.132] |
| parallel_mas_minus_sas | numeric | 68 | 0.118 | [0.029, 0.206] |
| parallel_mas_minus_sas | one_hop | 86 | 0.046 | [-0.012, 0.116] |
| parallel_mas_minus_sas | parallel_compare | 12 | 0.167 | [0.000, 0.417] |
| parallel_mas_minus_sas | temporal | 242 | 0.107 | [0.066, 0.153] |
| parallel_mas_minus_sas | yes_no | 1 | 0.000 | [0.000, 0.000] |
| router_minus_random | bridge_2hop | 379 | 0.003 | [-0.011, 0.016] |
| router_minus_random | bridge_3hop_plus | 212 | 0.000 | [-0.014, 0.014] |
| router_minus_random | numeric | 68 | 0.015 | [-0.029, 0.073] |
| router_minus_random | one_hop | 86 | -0.046 | [-0.093, -0.012] |
| router_minus_random | parallel_compare | 12 | -0.083 | [-0.250, 0.000] |
| router_minus_random | temporal | 242 | -0.004 | [-0.021, 0.008] |
| router_minus_random | yes_no | 1 | 0.000 | [0.000, 0.000] |

## 2wikimultihop

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.209 | 922.0 | true |
| force_easy | 0.239 | 2299.8 | true |
| opera | 0.082 | 3672.8 | false |
| random_route | 0.275 | 6817.8 | true |
| no_critic | 0.268 | 6846.3 | false |
| force_hard_no_critic_ms3 | 0.303 | 7136.5 | true |
| adaptive_default | 0.268 | 7312.8 | false |
| ircot | 0.159 | 7579.4 | false |
| force_hard_no_critic | 0.308 | 8137.9 | true |
| force_hard | 0.305 | 8852.2 | false |
| ma-rag | 0.218 | 9416.4 | false |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 305 | 0.347 | 0.347 | 0.466 | 2792.2 |
| hard | 695 | 0.233 | 0.191 | 0.234 | 9296.6 |

Scaling slope log(EM)~log(tokens): `0.1636`

| profile contrast | profile | n | delta EM | 95% CI |
|---|---|---:|---:|---|
| adaptive_minus_always_mas | bridge_2hop | 427 | -0.012 | [-0.028, 0.002] |
| adaptive_minus_always_mas | bridge_3hop_plus | 109 | 0.000 | [0.000, 0.000] |
| adaptive_minus_always_mas | numeric | 8 | 0.000 | [0.000, 0.000] |
| adaptive_minus_always_mas | one_hop | 256 | -0.055 | [-0.109, 0.000] |
| adaptive_minus_always_mas | parallel_compare | 69 | -0.058 | [-0.130, 0.000] |
| adaptive_minus_always_mas | temporal | 16 | -0.062 | [-0.188, 0.000] |
| adaptive_minus_always_mas | yes_no | 115 | -0.139 | [-0.235, -0.043] |
| adaptive_minus_sas | bridge_2hop | 427 | 0.028 | [-0.019, 0.075] |
| adaptive_minus_sas | bridge_3hop_plus | 109 | 0.248 | [0.156, 0.339] |
| adaptive_minus_sas | numeric | 8 | 0.375 | [0.125, 0.750] |
| adaptive_minus_sas | one_hop | 256 | 0.008 | [-0.035, 0.051] |
| adaptive_minus_sas | parallel_compare | 69 | -0.275 | [-0.420, -0.130] |
| adaptive_minus_sas | temporal | 16 | 0.250 | [0.062, 0.438] |
| adaptive_minus_sas | yes_no | 115 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | bridge_2hop | 427 | 0.000 | [-0.007, 0.007] |
| critic_effect_default_minus_nocritic | bridge_3hop_plus | 109 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | numeric | 8 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | one_hop | 256 | 0.004 | [-0.008, 0.016] |
| critic_effect_default_minus_nocritic | parallel_compare | 69 | -0.015 | [-0.072, 0.029] |
| critic_effect_default_minus_nocritic | temporal | 16 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | yes_no | 115 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | bridge_2hop | 427 | -0.007 | [-0.019, 0.002] |
| ms3_minus_ms5_no_critic | bridge_3hop_plus | 109 | -0.009 | [-0.028, 0.000] |
| ms3_minus_ms5_no_critic | numeric | 8 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | one_hop | 256 | -0.012 | [-0.031, 0.008] |
| ms3_minus_ms5_no_critic | parallel_compare | 69 | -0.029 | [-0.072, 0.000] |
| ms3_minus_ms5_no_critic | temporal | 16 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | yes_no | 115 | 0.035 | [-0.009, 0.087] |
| parallel_mas_minus_sas | bridge_2hop | 427 | 0.040 | [-0.007, 0.089] |
| parallel_mas_minus_sas | bridge_3hop_plus | 109 | 0.248 | [0.156, 0.339] |
| parallel_mas_minus_sas | numeric | 8 | 0.375 | [0.125, 0.750] |
| parallel_mas_minus_sas | one_hop | 256 | 0.062 | [-0.004, 0.133] |
| parallel_mas_minus_sas | parallel_compare | 69 | -0.217 | [-0.362, -0.058] |
| parallel_mas_minus_sas | temporal | 16 | 0.312 | [0.062, 0.562] |
| parallel_mas_minus_sas | yes_no | 115 | 0.139 | [0.043, 0.235] |
| router_minus_random | bridge_2hop | 427 | 0.009 | [-0.019, 0.037] |
| router_minus_random | bridge_3hop_plus | 109 | 0.073 | [0.009, 0.138] |
| router_minus_random | numeric | 8 | 0.250 | [0.000, 0.625] |
| router_minus_random | one_hop | 256 | -0.039 | [-0.094, 0.012] |
| router_minus_random | parallel_compare | 69 | -0.029 | [-0.116, 0.058] |
| router_minus_random | temporal | 16 | 0.125 | [0.000, 0.312] |
| router_minus_random | yes_no | 115 | -0.096 | [-0.174, -0.026] |

## hotpotqa

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.323 | 885.0 | true |
| force_easy | 0.297 | 1809.6 | false |
| opera | 0.143 | 3069.9 | false |
| random_route | 0.306 | 4775.2 | false |
| no_critic | 0.332 | 5097.0 | true |
| adaptive_default | 0.332 | 5481.0 | false |
| force_hard_no_critic | 0.323 | 6051.1 | false |
| ircot | 0.311 | 6456.6 | false |
| force_hard | 0.328 | 6662.7 | false |
| ma-rag | 0.244 | 8829.5 | false |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 373 | 0.402 | 0.402 | 0.391 | 2290.5 |
| hard | 627 | 0.290 | 0.234 | 0.290 | 7379.0 |

Scaling slope log(EM)~log(tokens): `0.0223`

| profile contrast | profile | n | delta EM | 95% CI |
|---|---|---:|---:|---|
| adaptive_minus_always_mas | bridge_2hop | 400 | 0.028 | [0.005, 0.052] |
| adaptive_minus_always_mas | bridge_3hop_plus | 125 | 0.032 | [0.000, 0.072] |
| adaptive_minus_always_mas | numeric | 43 | 0.000 | [0.000, 0.000] |
| adaptive_minus_always_mas | one_hop | 201 | -0.020 | [-0.065, 0.025] |
| adaptive_minus_always_mas | parallel_compare | 72 | 0.042 | [-0.042, 0.125] |
| adaptive_minus_always_mas | temporal | 89 | 0.000 | [-0.056, 0.056] |
| adaptive_minus_always_mas | yes_no | 70 | -0.071 | [-0.186, 0.029] |
| adaptive_minus_sas | bridge_2hop | 400 | 0.040 | [-0.005, 0.085] |
| adaptive_minus_sas | bridge_3hop_plus | 125 | 0.024 | [-0.056, 0.104] |
| adaptive_minus_sas | numeric | 43 | 0.023 | [-0.046, 0.116] |
| adaptive_minus_sas | one_hop | 201 | 0.035 | [0.005, 0.075] |
| adaptive_minus_sas | parallel_compare | 72 | -0.028 | [-0.111, 0.056] |
| adaptive_minus_sas | temporal | 89 | 0.101 | [-0.011, 0.213] |
| adaptive_minus_sas | yes_no | 70 | 0.014 | [0.000, 0.043] |
| critic_effect_default_minus_nocritic | bridge_2hop | 400 | 0.003 | [-0.005, 0.013] |
| critic_effect_default_minus_nocritic | bridge_3hop_plus | 125 | 0.000 | [-0.024, 0.024] |
| critic_effect_default_minus_nocritic | numeric | 43 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | one_hop | 201 | -0.005 | [-0.020, 0.000] |
| critic_effect_default_minus_nocritic | parallel_compare | 72 | 0.000 | [-0.042, 0.042] |
| critic_effect_default_minus_nocritic | temporal | 89 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | yes_no | 70 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | bridge_2hop | 11 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | bridge_3hop_plus | 5 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | numeric | 3 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | one_hop | 3 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | parallel_compare | 1 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | temporal | 2 | 0.000 | [0.000, 0.000] |
| ms3_minus_ms5_no_critic | yes_no | 3 | 0.000 | [0.000, 0.000] |
| parallel_mas_minus_sas | bridge_2hop | 400 | 0.013 | [-0.040, 0.065] |
| parallel_mas_minus_sas | bridge_3hop_plus | 125 | -0.008 | [-0.096, 0.080] |
| parallel_mas_minus_sas | numeric | 43 | 0.023 | [-0.046, 0.116] |
| parallel_mas_minus_sas | one_hop | 201 | 0.055 | [0.000, 0.110] |
| parallel_mas_minus_sas | parallel_compare | 72 | -0.069 | [-0.167, 0.028] |
| parallel_mas_minus_sas | temporal | 89 | 0.101 | [-0.011, 0.213] |
| parallel_mas_minus_sas | yes_no | 70 | 0.086 | [-0.014, 0.186] |
| router_minus_random | bridge_2hop | 400 | 0.050 | [0.018, 0.085] |
| router_minus_random | bridge_3hop_plus | 125 | 0.008 | [-0.064, 0.072] |
| router_minus_random | numeric | 43 | 0.023 | [-0.046, 0.116] |
| router_minus_random | one_hop | 201 | 0.000 | [-0.040, 0.040] |
| router_minus_random | parallel_compare | 72 | 0.028 | [-0.056, 0.111] |
| router_minus_random | temporal | 89 | 0.045 | [-0.045, 0.135] |
| router_minus_random | yes_no | 70 | -0.029 | [-0.114, 0.057] |

## bamboogle

| method | EM | tokens | Pareto |
|---|---:|---:|---|
| naive | 0.160 | 854.4 | true |
| force_easy | 0.112 | 1742.6 | false |
| opera | 0.264 | 2720.4 | true |
| random_route | 0.272 | 4535.9 | true |
| force_hard_no_critic | 0.312 | 4700.7 | true |
| no_critic | 0.264 | 4746.2 | false |
| adaptive_default | 0.264 | 5163.3 | false |
| force_hard | 0.312 | 5273.1 | false |
| ircot | 0.208 | 6774.5 | false |
| ma-rag | 0.384 | 6827.5 | true |

| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |
|---|---:|---:|---:|---:|---:|
| easy | 22 | 0.227 | 0.227 | 0.500 | 2519.1 |
| hard | 103 | 0.272 | 0.087 | 0.272 | 5728.1 |

Scaling slope log(EM)~log(tokens): `0.3993`

| profile contrast | profile | n | delta EM | 95% CI |
|---|---|---:|---:|---|
| adaptive_minus_always_mas | bridge_2hop | 52 | -0.077 | [-0.154, -0.019] |
| adaptive_minus_always_mas | bridge_3hop_plus | 9 | -0.111 | [-0.333, 0.000] |
| adaptive_minus_always_mas | numeric | 7 | 0.000 | [0.000, 0.000] |
| adaptive_minus_always_mas | one_hop | 27 | 0.000 | [0.000, 0.000] |
| adaptive_minus_always_mas | parallel_compare | 2 | 0.000 | [0.000, 0.000] |
| adaptive_minus_always_mas | temporal | 28 | -0.036 | [-0.143, 0.071] |
| adaptive_minus_sas | bridge_2hop | 52 | 0.192 | [0.096, 0.308] |
| adaptive_minus_sas | bridge_3hop_plus | 9 | 0.222 | [0.000, 0.556] |
| adaptive_minus_sas | numeric | 7 | 0.286 | [0.000, 0.571] |
| adaptive_minus_sas | one_hop | 27 | 0.074 | [0.000, 0.185] |
| adaptive_minus_sas | parallel_compare | 2 | 0.000 | [0.000, 0.000] |
| adaptive_minus_sas | temporal | 28 | 0.107 | [-0.036, 0.286] |
| critic_effect_default_minus_nocritic | bridge_2hop | 52 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | bridge_3hop_plus | 9 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | numeric | 7 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | one_hop | 27 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | parallel_compare | 2 | 0.000 | [0.000, 0.000] |
| critic_effect_default_minus_nocritic | temporal | 28 | 0.000 | [0.000, 0.000] |
| parallel_mas_minus_sas | bridge_2hop | 52 | 0.269 | [0.154, 0.385] |
| parallel_mas_minus_sas | bridge_3hop_plus | 9 | 0.333 | [0.000, 0.667] |
| parallel_mas_minus_sas | numeric | 7 | 0.286 | [0.000, 0.571] |
| parallel_mas_minus_sas | one_hop | 27 | 0.074 | [0.000, 0.185] |
| parallel_mas_minus_sas | parallel_compare | 2 | 0.000 | [0.000, 0.000] |
| parallel_mas_minus_sas | temporal | 28 | 0.143 | [-0.036, 0.357] |
| router_minus_random | bridge_2hop | 52 | -0.019 | [-0.096, 0.058] |
| router_minus_random | bridge_3hop_plus | 9 | 0.000 | [-0.333, 0.333] |
| router_minus_random | numeric | 7 | 0.143 | [0.000, 0.429] |
| router_minus_random | one_hop | 27 | 0.000 | [0.000, 0.000] |
| router_minus_random | parallel_compare | 2 | 0.000 | [0.000, 0.000] |
| router_minus_random | temporal | 28 | -0.036 | [-0.143, 0.071] |
