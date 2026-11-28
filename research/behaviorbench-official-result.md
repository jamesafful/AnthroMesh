# Official BehaviorBench result

The official test ran once after the signed freeze tag was published.
The primary mixed-history cohort contained 9,145 decisions from 998
wallets. All four gates were required. One passed.

| Frozen gate | Observed result | Decision |
| --- | --- | --- |
| 25% matched coverage, improvement over recent-five frequency | Pooled error 28.39% versus 31.28%. Reduction 2.89 points, paired 97.5% wallet interval +0.41 to +5.39 | Pass |
| 10% matched coverage, rule out a loss greater than 2 points | Pooled error 23.09% versus 24.51%. Reduction 1.42 points, interval -2.55 to +5.67 | Fail |
| 15% error cap at the 10%-targeted cutoff | 23.06% error at 9.86% coverage, 95% upper bound 25.86% | Fail |
| 20% error cap at the 25%-targeted cutoff | 28.25% error at 24.08% coverage, 95% upper bound 30.13% | Fail |

The pooled model improved ranking at the 25% operating point. Its
validation-selected cutoffs exceeded both absolute error caps. The
result provides no basis to replace the runtime decision gate.
A capable direct-history LLM comparison was unrun under the frozen
protocol. Wallet stances are the scope of the evaluation.

The [aggregate result](behaviorbench-official-results.json) includes
every evaluated coverage point and the full-population secondary results.
The [signed freeze](https://github.com/jamesafful/AnthroMesh/tree/research/behaviorbench-official-freeze-v1)
contains the fitted weights, cutoffs, hashes, and scorer. The BehaviorBench
study is closed following the one official run.
