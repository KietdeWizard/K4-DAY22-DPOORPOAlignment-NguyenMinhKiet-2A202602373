# Lab 22 reflection — DPO/ORPO alignment

**Student:** Nguyen Minh Kiet  
**Tier/date:** Colab T4, 2026-10-09

## 1. Configuration

| Item | Value |
|---|---|
| Base model | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| SFT data | Vietnamese instruction data, 1,000 examples |
| Preference data | `sailor2/sea-ultrafeedback-onpolicy`, 800 train / 100 held-out |
| Chosen longer fraction | 0.59625 (59.625%) |
| DPO beta / learning rate / epochs | 0.1 / 5e-6 / 1 |
| Judge | `Skywork/Skywork-Reward-V2-Llama-3.2-3B`, sanity 100% |

## 2. DPO result

| Metric | Measured value |
|---|---:|
| First logged train loss | 0.692443 |
| Final train loss | 0.679868 |
| Train chosen / rejected reward | 0.217943 / 0.175484 |
| Train reward gap | 0.042459 |
| Held-out chosen / rejected reward | 0.252922 / 0.192960 |
| Held-out reward gap | 0.059963 |
| Held-out reward accuracy | 0.71 |
| Diagnosis | INTENDED |
| Mean response length SFT → DPO | 463.32 → 437.16 characters (held-out) |

## 3. Reading the reward curves

The measured DPO loss decreased from 0.692443 to 0.679868, while the train chosen reward reached 0.217943 and the rejected reward reached 0.175484. The resulting train gap was 0.042459. On held-out data, chosen and rejected rewards were 0.252922 and 0.192960, giving a larger gap of 0.059963 and reward accuracy 0.71. Thus chosen moved upward and rejected also moved upward, but chosen moved farther relative to the reference. This is consistent with the automatic `INTENDED` diagnosis rather than likelihood displacement: there is no evidence here that chosen probability fell while rejected fell faster. The held-out gap being positive and larger than the train gap is also reassuring, although reward accuracy is not perfect and should not be treated as a quality score by itself. The initial loss was close to log(2), as expected when policy and reference start equal. In general, a margin can rise even when chosen probability falls if rejected probability falls faster; that is why both reward curves must be inspected separately.

## 4. SFT versus SFT+DPO

| Group | n | DPO wins | SFT wins | Ties | Win rate (95% CI) | Length-matched win rate | Longer answer won |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 1 | 43 | 0.550 [0.500, 0.600] | 0.533 (n=46) | 0.429 |
| helpfulness | 4 | 0 | 1 | 3 | 0.375 [0.125, 0.500] | 0.375 | 0.000 |
| safety | 4 | 1 | 1 | 2 | 0.500 [0.125, 0.875] | 0.500 | 1.000 |

The sanity accuracy was 1.00. Only the Llama reward model could fit after the Qwen3 reward model caused CUDA OOM, so there is no valid two-judge agreement statistic; this is a limitation, not evidence of agreement. The held-out CI includes 0.5, so the run does not establish that DPO is better than SFT. The mean answer length actually decreased from 463.32 to 437.16 characters, and the length-matched win rate was 0.533, so the result is not explained simply by longer answers. For a usefulness example, the quicksort answer was essentially equivalent between SFT and DPO, with DPO slightly shorter and more concise. For safety, both versions refused the request for explosive-making instructions; DPO used slightly different wording but preserved the refusal. These examples fit the high tie count and the cautious conclusion.

## 5. Bonus

Not run.

## 6. One important decision

I chose beta = 0.1 with one epoch and a learning rate of 5e-6. The main alternative would have been a beta sweep, for example 0.05, 0.1, and 0.5, or an RPO-style objective that adds a chosen-answer NLL term. I kept beta at 0.1 because it is the lab baseline and is a reasonable compromise between learning the preference margin and staying near the SFT reference. A smaller beta would permit a stronger policy shift and might increase the reward gap, but it would also raise the risk of verbosity, formatting drift, or loss of useful SFT behavior. A larger beta would be more conservative and might leave the model nearly unchanged. The measured run reduced loss, produced positive train and held-out reward gaps, and received the intended diagnosis. The held-out comparison was modest: DPO won 6 of 50 non-tied cases, SFT won 1, and 43 were ties; the confidence interval still contains 0.5. That result is useful because it prevents overclaiming. If I repeated the experiment, I would first run the beta sweep and add a second, non-Skywork judge on a larger GPU. I would also keep the T4-safe sequence length and explicitly monitor response length, because the preference data had chosen longer in 59.625% of pairs even though the final DPO answers were shorter on held-out prompts.

## 7–9. Bonus sections

Not run.
