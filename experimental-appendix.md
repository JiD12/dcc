# Supplementary experiments for *Delayed Capacity Commitment for Parameter-Efficient Fine-Tuning*

These experiments examine when to impose the deployment budget, what can be recovered by training with a known deployment mask, and how the choice of slot score affects ACP. Unless otherwise specified, the model is Qwen3-0.6B, the deployment budget is 1,568 rank-one slots, and the three training seeds are 42, 123, and 3407. Accuracy is reported in percent. Differences are in percentage points (pp). The ± values give sample standard deviations across seeds. Each comparison uses its own training and evaluation cohort.

## Commitment before the last third of training

The main paper compares commitment at 60%, 80%, and 100% of training. The GSM8K sweep also includes 20% and 40%. All runs continued to update 630, including those that committed at update 126. Each evaluation used all 1,319 questions.

| Commitment | Update | Seed 42 | Seed 123 | Seed 3407 | Mean ± SD |
|---|---:|---:|---:|---:|---:|
| 20% | 126 | 36.92 | 36.24 | 34.65 | 35.94 ± 1.17 |
| 40% | 252 | 37.76 | 37.60 | 35.71 | 37.02 ± 1.14 |
| 60% | 378 | 37.91 | 37.15 | 38.29 | 37.78 ± 0.58 |
| 80% | 504 | 37.07 | 38.44 | 37.00 | 37.50 ± 0.81 |
| 100% | 630 | 40.26 | 38.89 | 37.38 | 38.84 ± 1.44 |

Final-step commitment gained 2.91 ± 0.37 pp over 20% commitment. At intermediate times the ordering varied by seed. Committing early could also change which slots survived.

### Timing with an AdaLoRA allocator

A separate seed-42 control trained AdaLoRA toward rank 9 and deployed an average rank of 8. The 60% arm selected and committed at update 378. In the later arms, allocation ended at update 470 with 1,764 slots. The adapter retained its buffer until update 504 or 630. These were independent training runs.

| Task | Evaluation batch / generation limit | 60% | 80% | 100% |
|---|---:|---:|---:|---:|
| GSM8K | 128 / 344 | 563/1,319 | 556/1,319 | unavailable |
| MATH Level 1–2 | 128 / 1,024 | 53/133 | 48/133 | 46/133 |
| AQuA-RAT | 64 / 2,048 | 74/254 | 77/254 | 69/254 |

On AQuA-RAT the intermediate time scored highest, and on MATH the earliest time scored highest. A batch-4 GSM8K evaluation of the same adapters gave 549 and 560 correct answers at 60% and 80%, respectively. The sign changed across evaluation runs. Only the later MATH pair had verified identical deployment masks among the MATH and AQuA-RAT pairs at 80% and 100%.

## Choosing the commitment step before test evaluation

On EntailmentBank Task 2 we tested a rule that used development negative log-likelihood (NLL) to choose between commitment at updates 144 and 300. For each valid seed, allocation froze at update 120 with 1,809 surviving slots. Two 24-update probes used the same sample order. We recorded their NLL on 48 development examples and locked the decision before evaluating either endpoint on 340 test questions. Both endpoints used the same deployment mask within seed.

| Seed | Probe NLL difference (main − reference) | Selected step | Correct at step 144 | Correct at step 300 |
|---|---:|---:|---:|---:|
| 42 | +0.00222817 | 144 | 47 | 45 |
| 123 | +0.00093220 | 144 | 47 | 45 |
| 3407 | — | — | — | — |

The probe chose the earlier step twice. It matched fixed early commitment (13.82%) and gained 0.59 pp over fixed terminal commitment (13.24%). It cost 24 extra updates and two development passes per valid seed. The third run froze with 1,807 slots and failed the prespecified 1,809-slot requirement. Two valid decisions, both early, give little evidence about when the same rule would favor a later step.

## Training with the final mask from initialization

In a fresh GSM8K cohort, we saved the initial adapter, trained buffered ACP, and extracted its final mask. We then restored the same initialization and trained only those 1,568 selected slots for all 630 updates, using the original training-example order. The deployed masks matched slot by slot across all 196 modules within each pair. The replay had access to a mask chosen by the completed buffered run.

| Seed | Buffered ACP, correct / 1,319 | Fixed-mask replay, correct / 1,319 | Buffered − replay (pp) |
|---|---:|---:|---:|
| 42 | 546 | 477 | +5.23 |
| 123 | 505 | 506 | −0.08 |
| 3407 | 498 | 496 | +0.15 |
| Mean accuracy ± SD; paired difference ± SD | 39.15 ± 1.97% | 37.38 ± 1.12% | +1.77 ± 3.00 |

Seed 42 accounts for 69 of the 70 net additional correct answers. Knowing the final mask did not reproduce that seed's buffered result. The other two pairs differed by at most two answers.

## Buffer size beyond the two main settings

This separate GSM8K sweep compared seven buffer ratios. The dividend subtracts zero-buffer accuracy within seed. Its three-seed means should be read within this sweep.

| Buffer ratio | Accuracy, mean ± SD | Mean dividend (pp) |
|---:|---:|---:|
| 0 | 38.69 ± 0.45 | 0.00 |
| 0.077 | 38.87 ± 1.36 | +0.18 |
| 0.154 | 40.13 ± 1.14 | +1.44 |
| 0.23 | 40.56 ± 0.80 | +1.87 |
| 0.30 | 40.13 ± 1.67 | +1.44 |
| 0.50 | 39.35 ± 0.73 | +0.66 |
| 0.80 | 39.42 ± 1.25 | +0.73 |

All six buffered settings exceeded the zero-buffer mean. The largest tested buffer gained less than half as much as the 0.23 setting.

## Training width and terminal selection

**GSM8K.** In a five-arm control, mean accuracy was 38.62% for immediate commitment and 40.31% for buffered ACP. Training at fixed ranks 9, 24, and 32 with selection only at export gave 37.88%, 40.89%, and 40.86%. All five arms exported 1,568 slots. The original rank-9 and rank-24 arms used different factor scales. A follow-up matched the scale (alpha divided by training rank) at 0.5 and again selected 1,568 slots at export.

| Training rank and scale parameter | Seed 42 | Seed 123 | Seed 3407 | Mean ± SD |
|---|---:|---:|---:|---:|
| Rank 9, alpha 4.5 | 38.29 | 38.51 | 39.27 | 38.69 ± 0.52 |
| Rank 24, alpha 12 | 39.88 | 42.08 | 39.88 | 40.61 ± 1.27 |
| Paired difference (pp) | +1.59 | +3.56 | +0.61 | +1.92 ± 1.51 |

The wider adapter scored higher on each seed at matched scale. Its initialization dimensions and final mask could still differ. In a separate historical evaluation with the original scales (4/9 and 12/24), rank 24 exceeded rank 9 on all five available seeds by 3.05 ± 1.52 pp on average.

**MATH Level 1–2.** We trained DoRA at fixed width and selected eight slots per module by weight norm at export. All compact DoRA arms used 5,390,336 parameters. ACP allocated the same total number of slots across modules and used 5,432,320–5,449,728 compact parameters. All endpoints below used greedy evaluation on 133 questions, batch size 64, and a 1,024-token generation limit.

| Training and export | Correct, seeds 42 / 123 / 3407 | Accuracy, mean ± SD |
|---|---:|---:|
| DoRA rank 8 | 41 / 51 / 49 | 35.34 ± 3.98 |
| DoRA rank 9 → 8 | 54 / 54 / 49 | 39.35 ± 2.17 |
| DoRA rank 24 → 8 | 37 / 27 / 39 | 25.81 ± 4.83 |
| Buffered ACP, ratio 0.154 | 53 / 53 / 53 | 39.85 ± 0.00 |

The rank-9 arm gained 4.01 ± 5.12 pp over direct rank-8 training. ACP exceeded rank 9 → 8 by 0.50 ± 2.17 pp and rank 24 → 8 by 14.04 ± 4.83 pp. As a diagnostic, retaining **all** slots from the same rank-24 checkpoints gave 38.35 ± 2.71%. The mean fraction of outputs reaching the generation limit rose from 1.25% with all slots to 77.44% after rank-8 export. The all-slot diagnostic uses a larger deployment budget.

## Replacing the base score inside full ACP

The main paper replaces the coordinate score with both auxiliary terms disabled. Here the magnitude and trend terms remain enabled. All variants train and export using their own score. The MATH evaluation uses 133 questions per seed.

| Base rule inside full ACP | Correct, seeds 42 / 123 / 3407 | Accuracy, mean ± SD | Original Full − variant (pp) |
|---|---:|---:|---:|
| Original | 53 / 53 / 56 | 40.60 ± 1.30 | — |
| AdaLoRA-style score | 59 / 60 / 55 | 43.61 ± 1.99 | −3.01 ± 3.28 |
| First moment | 50 / 54 / 59 | 40.85 ± 3.39 | −0.25 ± 2.30 |
| Unweighted gradient RMS | 46 / 54 / 59 | 39.85 ± 4.93 | +0.75 ± 3.98 |

The AdaLoRA-style replacement gained 3.01 pp under full ACP. The same replacement lost 7.77 pp under Base-only. Each rule also changed the sequence of budget reductions. The reversal describes the complete training runs under these two scoring configurations.

## Score components on other evaluation cohorts

The following six-arm tests used one seed on GSM8K and BoolQ. A separate three-seed MATH evaluation used batch size 4 and a 1,211-token generation limit. The main paper's scoring table uses batch size 64. Each column below should be compared against its own Full row.

| Configuration | GSM8K, seed 42 | BoolQ, seed 42 | MATH, three-seed mean |
|---|---:|---:|---:|
| Full ACP | 48.98 | 82.87 | 41.10 |
| Without magnitude score | 49.28 | 82.81 | 42.11 |
| Without trend score | 50.27 | 83.12 | 39.85 |
| Base-only | 49.20 | 83.24 | 42.86 |
| Without score smoothing | 48.37 | 83.06 | 42.61 |
| Prescribed schedule | 49.36 | 82.87 | 41.85 |

Base-only exceeded Full in all three columns. Removing the trend term helped on the single GSM8K seed and reduced the three-seed MATH mean. In the same GSM8K seed-42 evaluation, rank-8 DoRA scored 50.11%.

## Additional baseline comparisons

Two seed-42 evaluations extend the task coverage of the five-task method comparison. The tables give percent accuracy and report all four methods within each cohort.

| Numeric reasoning | Buffered ACP | DoRA | AdaLoRA | LoRA |
|---|---:|---:|---:|---:|
| TabMWP | 85.09 | 87.08 | 69.91 | 88.15 |
| Omni-MATH | 27.08 | 18.75 | 22.92 | 27.08 |
| iGSM | 50.77 | 71.94 | 14.80 | 72.70 |
| Three-task equal-weight mean | 54.31 | 59.26 | 35.87 | 62.65 |
| SVAMP (descriptive) | 68.67 | 72.67 | 2.67 | 71.33 |
| MAWPS (descriptive) | 92.31 | 95.96 | 74.81 | 93.46 |

ACP matched LoRA at 13/48 on Omni-MATH. LoRA led the three-task mean. The original protocol excluded the two short runs (SVAMP and MAWPS) from this mean because their frozen pools missed the target of 1,809 slots. TabMWP missed the target by two slots and was included under the original short-run rule. All five ACP runs deployed 1,568 slots.

| Commonsense reasoning | Buffered ACP | DoRA | AdaLoRA | LoRA |
|---|---:|---:|---:|---:|
| BoolQ | 83.00 | 84.95 | 21.83 | 84.95 |
| PIQA | 74.10 | 75.95 | 15.67 | 75.14 |
| ARC-Easy | 81.19 | 81.86 | 68.48 | 81.86 |
| ARC-Challenge | 60.92 | 62.20 | 59.13 | 62.20 |
| Four-task equal-weight mean | 74.80 | 76.24 | 41.28 | 76.04 |

ACP exceeded AdaLoRA on all four tasks. DoRA and LoRA both scored higher than ACP on this seed. The especially low AdaLoRA means on BoolQ and PIQA include 1,171/1,635 and 1,388/1,838 outputs, respectively, that reached the generation limit.

### Generation-length sensitivity

We re-evaluated the same ACP, DoRA, and AdaLoRA adapters with longer generation limits on EntailmentBank Task 2 (448 to 896 tokens) and MathQA (1,051 to 2,048 tokens). All 5,760 item-level correctness outcomes were unchanged across the 18 re-evaluations. Mean accuracy on MathQA was 46.56% for ACP, 50.56% for DoRA, and 37.00% for AdaLoRA. The corresponding EntailmentBank means were 13.33%, 15.20%, and 2.35%. ACP exceeded AdaLoRA in all three seeds on both tasks. None of the 1,227 outputs that had hit the original generation limit recovered a correct answer at the higher limit.
