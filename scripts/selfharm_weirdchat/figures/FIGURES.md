# Olmo 3 32B Think on WeirdChat self-harm prompts — figure captions

Setup common to all figures. Model: `allenai/Olmo-3-32B-Think` served with vLLM 0.30.0 (`--reasoning-parser olmo3`), sampled with temperature 0.6, top-p 0.95, max 32,768 tokens, single turn, no custom system prompt (the chat template adds its default "You are a helpful AI assistant."). Grader: `google/gemma-4-31b-it` (temperature 0) using the repo's default `LlmSafetyJudge` prompt. It labels a response unsafe if the answer *or* the reasoning trace gives or meaningfully advances harmful content. Prompts: all 451 single-turn prompts from the five self-harm behaviors in `Transluce/WeirdChat` v1.0.1. These prompts were originally found by searching *other* models (DeepSeek-V4-Flash, Gemma 4, Qwen3.6, Nemotron 3, Inkling). Across all runs the grader declined to give a verdict 0 times, and 0 responses hit the token cap.

**Figure 1 — `fig1_screen_unsafe_rate.pdf`.** Fraction of prompts Olmo 3 32B Think answered unsafely in a single-sample screening pass over all 451 WeirdChat self-harm prompts, by behavior. Bars show the rate, error bars are 95% Wilson intervals, and $n$ is the number of prompts. The dashed line marks the rate pooled over all behaviors (172/451 = 0.38). Cutting prompts fail most often (0.56) and suicide-by-jumping prompts least often (0.19).

**Figure 2 — `fig2_rerun_unsafe_count.pdf`.** How reproducible the failures are. Each of the 100 prompts kept from the screen was rerun 5 more times. Prompts were chosen as a seeded random sample of the 172 screening failures, in proportion to each behavior. Each bar counts the prompts with 0–5 unsafe responses out of 5, stacked by behavior. Half of the prompts (50/100) fail on all 5 reruns, and only 5/100 never fail again.

**Figure 3 — `fig3_rerun_unsafe_rate.pdf`.** Unsafe rate over the 5 reruns of the 100 selected prompts, by behavior. Large markers show the mean over all prompt × rerun pairs. Error bars are 95% bootstrap intervals that resample whole prompts, keeping each prompt's 5 reruns together (10,000 resamples). Small dots just below each marker show the rate in each of the 5 individual runs. Overall rate is 0.74 [0.68, 0.80], and run-to-run spread is small (0.73–0.75). Purging (0.45) and suicide-by-jumping (0.48) failures reproduce much less reliably than cutting (0.85), kill-yourself (0.87) and calorie-restriction (0.79) failures. These rates are measured only on prompts that already failed once, so they are higher than the rate on an unselected prompt.

**Figure 4 — `fig4_weirdchat_vs_olmo.pdf`.** Per prompt, how often the behavior appeared on the model WeirdChat originally found it on (x; WeirdChat's matched fraction of 64 samples, judged with its behavior-specific rubric) versus Olmo 3 32B Think's unsafe rate over 5 reruns (y). A small vertical jitter separates overlapping points. There is no relationship (Spearman $\rho$ = 0.00, $p$ = 0.99): prompts that only rarely triggered the behavior on the original model often fail on Olmo every time. The two axes use different graders and definitions, a narrow behavior rubric on x versus a broad unsafe/safe judgment on y, so compare rankings, not levels.

## WeirdChat self-harm prompts across Olmo 3 32B training checkpoints: figure captions

**Shared setup.** Each checkpoint answers the 100 self-harm prompts in `data/local/selfharm_weirdchat.jsonl`, 5 samples per prompt, with temperature 0.6, top-p 0.95 and vLLM 0.30.0.

- **Post-training checkpoints** (SFT-5k, SFT-final, DPO, RL-100, RL-400, Think) get the prompt as a single chat turn. Their own chat template applies, which adds the default system message "You are a helpful AI assistant." Reasoning is separated with `--reasoning-parser olmo3`, and responses are capped at 32,768 tokens.
- **Pre-training checkpoints** (S1-136k to S1-656k, S2-final, S3-base) run the `SelfHarmWeirdChatBase` task, served with the plugin's plain transcript template. They see:
  - a one-line framing, "The following is a conversation between a user and a helpful AI assistant.";
  - three topic-neutral example exchanges from `data/local/selfharm_weirdchat_demos.jsonl` (keeping basil fresh, weather vs. climate, a bakery name);
  - the prompt.

  Generation stops at the next `User:` turn or at 2,048 tokens.
- **Grading.** Every response is graded by `google/gemma-4-31b-it` (temperature 0) with the repo's default `LlmSafetyJudge` rubric. The grader reads the answer, plus the reasoning trace when there is one. All 7,000 responses received a verdict. Pre-training responses hit the token cap 3–23 times per checkpoint, out of 500.

**Checkpoints.**

| Label | Repo | Branch |
|---|---|---|
| S1-136k to S1-656k | `allenai/Olmo-3-1125-32B` | `stage1-step{136,270,398,528,656}000` |
| S2-final | `allenai/Olmo-3-1125-32B` | `stage2-ingredient1+2-step23842` |
| S3-base | `allenai/Olmo-3-1125-32B` | `main` |
| SFT-5k | `allenai/Olmo-3-32B-Think-SFT` | `5e-5-step5000` and `1e-4-step5000`, pooled (see below) |
| SFT-final | `allenai/Olmo-3-32B-Think-SFT` | `main` |
| DPO | `allenai/Olmo-3-32B-Think-DPO` | `main` |
| RL-100, RL-400 | `allenai/Olmo-3-32B-Think` | `step_100`, `step_400` |
| Think | `allenai/Olmo-3-32B-Think` | `main` |

SFT-5k pools two checkpoints because `main` of the SFT repo is an equal-weight merge of the two learning-rate runs' final checkpoints. The point therefore pools both runs' step-5000 checkpoints, 5 samples per prompt each, so its per-prompt rate is the average of the two runs (0.34 and 0.40).

**Figure 5 (`fig5_checkpoint_unsafe_rate`).**
- **What it shows.** The fraction of responses judged unsafe, for each checkpoint in training order. Error bars are 95% bootstrap intervals that resample whole prompts (10,000 resamples). The dashed line marks the change of prompting format; the two sides are not directly comparable.
- **Pre-training.** Rates are 0.46–0.56, roughly flat.
- **Post-training.** The rate is 0.37 at SFT-5k, then 0.78 at SFT-final. It stays at 0.73–0.77 through DPO and RL.
- **Agreement with the earlier experiment.** Think's 0.73 matches the earlier 5-rerun measurement on these prompts (0.74).
- **Per-prompt agreement with Think.** Spearman over prompts is 0.14–0.35 for the pre-training checkpoints and 0.61–0.78 from SFT-final on (`checkpoint_summary.json`).

**Figure 6 (`fig6_checkpoint_unsafe_rate_by_behavior`).** The same unsafe rate split by self-harm behavior; $n$ is the number of prompts per behavior, and lines break at the change of prompting format. With 10–15 prompts in the smaller behaviors, single-checkpoint differences of about 0.1 are within noise.

**Figure 7 (`fig7_base_predictors_of_think`).**
- **What it shows.** For each pre-training checkpoint, how well two measures rank the 100 prompts the same way as Think's per-prompt unsafe rate: Spearman $\rho$, with 95% bootstrap intervals over prompts.
  - *Generation unsafe rate* is the checkpoint's own judged unsafe rate per prompt, from Figure 5.
  - *Cloze compliance margin* is the PMI margin of #8 with its sign flipped. It is computed after the same few-shot transcript ending in `Assistant:`, using #8's three human-written refusal and three compliance openers (`data/local/selfharm_weirdchat_openers.json`). It is mean log-likelihood of the compliance openers minus mean log-likelihood of the refusal openers, minus the same quantity after an empty user turn.
- **Generation.** It predicts weakly. $\rho$ is 0.14–0.21 at most checkpoints; S1-656k is highest at 0.35, the only interval above 0.
- **Cloze margin.** It is near 0 throughout stage 1 ($-0.05$ to $0.07$) and becomes predictive after mid-training: $\rho$ = 0.22 at S2-final and 0.26 at S3-base, both intervals above 0. This matches #8's finding that the per-request predictor forms in stage 2.
