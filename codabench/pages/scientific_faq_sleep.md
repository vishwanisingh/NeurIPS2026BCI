# Scientific FAQ · Track 03

This page answers recurring scientific questions about the model-facing
contract. The **Track description** defines the phase specifications; the
public [`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/objective.py)
and active phase configuration are the executable references. Packaging and
upload instructions are in **Get Started → Submission Guide**.

## Warm-up and sealed evaluation

### What changed, and what remains the final contract?

| | Warm-up (previous) | Warm-up (current) | Sealed final |
|---|---|---|---|
| **Evaluation data** | Public Sleep-EDF test participants | 40 public Muse test recordings | Private Muse recordings from represented and unseen sleepers |
| **Input and delivery** | Two bipolar channels; batched windows from a fixed pre-N2 crop | Four Muse channels at native 128 Hz; one chronological five-second window at a time, from recording start to first N2 | The same model-facing streaming contract |
| **Ranking metric** | Unweighted bMAE | W-bMAE per recording, averaged over recordings | W-bMAE for seen and unseen groups, macro-averaged |

Warm-up scores are indicative only. Old Sleep-EDF submissions must be rebuilt
for the four-channel streamed Muse contract.

### What generalization does the sealed phase test?

It combines new recordings from sleepers represented in public training data
with recordings from sleepers absent from it. Exact cohort size, identities,
group balance, recording lengths, and evaluation order remain private. The
public [competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/datasets/interaxon2026muse.py)
shows the current warm-up data path; sealed data remain worker-side.

## Public data, target, and preprocessing

### What is in the public Muse release?

[NEMAR `nm000287`](https://nemar.org/dataset/nm000287) contains 540 at-home
recordings from 203 participants, totaling about 157.5 hours. Recordings use
TP9, AF7, AF8, and TP10 at 128 Hz and include one first-N2 annotation and
signal-quality flags. Its supplied split has 500 training recordings and 40
test recordings; every warm-up test participant is represented in training.
The release is licensed CC BY-NC-SA 4.0. NeuralBench encodes the supplied split
in its [Muse dataset config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/sleep_onset/datasets/interaxon2026muse.yaml).

### May NSRR or other public sleep datasets be used?

Yes. NSRR datasets under an active licence and other public datasets may be
used when their original terms permit this use. Declare every external corpus
in the final method description for the reproducibility audit, and follow the
competition [Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md).
Private sealed data and non-permitted closed clinical datasets may not be used.

### What exactly is the target?

The supplied annotation is the earliest epoch labelled N2; the competition
does not require a full hypnogram or impose an additional multi-epoch rule.
For a window ending at time `t`, the target is `first_N2 - t` in seconds,
clipped to `[0, 600]`. The target construction is public in the
[NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_sleep_onset_stream/config.yaml).

### What reaches `predict(X)`?

Recordings stay at their native 128 Hz and are cut into contiguous,
non-overlapping five-second windows. No filter, notch filter, learned scaling,
or clipping is applied; signals are expressed in microvolts. Each evaluation
input has shape `(1, 4, 640)`. These settings are defined by the
[NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_sleep_onset_stream/config.yaml)
and invoked by the [competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/datasets/interaxon2026muse.py).

“Causal” refers to the model-facing contract: the model receives only the
current and previous windows of its current recording.

## Streaming and adaptation

### Where does a stream start and end?

Warm-up and sealed evaluation start at the beginning of each selected
recording and stop at first N2. There is no shuffling within a recording:
windows arrive one at a time in strict chronological order. There is no random
stream start and no post-onset window. The order of independent recordings is
not exposed as a model signal; the recording duration and future endpoint are
also not passed to the model. The source settings are visible in the
[NeuralBench config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_sleep_onset_stream/config.yaml).

### How is a new recording signaled, and may a model retain state?

Each recording starts from a fresh copy of the submitted model and calls its
optional `reset_state()`. A model may retain recurrent state, summaries,
elapsed-window count, or past windows within the current recording, subject to
the worker memory limit. All state must be cleared at the boundary. The public
[streaming evaluator](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/streaming.py)
implements the one-window calls, model copy, and reset.

### What metadata does the model receive?

`predict(X)` receives only the EEG tensor. Sleeper and recording identifiers,
seen/unseen group, labels, and future context are retained by the evaluator and
not passed to the model. Time elapsed since the stream began may be inferred
from the number of windows already received and is valid causal context.

### What must the model return?

One finite floating-point latency in seconds per input window, with shape
`(B,)`. During streamed evaluation `B = 1`. The public
[objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/objective.py)
validates one prediction per window.

## Scoring

### How is current warm-up W-bMAE computed?

Within each recording, MAE is computed in the true-target ranges **[0, 40)**,
**[40, 90)**, **[90, 300)**, and **[300, 600]** seconds. Non-empty ranges are
combined as `sum(weight × MAE) / sum(weight)` with weights **10, 5, 3, and 1**.
Recording scores are then averaged equally. NeuralBench's
`test/wbmae_stream_mean` is the same warm-up score. Unweighted bMAE and pooled
MAE are diagnostics. See the public [objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/objective.py)
and [metric implementation](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py).

### How does sealed scoring differ?

Codabench computes W-bMAE separately for recordings from seen and unseen
sleepers, then macro-averages the two group scores. This gives equal importance
to night-to-night and inter-person transfer.

## Resources

**[Track description](https://www.codabench.org/competitions/17983/)** ·
**[Muse data on NEMAR](https://nemar.org/dataset/nm000287)** ·
**[NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)** ·
**[Competition implementation](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/sleep_onset)**
