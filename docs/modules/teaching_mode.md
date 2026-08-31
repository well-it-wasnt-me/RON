# Teaching Mode — the human teaching & learning loop

DeskBot's **teaching mode** lets a developer teach the robot a new
behaviour by demonstration and real human feedback, on-device, with no
external inference and no synthetic training data. This is the loop the
local-brain [learning module](learning.md) targets end to end.

```
developer: "RON, when I wave, wave back"
repeat:
    human waves  ->  RON waves  ->  human says "Good"
    (a real transition is recorded; praise is attributed post-hoc)
after enough repetitions: Q(state, wave) > Q(state, unrelated)
```

---

## Enabling

Teaching is **opt-in** and requires the learning system to be on:

```bash
DESKBOT_LEARNING__ENABLED=true
DESKBOT_TEACHING__ENABLED=true
```

Teaching is a **context flag** on the state vector, *not* a `RobotState`.
This keeps `STATE_SIZE` at 91 (and the multimodal vector at 570) — teaching
repurposes reserved slots instead of resizing the vector. See
[State-size decision](#state-size-decision) below.

All teaching parameters are environment-configurable via
`DESKBOT_TEACHING__*` (see `.env.example`):

| Variable                                    | Default | Description                                                       |
|---------------------------------------------|---------|-------------------------------------------------------------------|
| `DESKBOT_TEACHING__ENABLED`                 | `false` | Enable the teaching loop                                          |
| `DESKBOT_TEACHING__FEEDBACK_WINDOW_S`        | `5.0`   | Seconds after a transition during which feedback can attach to it |
| `DESKBOT_TEACHING__STALENESS_S`             | `30.0`  | Stored on `FeedbackService` as a staleness bound; **not currently enforced** — `reward_for_transition` applies attributed feedback regardless of age |
| `DESKBOT_TEACHING__PRACTICE_EPSILON`        | `0.2`   | Defined but **not currently wired** — the policy uses its own epsilon-greedy decay schedule. Reserved for future use |
| `DESKBOT_TEACHING__COOLDOWN_S`              | `0.2`   | Minimum seconds between two executed actions (safety cooldown)   |
| `DESKBOT_TEACHING__MIN_EXPERIENCES_FOR_PRACTICE` | `64` | Min total experiences before practice may let the policy propose  |

---

## The flow

1. **Arm a session from a spoken instruction.** The developer says (or
   types, or POSTs) a constrained instruction of the form
   `"when I {gesture}, {action}"` — e.g. `"when I wave, wave back"`. The
   constrained parser
   ([`parse_teaching_instruction`](https://github.com/well-it-wasnt-me/RON/tree/main/src/robot/learning/teaching_parser.py))
   resolves the action against the registered action-space names
   (multi-word spoken forms matched longest-first) and arms a
   `TeachingController` session in `demonstrate` (or `practice`) mode.
   **The LLM never decides what action the robot should learn.** A
   non-teaching utterance does not arm a session and falls through to the
   normal conversation turn.
2. **Inject the gesture.** A `GestureDetected` event whose gesture matches
   the spec's trigger fires. `TeachingController.on_gesture_detected`
   opens an **interaction** and either:
   - **demonstrates** — executes the human-specified desired action
     through the canonical `ActionExecutor` (the single learning recording
     point), recording a **real** transition tagged with the teaching
     session id + interaction id; or
   - in **practice** mode, asks the policy (`ActionLearner`) to propose an
     action and executes *that* — gated by the `SafetyGate`.
3. **Give feedback.** A spoken `"good"` / `"no"` (or an explicit
   `POST /api/v1/teaching/feedback`) is read as praise / correction. The
   `FeedbackService` attributes it to the most-recent **eligible** real
   transition (within `feedback_window_s`), recording a `FeedbackLedger`
   entry. Feedback is **never invented**: if no recent transition is
   eligible, it is dropped.
4. **Learn.** In each background training cycle, the `ActionLearner`
   trains on the feedback-amended reward (`reward_for_transition` = base
   reward + post-hoc ledger praise/correction). After enough repetitions,
   `Q(state, wave)` rises above an unrelated action. Nothing hard-codes
   `wave` as the answer — the only signal that wave is "good" is the
   human's praise.

The whole loop is exercised end to end by
[`tests/integration/test_teaching_e2e.py`](https://github.com/well-it-wasnt-me/RON/tree/main/tests/integration/test_teaching_e2e.py),
driven entirely by real events on the real event bus.

---

## Synthetic gesture channel (limitation, by design)

**DeskBot has no built-in computer-vision gesture / hand detector.**
Gesture is therefore a **synthetic channel**: a `GestureDetected` event
is produced by one of these injection paths rather than a CV model —

- the **teaching REST API** (`POST /api/v1/teaching/demonstration` with a
  `gesture` field — used by the dashboard and external tools),
- the **CLI**,
- the **constrained speech parser** (the instruction itself), or
- a **test**.

`GestureDetected` is an **observation, never an action**: it updates the
state encoder's gesture one-hot (and `person_present`), but it never
creates a transition. Only the `ActionExecutor` executing a real action
creates a transition. Building a real CV gesture detector is out of scope.

---

## Feedback semantics

- **Post-hoc, last-wins, recency-gated.** The `FeedbackService` scans
  `recorder.working_memory.recent(20)` for the most-recent transition
  within `feedback_window_s` of now, preferring one from the same
  interaction id. The most-recent (last-executed) eligible transition wins.
  When the caller passes no `interaction_id`, the most-recent eligible
  transition overall is used.
- **Never invented.** If no recent transition is eligible, the feedback
  is dropped with a log line and `None` is returned — no transition is
  fabricated, no reward is invented, no counter is bumped.
- **Amended reward.** `LearningService.reward_for_transition(tid)` returns
  the immediate `RewardModel` reward **plus** the ledger's
  praise/correction delta, clamped to `[-2, 2]` (and `0.0` when the
  transition is no longer in `working_memory.recent(256)`). The
  `staleness_s` bound is **not currently enforced** — attributed feedback
  is applied regardless of age. The action learner trains on this amended
  reward — so sparse base rewards (mostly 0 without face/audio) are
  dominated by the human feedback signal, by design.
- **Polarity / magnitude.** Praise is `polarity=+1`, correction `polarity=-1`,
  with a configurable `magnitude` (default `1.0`).

---

## Safety

Learning **never bypasses safety.** The mechanisms are preserved and
extended, never weakened:

- Every executed action — demonstrated or policy-proposed — flows through
  the **same canonical `ActionExecutor`**, the single point that records
  real transitions and drives hardware. The `TeachingController` never
  writes to hardware or the replay buffer directly.
- **Practice proposals** pass a **non-mutating** `SafetyGate.is_valid`
  check during `select_action`'s candidate loop, then are **re-validated**
  with the full, mutating `SafetyGate.validate` once, immediately before
  execution. An out-of-range or rate-limited proposal is rejected and
  becomes a no-op (the controller never raises a `ServoError`).
  `SafetyGate` also checks servo target angles against `servo_limits`
  (defense-in-depth).
- The `ActionValidator`, hardware safety limits, candidate evaluation,
  rollback, promotion, checkpointing, and resource limits all remain in
  force.
- The **LLM never directly creates learned actuator commands or arbitrary
  reward values.** It can only route through the executor / parser; the
  constrained parser (not the LLM) decides what action a teaching session
  targets.

---

## State-size decision

Teaching is a **context flag on the state vector**, deliberately **not**
a new `RobotState`. Adding a `TEACHING` state would shift the 8-wide
robot-state one-hot (slots 10–17) and break the 570-dim multimodal vector.
Instead, the former zero-filled reserved block `[51..61)` was repurposed
(`ENCODER_VERSION = 2`):

| Slot | Field              | Meaning                                            |
|------|--------------------|----------------------------------------------------|
| 51   | `teaching_context` | 0/1 — a teaching session is armed                   |
| 52   | `interaction_active` | 0/1 — inside a teaching interaction/episode       |
| 53   | `person_present`   | 0/1 — a face is present                             |
| 54–58 | `gesture`         | one-hot: none / wave / point / open_hand / other   |
| 59   | `conversation_turn` | recent turn count (normalised)                     |
| 60   | `last_action_index` | last executed action (normalised by space size)    |

This keeps `STATE_SIZE = 91` and the multimodal vector at 570 — the slots
merely carry meaning instead of zeros. An unknown gesture name falls back
to the `"none"` one-hot slot.

---

## Observability

- **REST API** under `/api/v1/teaching/*`: `GET /status`, `GET /transitions`,
  `POST /feedback`, `POST /demonstration`, `GET /qvalues`. State summaries
  in `/transitions` are derived from the reserved state slots — **never
  raw conversation text.** POST endpoints require the API key when one is
  configured (`DESKBOT_API__API_KEY`).
- **Web dashboard** at **`/teaching`** polls status, recent transitions,
  and Q-values, and offers forms to arm a session, inject a gesture, and
  submit feedback.
- `/api/v1/learning/status` reports `enabled` from
  `settings.learning.enabled` (a prior hard-coded `True` was fixed).