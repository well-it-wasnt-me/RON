"""Built-in face animations.

These are timeline-based animation sequences that the FaceOrchestrator
and FaceAnimator can use to express complex emotional states.

Currently available:

* :class:`ThinkingDotsAnimation` - animated "thinking dots" that cycle
  the gaze position periodically while in the thinking state.
* :class:`SpeakingAnimation` - mouth open/close cycles timed to phoneme
  visemes for lip-synced speech.
* :class:`WakeAnimation` - a bright wake-up flash + attention-getting
  animation that plays when the wake word is detected.
"""

from robot.face.animations.speaking import SpeakingAnimation
from robot.face.animations.thinking_dots import ThinkingDotsAnimation
from robot.face.animations.wake import WakeAnimation, WakeFrame

__all__ = [
    "SpeakingAnimation",
    "ThinkingDotsAnimation",
    "WakeAnimation",
    "WakeFrame",
]
