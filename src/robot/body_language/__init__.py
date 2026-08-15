"""Body Language Engine - high-level servo choreography.

The :class:`BodyLanguageEngine` consumes *requests* like
:class:`LookLeft`, :class:`HeadNod`, :class:`Wave`, :class:`Celebrate`,
:class:`ArmsRelax` and translates them into coordinated servo
motions on the four-servo chassis. It is the only component that
talks to the :class:`ServoController` for expressive motion.

The engine never knows about emotions directly. The EmotionEngine
emits a :class:`BodyLanguageHint` (head-tilt + arm-pose) which the
body-language engine interprets.
"""
