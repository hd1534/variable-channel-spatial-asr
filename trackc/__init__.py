"""Track C: Qwen3-ASR with a microphone-pair spatial adapter, one model for who-spoke-when and what.

``text``     target and output text (speaker tags) and grid slots, no heavy imports
``scenes``   word-aligned two-talker training and validation sessions
``align``    the one-off word-alignment index (Qwen3-ForcedAligner, Qwen venv)
``model``    the unified model (Qwen venv: ``qwen-asr`` + ``peft``)
``train``    training, validation and the eval-set run
"""
