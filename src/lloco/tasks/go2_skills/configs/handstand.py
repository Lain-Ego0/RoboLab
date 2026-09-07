"""Registration for Gym ``go2_leggedstand``, the actual Go2 handstand task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from ..common import make_handstand_env_cfg, make_handstand_runner_cfg
from ..profiles import HANDSTAND

register_mjlab_task(
  task_id=HANDSTAND.task_id,
  env_cfg=make_handstand_env_cfg(),
  play_env_cfg=make_handstand_env_cfg(play=True),
  rl_cfg=make_handstand_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
