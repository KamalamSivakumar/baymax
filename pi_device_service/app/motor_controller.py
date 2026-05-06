"""
motor_controller.py — L298N motor driver via RPi.GPIO (BCM numbering).

Wiring (BCM pin numbers):
  Motor A (left wheel):   IN1=5, IN2=6,   ENA=12  (hardware PWM)
  Motor B (right wheel):  IN3=13, IN4=19, ENB=26  (hardware PWM)

Speed range: -1.0 (full reverse) → 0.0 (stop) → 1.0 (full forward)

If RPi.GPIO is unavailable (e.g. running on a dev machine), falls back to
a stub that logs values — no exception is raised.
"""

import logging

logger = logging.getLogger(__name__)

# ── GPIO pin assignments ───────────────────────────────────────────────────────
IN1 = 5
IN2 = 6
IN3 = 13
IN4 = 19
ENA = 12   # PWM channel for Motor A (left)
ENB = 26   # PWM channel for Motor B (right)
PWM_FREQ = 1000  # Hz

# ── GPIO initialisation ───────────────────────────────────────────────────────
_GPIO_AVAILABLE = False
_pwm_a = None
_pwm_b = None

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    _pwm_a = GPIO.PWM(ENA, PWM_FREQ)
    _pwm_b = GPIO.PWM(ENB, PWM_FREQ)
    _pwm_a.start(0)
    _pwm_b.start(0)
    _GPIO_AVAILABLE = True
    logger.info("L298N GPIO initialised (BCM): IN1=%d IN2=%d ENA=%d | IN3=%d IN4=%d ENB=%d",
                IN1, IN2, ENA, IN3, IN4, ENB)
except (ImportError, RuntimeError) as e:
    logger.warning("RPi.GPIO unavailable (%s) — motor stub active", e)


# ── Speed helpers ─────────────────────────────────────────────────────────────

def get_speed_from_profile(motor_profile: str) -> float:
    speeds = {
        "slow_drive":   0.35,
        "medium_drive": 0.55,
        "fast_drive":   0.75,
    }
    return speeds.get(motor_profile, 0.35)


def apply_motor_profile(profile: str) -> dict:
    speed = get_speed_from_profile(profile)
    logger.info("[MOTOR PROFILE] profile=%s  speed=%.2f", profile, speed)
    return {"motor_profile_applied": True, "profile": profile, "speed": speed}


# ── Core drive function ────────────────────────────────────────────────────────

def set_motor_speeds(left_speed: float, right_speed: float) -> dict:
    """
    Set both motor speeds.
      -1.0 = full reverse, 0.0 = stop, 1.0 = full forward
    """
    left_speed  = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))

    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        # ── Motor A (left) ──────────────────────────────────────────────
        if left_speed > 0:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
        elif left_speed < 0:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)
        else:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.LOW)
        _pwm_a.ChangeDutyCycle(abs(left_speed) * 100)

        # ── Motor B (right) ─────────────────────────────────────────────
        if right_speed > 0:
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
        elif right_speed < 0:
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)
        else:
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.LOW)
        _pwm_b.ChangeDutyCycle(abs(right_speed) * 100)
    else:
        logger.info("[MOTORS stub] left=%.2f  right=%.2f", left_speed, right_speed)

    return {"left_motor": left_speed, "right_motor": right_speed}


# ── Action dispatcher ──────────────────────────────────────────────────────────

def execute_motor_action(action: str, mode: str, motor_profile: str) -> dict:
    speed = get_speed_from_profile(motor_profile)

    action_map = {
        "drive_forward":  ( speed,  speed),
        "drive_backward": (-speed, -speed),
        "turn_left":      (-speed,  speed),
        "turn_right":     ( speed, -speed),
        "stop":           (  0.0,    0.0),
    }

    if action not in action_map:
        return {
            "motor_executed": False,
            "action":  action,
            "message": "No motor movement required for this action.",
        }

    left, right = action_map[action]
    motor_command = set_motor_speeds(left, right)

    return {
        "motor_executed": True,
        "mode":           mode,
        "motor_profile":  motor_profile,
        "action":         action,
        "motor_command":  motor_command,
    }


def cleanup_gpio() -> None:
    """Call on shutdown to release GPIO resources."""
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        try:
            _pwm_a.stop()
            _pwm_b.stop()
            GPIO.cleanup()
            logger.info("GPIO cleaned up")
        except Exception as exc:
            logger.warning("GPIO cleanup error: %s", exc)

    }