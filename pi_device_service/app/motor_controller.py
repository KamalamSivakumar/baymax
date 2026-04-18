def get_speed_from_profile(motor_profile: str) -> float:
    speeds = {
        "slow_drive": 0.35,
        "medium_drive": 0.55,
        "fast_drive": 0.75,
    }

    return speeds.get(motor_profile, 0.35)


def apply_motor_profile(profile: str) -> dict:
    speed = get_speed_from_profile(profile)

    print(f"[MOTOR PROFILE] profile={profile}, speed={speed}")

    return {
        "motor_profile_applied": True,
        "profile": profile,
        "speed": speed,
    }


def set_motor_speeds(left_speed: float, right_speed: float) -> dict:
    """
    Replace this function with actual GPIO/PWM motor driver code.

    left_speed and right_speed range:
        -1.0 = full reverse
         0.0 = stop
         1.0 = full forward
    """

    left_speed = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))

    print(f"[MOTORS] left={left_speed}, right={right_speed}")

    return {
        "left_motor": left_speed,
        "right_motor": right_speed,
    }


def execute_motor_action(action: str, mode: str, motor_profile: str) -> dict:
    speed = get_speed_from_profile(motor_profile)

    if action == "drive_forward":
        motor_command = set_motor_speeds(speed, speed)
        command_name = "forward"

    elif action == "drive_backward":
        motor_command = set_motor_speeds(-speed, -speed)
        command_name = "backward"

    elif action == "turn_left":
        motor_command = set_motor_speeds(-speed, speed)
        command_name = "turn_left"

    elif action == "turn_right":
        motor_command = set_motor_speeds(speed, -speed)
        command_name = "turn_right"

    elif action == "stop":
        motor_command = set_motor_speeds(0.0, 0.0)
        command_name = "stop"

    else:
        return {
            "motor_executed": False,
            "action": action,
            "message": "No motor movement required for this action.",
        }

    return {
        "motor_executed": True,
        "mode": mode,
        "motor_profile": motor_profile,
        "action": action,
        "command": command_name,
        "motor_command": motor_command,
    }