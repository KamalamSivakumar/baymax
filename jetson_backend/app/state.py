from app.schemas import Mode


current_state = {
    "mode": Mode.kids,
    "last_voice_command": None,
    "last_robot_action": None,
    "pi_connected": False,
}