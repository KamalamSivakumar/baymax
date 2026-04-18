## Deskbot Baymax

Flow: Web app → Jetson FastAPI → Raspberry Pi FastAPI → motors/display/audio/buttons. 

Run the Jetson Backend on one terminal, The RPI backend on another terminal.  
Use /health and /device/status endpoints resp., to check the status of them both. 

For now kept place holder codes for audio and display. For the motor controller, kept only the basic actuation for the two wheels. We can tweak based on the motors and further iterations. 

Functionalities to implement:  
1. Microphone input to Jetson, and perform action. (speech to text)
2. Output llm stories/output as speech. (text to speech)
3. Boundary/Trajectory mapping of the mat/desk using video feed.
4. Facial Recognition of being happy/sad/etc using video feed (if it makes sense)