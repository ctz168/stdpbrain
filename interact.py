import sys
import os

pid_file = 'chat_pid.txt'

if len(sys.argv) > 1:
    command = sys.argv[1]
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = int(f.read())
        # This is a simplified example. In a real scenario, 
        # you'd use a more robust IPC mechanism like a named pipe or socket.
        # For this test, we'll just print the command, assuming the user will copy-paste it.
        print(f"Please manually enter the following into the chat process (PID: {pid}):")
        print(command)
else:
    # In the main script, we'd save the PID
    pid = os.getpid()
    with open(pid_file, 'w') as f:
        f.write(str(pid))
