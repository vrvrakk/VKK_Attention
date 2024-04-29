import time

def countdown(t):
    while t > 0:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")  # \r returns the cursor to the start of the line
        time.sleep(1)
        t -= 1
    print("Time's up!")

# Start the countdown for 120 seconds
countdown(120)