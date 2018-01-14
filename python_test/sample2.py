import sys, time
for i in range(100):
    sys.stdout.write("\r%d" % i)
    sys.stdout.flush()
    time.sleep(0.1)
