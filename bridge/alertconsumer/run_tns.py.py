

from bridge.alertconsumer import TNSConsumer
import time

if __name__ == "__main__":
    tns = TNSConsumer()
    try:
        tns.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tns.stop()
        print("TNSConsumer stopped.")
# %%