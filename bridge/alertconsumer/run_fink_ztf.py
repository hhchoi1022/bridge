

from bridge.alertconsumer import FINKConsumer
import time

if __name__ == "__main__":
    ztf = FINKConsumer(survey_type='ztf')
    try:
        ztf.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ztf.stop()
        print("FINKConsumer stopped.")
# %%