

from bridge.alertconsumer import FINKConsumer
import time

if __name__ == "__main__":
    lsst = FINKConsumer(survey_type='lsst')
    try:
        lsst.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        lsst.stop()
        print("FINKConsumer stopped.")
# %%