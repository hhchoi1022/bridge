

from bridge.alertconsumer import ALERCEConsumer
import time

if __name__ == "__main__":
    lsst = ALERCEConsumer(survey_type='lsst')
    try:
        lsst.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        lsst.stop()
        print("ALERCEConsumer stopped.")
# %%