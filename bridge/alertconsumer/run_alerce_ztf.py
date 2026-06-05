

from bridge.alertconsumer import ALERCEConsumer
import time

if __name__ == "__main__":
    ztf = ALERCEConsumer(survey_type='ztf')
    try:
        ztf.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ztf.stop()
        print("ALERCEConsumer stopped.")
# %%