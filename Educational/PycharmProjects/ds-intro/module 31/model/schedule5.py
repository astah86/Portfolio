import dill
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime
import pandas as pd

sched = BlockingScheduler(timezone=tzlocal.get_localzone())

df = pd.read_csv('data/homework.csv')
file_name = 'cars_pipe.pkl'
with open(file_name, 'rb') as file:
   model = dill.load(file)


@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(n=5)
    data['pred'] = model['model'].predict(data)
    print(data[['id', 'pred']])


if __name__ == '__main__':
    sched.start()



