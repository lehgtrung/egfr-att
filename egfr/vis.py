import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import argparse

LOG = "logs/"

def get_last_file():
    flist = glob.glob("**/events.out.tfevents*", recursive=True)
    ftime = [os.path.getmtime(f) for f in flist]
    idx = np.argmax(ftime)
    return flist[idx]


# Extraction function
def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    try:
        for e in tf.train.summary_iterator(path):
            for v in e.summary.value:
                r = {'metric': v.tag, 'value': v.simple_value}
                runlog = runlog.append(r, ignore_index=True)

    # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
        return None

    num_of_metric = len(set(runlog.metric))
    runlog['epoch'] = [item for sublist in [[i] * num_of_metric for i in range(0, len(runlog) // num_of_metric)] for
                       item in sublist]

    return runlog

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='Input tf.events', dest='fi', default=get_last_file())
    args = parser.parse_args()

    if not os.path.exists('./vis/' + os.path.basename(args.fi)):
        os.makedirs('./vis/' + os.path.basename(args.fi))

    df = sum_log(args.fi)
    #print(df)
    metrics = list(set([i.split('_')[1] for i in df.metric]))
    for m in metrics:
        plt.figure()
        plt.plot(df[df['metric'] == ('train_' + m)].value.values, label=('train_' + m))
        plt.plot(df[df['metric'] == ('val_' + m)].value.values, label=('val_' + m))
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig("./vis/" + os.path.basename(args.fi) + '/' + m + ".png", bbox_inches='tight')
    print('Done')