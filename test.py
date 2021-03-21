import pandas as pd
from datetime import datetime
import os


inp = [{"c1": 10, "c2": 100}, {"c1": 11, "c2": 110}, {"c1": 12, "c2": 120}]
log_df = pd.DataFrame(inp)


current_time = datetime.now().strftime("%H-%M-%S")
log_file_name = "Experiment-from-" + str(current_time) + ".log"

log_dir = os.path.join("project\log")
log_path = os.path.join(log_dir, log_file_name)

with open(log_path, mode="w", encoding="utf-8") as logfile:
    colums = log_df.columns
    for colum in colums:
        logfile.write(colum + "\t\n")
    for _, row in log_df.iterrows():
        for c in colums:
            logfile.write(str(row[c].item()))
            logfile.write("\t")
        logfile.write("\n")
