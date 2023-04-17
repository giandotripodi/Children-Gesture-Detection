import pandas as pd
import csv
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def main():
    # args = get_args()
    # filename = args.filename

    ts_path = "timeseries/motherT0.csv"
    filename = ts_path[11:]
    print(ts_path)

    dataframe = pd.read_csv(ts_path, header=None, sep=",")

    # Elimino l'ultima colonna contenente la label
    dataframe = dataframe.drop([2], axis=1)
    # Elimino la prima riga contenente gli headers
    dataframe = dataframe.drop([0], axis=0)

    dataframe.columns = ["FRAME", "TIME"]

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(dataframe)
    scaled_dataframe = pd.DataFrame(scaled_array, columns=dataframe.columns)

    dbscan_model = DBSCAN(eps=0.01, min_samples=5)
    dbscan_model.fit(scaled_dataframe)
    labels = dbscan_model.labels_
    dataframe["LABEL"] = labels

    dataframe.to_csv(r'clustering/' + filename, index=False, header=True)
    print("Cluster finished successfully. Saved at clustering/")

    with open('clustering/' + filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        time = []
        time_begin = []
        time_final = []
        cluster = []

        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif row[2] >= '0' and row[2] not in cluster and not time:
                cluster.append(row[2])
                time.append(row[1])
                line_count += 1
            elif row[2] >= '0' and row[2] in cluster:
                time.append(row[1])
                line_count += 1
            elif row[2] >= '0' and row[2] not in cluster and time:
                time_begin.append(time[0])
                time_final.append(time[-1])
                time = []
                time.append(row[1])
                cluster.append(row[2])
                line_count += 1
            elif row[2] == '-1' and time:
                time_begin.append(time[0])
                time_final.append(time[-1])
                time = []
                line_count += 1
        if time:
            time_begin.append(time[0])
            time_final.append(time[-1])
            time = []

        with open('clustering/final/' + filename, 'w', newline='') as csvfile:
            fieldnames = ['begin_time', 'finish_time', 'state']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(time_begin)):
                duration_pointing = round(float(time_final[i]) - float(time_begin[i]), 2)
                if duration_pointing > 0.4:
                    row = {'begin_time': time_begin[i], 'finish_time': time_final[i], 'state': 'POINTING'}
                    writer.writerow(row)
    return


#
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("filename", type=str)
#     args = parser.parse_args()
#     return args


if __name__ == '__main__':
    main()
