import csv


def main():
    file = 'clustering/motherT0_ts.csv'
    filename = file[11:-4]

    with open(file) as csv_file:
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

        with open(file + '_final.csv', 'w', newline='') as csvfile:
            fieldnames = ['begin_time', 'finish_time', 'duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(time_begin)):
                duration_pointing = round(float(time_final[i]) - float(time_begin[i]), 2)
                if duration_pointing > 0.4:
                    row = {'begin_time': time_begin[i], 'finish_time': time_final[i], 'duration': duration_pointing}
                    writer.writerow(row)

        print(time_begin)
        print(time_final)
        print(cluster)


if __name__ == '__main__':
    main()
