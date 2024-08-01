def timeSeries(df_start,df_current, date_start, date_current):
    df_start = df_start.sort_values(by=['id']).reset_index(drop=True)
    df_start = df_start.sort_values(by=['id']).reset_index(drop=True)
    list_start = tuple(filter(lambda x: x not in ['bridge', 'implant'], df_start['id']))
    list_current = tuple(filter(lambda x: x not in ['bridge', 'implant'], df_current['id']))
    if len(set(list_start)) == len(list_start) and len(set(list_current)) == len(list_current) and len(set(list_current) - set(list_start)) == 0:
        elapsed_date = int((datetime.datetime.strptime(date_current, '%Y-%m-%d') - datetime.datetime.strptime(date_start, '%Y-%m-%d')).days)
        df_current['ratio_difference'] = 0
        df_current['time_grade'] = ''
        for _, row in df_start.iterrows():
            if row['id'] not in class_list and pd.isna(row['비율']) == False:
                df_map = df_current[df_current['id'] == row['id']]
                if len(df_map) != 0 and pd.isna(df_map['비율'].values[0]) == False:
                    ratio_difference = (df_map['비율'].values[0] - row['비율']) / elapsed_date * 365 * 5
                    if ratio_difference < 3:
                        time_grade = 'A'
                    elif 3 <= ratio_difference < 10:
                        time_grade = 'B'
                    else:
                        time_grade = 'C'
                    df_current['ratio_difference'][df_map.index[0]] = ratio_difference
                    df_current['time_grade'][df_map.index[0]] = time_grade
                elif len(df_map) == 0:
                    df_current.loc[len(df_current)] = {'id': row['id'], 'ratio_difference': None, 'time_grade': 'C'}