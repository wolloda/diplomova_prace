from data_handling import *

def load_indexes(filename='level-2.txt', names=get_descriptive_col_names()):
    return pd.read_csv(filename, names = names + ["object_id"],  sep=r'[.\s+]', engine='python', header=None)

def get_num_of_numerical_rows(line, delimiter=";", col_to_skip="messif.objects.impl.ObjectGPSCoordinate"):
    """Determines the number of numerical rows that'll follow based on the number
    of columns in a current line. If there's a `col_to_skip`, returns position
    of its associated data
    """
    column_names = line.split(delimiter)
    num_of_rows = len(column_names)//2
    row_to_skip = None
    if col_to_skip in column_names:
        row_to_skip = column_names.index(col_to_skip) // 2
    return num_of_rows, row_to_skip

def get_label(h_line, should_be_int=False):
    """Gets the label (`object_id`) from the first column by searching a phrase
    just before it.
    """
    re_match = re.search('BucketIdObjectKey ', h_line)
    if re_match:
        if should_be_int:
            return int(h_line[re_match.end():])
        else:
            return h_line[re_match.end():]
    else:
        re_match = re.search('AbstractObjectKey ', h_line)
        if re_match:
            if should_be_int:
                return int(h_line[re_match.end():])
            else:
                return h_line[re_match.end():]
        else: return None

def parse_objects(filename="objects.txt", is_filter=True):
    """Loads a file with objects line by line, extracting labels (object_id) and
    numerical data. Returns list of labels and list of numerical values, merged to
    a single row (so the information about which value corresponds to which column
    is lost).
    """
    labels = []; numerical = []; numerical_row = []; attributes_per_descr_len = []
    counter = 0
    if is_filter:
        next_line = 2
    else:
        next_line = 1
    with open(filename) as file:
        line = file.readline().rstrip('\n')
        while line:
            # the 0th line contains the label
            if counter == 0:
                labels.append(get_label(line))
                counter += 1
            # the 2nd line contains column names = how many numerical rows will follow
            elif counter == next_line:
                num_of_rows, row_to_skip = get_num_of_numerical_rows(line) 
                # get a list of integers of all the consecutive numerical rows
                for n in range(num_of_rows):
                    line = file.readline().rstrip('\n')
                    #n_of_descriptors = num_of_rows - 1 if row_to_skip is not None else num_of_rows
                    if not row_to_skip or n != row_to_skip:
                        found_attributes = list(map(int, re.findall(r'[\-0-9\.]+', line)))
                        if len(attributes_per_descr_len) < num_of_rows - int(row_to_skip is not None):
                            attributes_per_descr_len.append(len(found_attributes))
                        numerical_row += found_attributes
                counter = 0
                numerical.append(numerical_row)
                numerical_row = []
            
            else:
                counter += 1
            line = file.readline().rstrip('\n')
        return labels, numerical, attributes_per_descr_len

def merge_dfs(normalized, labels, index_df):
    """Merges normalized numerical data, labels and dataframe of indexes to one
    dataframe. All the columns from both of them are kept.
    """
    df = pd.DataFrame(normalized)

    df['object_id'] = labels
    try:
        df["object_id"] = df["object_id"].astype(np.int64)
    except:
        pass
    #return df, labels
    # Move "object_id" column to the front
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    final = pd.merge(df, index_df, on=['object_id'], how = 'outer')[:len(normalized)]
    # move first-level and second-level columns to the front
    cols = final.columns.tolist()
    final = final[cols[-2:] + cols[:-2]]
    return final

def get_objects_with_indexes(indexes_filename='level-2.txt', objects_filename='objects.txt', names=get_descriptive_col_names(), is_filter=True):
    """ Gets values of descriptors and indexes (labels), combines them into a pandas df.
    """
    index_df = load_indexes(indexes_filename, names)
    labels, numerical, descr_lengths = parse_objects(objects_filename, is_filter)
    if not is_filter:
        labels = labels[:5000000]
        numerical = numerical[:5000000]
    df = merge_dfs(numerical, labels, index_df)
    #df, labels = merge_dfs(numerical, labels, index_df)
    return df, descr_lengths
    #return labels, numerical