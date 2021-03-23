from data_handling import *
from random import shuffle
from cophir_distances import get_distance
import math
from sklearn.metrics import pairwise_distances  

import time


def get_classification_probs_per_model(model, x):
    """Computes each estimator's vote on the predicted class for a single object

    Parameters
    ----------
    model : RF model
    x: numpy array
        single row (without labels)

    Returns
    -------
    numpy array
        array of predictions (size: num_estimators)
    """
    p = []
    #assert x.shape[1] == 282 or x.shape[1] == 4096
    if model == []:
        return np.array(p)
    x = x.reshape(1, -1)
    for e in model.estimators_:
        #                    class_l = encoder.classes_[i]
        #            classes_votes.append({value: int(class_l) if not np.isnan(class_l) else class_l, "votes_perc": p})
        #p.append(model.classes_[int(e.predict(x)[0])])
        #print(e.predict(x), e.predict(x)[0], model.classes_[int(e.predict(x)[0])])
        #output = encoder.inverse_transform([model.classes_[int(e.predict(x)[0])]])
        p.append(model.classes_[int(e.predict(x)[0])])
    return np.array(p)

def get_classification_probs_per_model_nn(model, x, est_mapping=[]):
    p = []
    """
    for e in range(len(model.estimators_)):
        if est_mapping == []:
            p.append(model.classes_[int(model.estimators_[e].predict(x.reshape(1, -1))[0])])
            #print(model.classes_)
        else:
            print(np.int64(model.estimators_[e].predict(x.reshape(1, -1))[0]))
            p.append(est_mapping[np.int64(model.estimators_[e].predict(x.reshape(1, -1))[0])])
    """
    #print(model.predict(x)[0])
    return model.predict(x)[0]

def get_classification_probs_per_level_new(x, model, value="value_l1"):
    # Don't use - .predict_proba() is much slower than iterating through estimators manually
    if "RandomForestClassifier" in str(type(model)):
        classes_votes = []
        probas = model.predict_proba(x)[0]
        for i,p in enumerate(probas):
            #model.classes_[i]
            classes_votes.append({value: int(model.classes_[i]), "votes_perc": p})
        classes_votes_l2 = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
        #print(classes_votes_l2)
        #print(classes_votes_l2)
        return classes_votes_l2

from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower

def is_iterable(obj):
    return isinstance(obj, Iterable)

def get_classification_probs_per_level(x, model, encoder=[], value="value_l1", custom_classifier=None):
    """Collects classification probabilities for an object on 1 level

    Parameters
    ----------
    x: numpy array
        single row (without labels)
    model : RF model
    value : String
        label of the level

    Returns
    -------
    list
        list of sorted dict values: label: percentage of votes
    """
    """
    classes_votes = []
    probs_2 = get_classification_probs_per_model_nn(model, x)
    for e, p in enumerate(probs_2):
        classes_votes.append({value: int(e), "votes_perc": p})
    classes_votes = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
    return classes_votes
    """
    #print(model.__class__.__name__)
    if custom_classifier == "multilabel":
        classes_votes = []
        if "DecisionTreeClassifier" in str(type(model)):
            #probas = model.predict(x)
            #print(model.classes_, model.predict(x)[0])
            #print("here", model.classes_[model.predict(x)[0]])
            classes_votes.append({value: model.classes_[0], "votes_perc": 1.0})
            #print(classes_votes)
            return classes_votes
        #print(x.shape)
        null_classes = []
        #s = time.time()
        predictions = model.predict_on_batch(x)[0].numpy()
        #print(time.time() - s)
        #print(f"enc: {encoder}")
        predictions_rationed = predictions #/ sum(predictions)
        #print(f"change: len: {len(predictions)}")
        #print(predictions_rationed)
        #print(np.argmax(predictions_rationed))
        #print(encoder); print(predictions)
        for c,p in enumerate(predictions_rationed):
            #if p != 0:
            #   try:
            if encoder:
                classes_votes.append({value: encoder.inverse_transform([c])[0]+1, "votes_perc": p})
            else:
                classes_votes.append({value: c+1, "votes_perc": p})
            #    except ValueError:
            #        null_classes.append(c)
            #elif c != 150:
            #    null_classes.append(c)
        #print(null_classes)
        #shuffle(null_classes)
        #print(encoder, classes_votes)
        #for c in null_classes:
        #    classes_votes.append({value: c+1, "votes_perc": 0})
        #print(classes_votes)
        classes_votes_l2 = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
        return classes_votes_l2
    elif custom_classifier == "GMM":
        classes_votes = []
        null_classes = []
        if "DecisionTreeClassifier" in str(type(model)):
            print("Here, dec tree")
            #probas = model.predict(x)
            classes_votes.append({value: model.classes_[0], "votes_perc": 1.0})
            return classes_votes
        predictions = model.predict_proba(x)[0]
        for c,p in enumerate(predictions):
            if p != 0:
                try:
                    classes_votes.append({value: c, "votes_perc": p})
                except ValueError:
                    null_classes.append(c)
            elif c != 150:
                null_classes.append(c)
        shuffle(null_classes)
        for c in null_classes:
            classes_votes.append({value: c, "votes_perc": 0})
        #print(classes_votes)
        classes_votes_l2 = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
        return classes_votes_l2
    elif custom_classifier is not None:
        classes_votes = []
        if "DecisionTreeClassifier" in str(type(model)):
            #probas = model.predict(x)
            classes_votes.append({value: model.classes_[model.predict(x)[0]], "votes_perc": 1.0})
            return classes_votes
        else:
            probas = model.predict_proba_single(x)

            #print(probas)
            #s= time.time()
            for i,p in enumerate(probas):
                if len(model.classes_) > i:
                    if encoder and len(encoder.classes_) > i:
                        class_l = encoder.classes_[i]
                        classes_votes.append({value: int(class_l) if not np.isnan(class_l) else class_l, "votes_perc": p})
                    else:
                        classes_votes.append({value: i+1, "votes_perc": p})
                    #else:
                    #    print(f"{i} not included")
            classes_votes_l2 = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
            #t = time.time() - s
            #print(t)
            #print(classes_votes_l2)
            return classes_votes_l2
    elif "RandomForestClassifier" in str(type(model)):
        #print("Here")
        classes_votes = []
        #s = time.time()
        probs_2 = get_classification_probs_per_model(model, x)
        if encoder:
            probs_2 = encoder.inverse_transform(probs_2)
        #print(probs_2)
        #print("only classifications", time.time() - s)
        #if len(probs_2) == 0:
        #    return []
        n_estimators_2 = probs_2.shape[0]
        #print(probs_2)
        uniques, counts = np.unique(probs_2, return_counts=True)
        for u,c in zip(uniques, counts):
            if str(u) != "nan":
                classes_votes.append({value: int(u), "votes_perc": c / n_estimators_2})
        zero_prob_classes = np.setdiff1d(model.classes_, uniques)#list(set(model.classes_).difference(set(probs_2)))
        shuffle(zero_prob_classes)
        for c in zero_prob_classes:
            classes_votes.append({value: int(c), "votes_perc": 0})
        classes_votes_l2 = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
        #print(classes_votes_l2)
        #print(classes_votes_l2)
        return classes_votes_l2
    else:
        """
        # if CNN
        try:
            #s = time.time()
            predictions = model.predict(x)[0]
            #e = time.time()
            #print(f"Time on non-cnn net: {e-s}")
        except:
            #s = time.time()
            #print(x.shape)
            if x.shape == (1,282):
                x = x.reshape(1,282,1)
            else:
                x = x.reshape(1,4096,1)
            predictions = model.predict(x)[0]
            #e = time.time()
            #print(f"Time on cnn net: {e-s}")
        """
        #print(encoder)
        classes_votes = []
        if "DecisionTreeClassifier" in str(type(model)):
            #probas = model.predict(x)
            classes_votes.append({value: model.classes_[0], "votes_perc": 1.0})
            return classes_votes
        else:
            #print("Prediction alone")
            #s = time.time()
            #predictions = model.predict(x)[0]
            #model.load_weights(f"../tmp_nn/0.h5")
            #predictions = model.predict(x)[0]
            predictions = model.predict_on_batch(x)[0].numpy()
            nan_allowed = True
            #model.load_weights(f"./tmp_nn/0.h5")
            #print(time.time()-s)
            null_classes = []
            #if type(predictions) is int or np.issubdtype(type(predictions), int):
            #    predictions = [predictions]
            #print(predictions)
            if encoder:
                try:
                    classes = encoder.inverse_transform([c for c in range(int(predictions.shape[0])-1)])
                except:
                    classes = encoder.inverse_transform([c for c in range(int(predictions.shape[0] // 2))])

            if np.isnan(classes).any():
                #print(np.isnan(classes), predictions)
                predictions = predictions[:-1]
                na_pred = sum(np.array(predictions)[np.isnan(classes)])
            #print(len(predictions), len(classes))
            for i, (p, c) in enumerate(zip(predictions, classes)):
                #if p != 0:
                    #print(f"Here, encoder {encoder.classes_[c]} {p}")
                if str(c) != "nan":
                    classes_votes.append({value: int(c), "votes_perc": p})
                elif nan_allowed:
                    nan_allowed = False
                    classes_votes.append({value: c, "votes_perc": na_pred})
                    #else:
                    #    classes_votes.append({value: int(c)+1, "votes_perc": p})
                    #classes_votes.append({value: c+1, "votes_perc": p})
                #elif c != 0:
                #    null_classes.append(int(c)+1)
            #shuffle(null_classes)
            #for c in null_classes:
            #    classes_votes.append({value: c, "votes_perc": 0})
            #print(len(classes_votes))
            classes_votes_l2 = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
            #print(classes_votes_l2)
            return classes_votes_l2

"""
def get_mtree_distance(pivot_id, objects_in_region, radius):
    pass
    # get all distances to all objects in a given region_id (find out according to IP)
    # subtract radius from it (it'll be negative numbers)
    # sort by the smallest (e.g., -17, then -15)
"""

#from test_euclid import cdist_generic
#from scipy.spatial.distance import euclidean
#from eucl_dist.cpu_dist import dist

def get_euclidean(object_1, object_2):
    #print(object_1.shape)
    assert object_1.shape == object_2.shape 
    if object_1.shape[0] != 4096:
        object_1 = object_1.values.reshape(-1, 1)
    if object_2.shape[0] != 4096:
        object_2 = object_2.reshape(-1, 1)
    assert object_2.shape[0] == 4096
    return np.linalg.norm(object_1-object_2)

def get_mtree_distance(priority_queue, L1_regions, query_row, df_orig, labels, is_profi=True):
    #n_euclidean = 0
    for (pivot_id, ip, radius) in np.array(L1_regions):
        #print(pivot_id, radius, ip)
        pivot_row = df_orig[df_orig["object_id"] == pivot_id].drop(["object_id"], axis=1, errors='ignore')
        #if r["object_id"] != pivot_id:
        """
        if n_euclidean == 0:
            s = time.time()
            priority_queue.append([str(ip), get_euclidean(query_row.drop(labels, axis=1), pivot_row.drop(labels, axis=1)) - np.float(radius)])
            e = time.time()
            print(e-s)
                        pivot_distances.append([str(pivot_ip), get_euclidean(query_row.drop(labels, axis=1), pivot_descriptor.reshape(1, -1))])

        else:
        """
        if is_profi:
            #print(get_euclidean(query_row.drop(labels, axis=1), pivot_row.drop(labels, axis=1)))
            #print(query_row.drop(labels, axis=1).values[0].T.shape)
            #print(pivot_row.drop(labels, axis=1).values[0].T.shape)
            #print(query_row.drop(labels, axis=1).values)
            #print(pivot_row.drop(labels, axis=1).values)
            priority_queue.append([str(ip), get_euclidean(query_row.drop(labels, axis=1, errors='ignore').values[0].T, \
                                                          pivot_row.drop(labels, axis=1, errors='ignore').values[0].T) - np.float(radius)])
        else:
            priority_queue.append([str(ip), get_distance(query_row.drop(labels, axis=1, errors='ignore').values[0].T, \
                                                         pivot_row.drop(labels, axis=1, errors='ignore').values[0].T) - np.float(radius)])
        #n_euclidean += 1
    #print(n_euclidean)
    #s = time.time()
    priority_queue = sorted(priority_queue, key=lambda x: x[1])
    #e = time.time()
    #print(e-s)
    return priority_queue

def get_mtree_distance_profi_(priority_queue, pivot_id, ip, radius, query_row, df_orig):
    #for pivot_id, radius, ip in zip(L1_regions["object_id"].values, L1_regions["radius"].values, L1_regions["IP"].values):
    pivot_row = df_orig[df_orig["object_id"] == pivot_id]#.drop(["object_id"], axis=1)
        #if r["object_id"] != pivot_id:
    """
    if n_euclidean == 0:
        s = time.time()
        distances.append([ip, get_euclidean(query_row, pivot_row.drop(["object_id"], axis=1)) - radius])
        e = time.time()
        print(e-s)
    else:
    """
    priority_queue.append([ip, get_euclidean(query_row, pivot_row.drop(["object_id"], axis=1)) - radius])
    #n_steps_global += 1
    #s = time.time()
    priority_queue = sorted(priority_queue, key=lambda x: x[1])
    #e = time.time()
    #print(e-s)
    return priority_queue #, n_steps_global

# Mtree - 0: object_id - IP - radius | 
# 1: 

def get_mindex_distance(pivot_ids, pivot_descriptors, query_row, labels, is_profi=True, objects_labels=None):
    # Gets base euclidean or cohpir distance of the query to the pivots
    pivot_distances = []
    max_pivot_ip = 127
    for pivot_ip, (pivot_id, pivot_descriptor) in enumerate(zip(pivot_ids, pivot_descriptors)):
        if objects_labels is not None:
            pivot_ip = objects_labels[pivot_ip]
        if is_profi:
            #print(pivot_ip, get_euclidean(query_row.drop(labels, axis=1), pivot_descriptor.reshape(1, -1)))
            pivot_distances.append([str(pivot_ip), get_euclidean(query_row.drop(labels, axis=1), pivot_descriptor.reshape(1, -1))])
        else:
            pivot_distances.append([str(pivot_ip), get_distance(query_row.drop(labels, axis=1).values[0].T, pivot_descriptor.T)])
        if pivot_ip == max_pivot_ip:
            break
    pivot_distances = sorted(pivot_distances, key=lambda x: x[1])
    return pivot_distances

def get_closest_pivots_for_pivot(pivot, top_n_pivot_distances, bucket_size=8):
    top_region_ids = [d[0] for d in top_n_pivot_distances]
    # Gets the correct top pivots list in regards to a given pivot

    closest_pivots = top_region_ids
    closest_pivots_dist = top_n_pivot_distances.copy()
    #print(top_n_pivot_distances)
    removed = 0
    for current_pivot in pivot:
        for current_pivot_part in current_pivot[0].split("."):
            #print(current_pivot_part)
            #print(current_pivot)
            if current_pivot_part in closest_pivots:
                #print(f"{current_pivot_part} found")
                idx = closest_pivots.index(current_pivot_part)
                del closest_pivots_dist[idx]
                removed += 1
                #closest_pivots_dist = closest_pivots_dist[:idx] + closest_pivots_dist[idx+1:]
                closest_pivots = [d[0] for d in closest_pivots_dist]
    """ 
    if pivot[0] not in top_region_ids:
        closest_pivots = top_n_pivot_distances[:-1]
    else:
        idx = top_region_ids.index(pivot[0])
        # if it's a top pivot as well, removes it and adds one extra pivot
        closest_pivots = top_n_pivot_distances[:idx] + top_n_pivot_distances[idx+1:]
    """
    #print(bucket_size-removed)
    return closest_pivots_dist[:bucket_size-removed]

def estimate_distance_of_best_deepest_path(pivot_distances, pow_list, max_level=8):
    # Gets the closest `max_level` pivots and estimates what would the best scenario 
    # of the deepest possible distance (`max_level`) be in combination with every pivot.
    # This serves as the basis for priority queue, NOT the original distances.
    pivot_distances_norm = []
    for p in pivot_distances:
        pivot_normalized_sum = p[1]
        closest_pivots = get_closest_pivots_for_pivot([p], pivot_distances[:max_level])
        for closest_pivot, power in zip(closest_pivots, pow_list[1:]):
            # WSPD
            pivot_normalized_sum += closest_pivot[1]*power

        pivot_distances_norm.append([p[0], pivot_normalized_sum])
    pivot_distances_norm = sorted(pivot_distances_norm, key=lambda x: x[1])
    return pivot_distances_norm

def extend_path(longest_path, pivot_distances_top, pow_list, max_length, debug=True):
    used_pivots = longest_path[0].split(".")
    path_length = len(used_pivots) + 1
    #print(f"Top dist: {pivot_distances_top}")
    top_region_ids = [d[0] for d in pivot_distances_top]
    top_region_ids
    #set(top_region_ids).difference(set(used_pivots))
    distance = longest_path[1]
    used_regs = []
    for i, reg in enumerate(top_region_ids):
        if not reg in used_pivots:
            #print(pivot_distances_top, i, used_pivots, pow_list, path_length-1)
            #print(pivot_distances_top[i], pow_list[path_length-1])
            distance += pivot_distances_top[i][1]*pow_list[path_length-1]
            path_length += 1
            used_regs.append(reg)
            if path_length > max_length:
                break
    if debug:
        print(f"Modified distance path: {used_pivots} . {used_regs} | dist={distance}")
    return [longest_path[0], distance]

def process_wspd_node(prev_pivot, node_region_id, pivot_distances, existing_regions_dict, level, pivot_distances_top, pow_list, max_length, priority_queue, find_bucket=True):
    nodes_to_return = []
    node_base_distance = dict(pivot_distances)[node_region_id]
    current_node = [prev_pivot[0] + "." + node_region_id, (prev_pivot[1] + node_base_distance*pow_list[level])]
    #if current_node[0] == "109.44": print(f"appending {current_node}")
    pivot_distances.append(current_node)
    #print(current_node); print(pivot_distances_top)
    nodes_to_return.append(current_node)

    counter = 1
    #print(current_node[0])
    if find_bucket:
        while current_node[0] in existing_regions_dict:
            if current_node not in priority_queue:
                pivot_distances.append(current_node)
                nodes_to_return.append(current_node)
            # continuing the search for lower better result
            prev_pivot = current_node
            p = pivot_distances_top[counter]
            level += 1
            #print(level, pow_list[level], pow_list_sums[level])
            current_node = [prev_pivot[0] + "." + p[0], (prev_pivot[1] + p[1]*pow_list[level])]
            counter += 1
    longest_path = nodes_to_return[-1]
    longest_path = extend_path(longest_path, pivot_distances_top, pow_list, max_length)
    print(f"returning: {longest_path}")
    if find_bucket:
        for n in nodes_to_return:
            n[1] = longest_path[1]
        other_nodes = nodes_to_return[1:-1]
        #other_nodes = [nodes_to_return[0]]
    if find_bucket:
        #print(longest_path, other_nodes, pivot_distances)
        #return longest_path, other_nodes, pivot_distances
        return nodes_to_return, other_nodes, pivot_distances
    else:
        return nodes_to_return, pivot_distances

def get_wspd(priority_queue, pivot_distances, regions, pow_list, popped, objects_in_buckets, max_levels=2, is_profi=True, find_bucket=True):
    #if current_popped[0] not in objects_in_buckets:
    pivot_distances_current = pivot_distances.copy()
    #print(pivot_distances_current)
    pivot_distances_current_keys = [p[0] for p in pivot_distances_current]
    index = pivot_distances_current_keys.index(popped[0])
    popped = [popped[0], pivot_distances_current[index][1]]
    popped_name_split = popped[0].split(".")
    priority_queue_curr = []
    for pivot in pivot_distances_current:
        pivot_name_split = pivot[0].split(".")
        
        if len(set(pivot_name_split).intersection(set(popped_name_split))) == 0:
            current_region_str = f'{popped[0]}.{pivot[0]}'
            if current_region_str in regions and current_region_str not in pivot_distances_current_keys:
                distance = popped[1]
                counter = len(popped[0].split("."))
                if len(pivot_name_split) == 1:
                    distance += pow_list[counter]*pivot[1]
                else:
                    pow_list_from = pow_list[counter:]
                    for c, p in enumerate(pivot_name_split):
                        idx = pivot_distances_current_keys.index(p)
                        distance += pivot_distances_current[idx][1]*pow_list_from[c]
                counter += len(pivot_name_split)
                pivot_distances.append([current_region_str, distance])

                ext_path = get_closest_pivots_for_pivot([popped, pivot], pivot_distances[:max_levels], max_levels)
                potential_regions = []
                potential_regions.append(current_region_str)
                #if current_region_str == "21.92.10": print(distance, counter)
                for l, node in zip(pow_list[counter:], ext_path):
                    """
                    if potential_region in regions:
                        current_region_str += f'.{node[0]}'
                        potential_regions.append(current_region_str)
                    """ 
                    distance += l*node[1]
                for reg in potential_regions:
                    priority_queue_curr.append([reg, distance])
    priority_queue = priority_queue_curr + priority_queue
    priority_queue = sorted(priority_queue, key=lambda x: x[1])
    #print(f"Top 5 pq: {priority_queue[:5]}: length: {len(priority_queue)}")
    return priority_queue, pivot_distances

def get_wspd_old(priority_queue, pivot_distances, pivot_distances_normalized, existing_regions_dict, pow_list, current_popped, max_levels=2, is_profi=True, find_bucket=True):
    p_area = current_popped[1]#pivot_distances[prev_pivot]
    level_path = [int(x) for x in current_popped[0].split(".")]
    level = len(level_path)
    #print(level_path, sum(pow_list[:level+1]))
    other_nodes_all = []
    for i, p in enumerate(pivot_distances_normalized):
        print(f"p: {p}")
        print(f"dist popped: {dict(pivot_distances)[current_popped[0]]}")
        if (p[0] != current_popped[0]) and f"{current_popped[0]}.{p[0]}" in existing_regions_dict:
            pivot_distances_top = get_closest_pivots_for_pivot(p, pivot_distances[:max_levels], max_levels)
            if find_bucket:
                found_node, other_nodes, _ = process_wspd_node([current_popped[0], dict(pivot_distances)[current_popped[0]]], \
                                                p[0], pivot_distances, existing_regions_dict, level, \
                                                pivot_distances_top, pow_list, max_levels, priority_queue, find_bucket=find_bucket)
                print(other_nodes)
                other_nodes_all.extend(other_nodes)
            else:
                found_node, _ = process_wspd_node([current_popped[0], dict(pivot_distances)[current_popped[0]]], \
                                                p[0], pivot_distances, existing_regions_dict, level, \
                                                pivot_distances_top, pow_list, max_levels, priority_queue, find_bucket=find_bucket)
            #current_node.append(found_nodes)
            #priority_queue.append(found_node)
            priority_queue.extend(found_node)
        priority_queue = sorted(priority_queue, key=lambda x: x[1])
    #if not find_bucket: 
    #print(f"Top 5 pq: {priority_queue[:5]}: length: {len(priority_queue)}")
    #return None
    return priority_queue, other_nodes_all

def get_wspd_cophir(priority_queue, pivot_distances, existing_regions_dict, pow_list, prev_pivot, is_profi=False):
    p_area = prev_pivot[1]
    level_path = [int(x) for x in prev_pivot[0].split(".")]
    level = len(level_path)
    total = 0
    for i, p in enumerate(pivot_distances):
        if (p[0] != prev_pivot[0]) and f"{prev_pivot[0]}.{p[0]}" in existing_regions_dict:
            #priority_queue.append([prev_pivot[0] + "." + p[0], p_area + p[1]*pow_list[level]])
            found_nodes = process_wspd_node(prev_pivot, p, existing_regions_dict, level, pivot_distances[1:], pow_list, priority_queue)
            #print(found_nodes)
            priority_queue.extend(found_nodes)
    s = time.time()
    priority_queue = sorted(priority_queue, key=lambda x: x[1])
    e = time.time()
    return priority_queue, e-s

def get_classification_probs(model_stack, row, est_mapping_l2, est_mapping_l3):
    """Collects classification probabilities for an object on all levels

    Parameters
    ----------
    model_stack : (nested) list of RF models
    row: pandas' df row
    est_mapping_l2 : Dictionary
        mapping of estimator representation of the class (begins with 0) to real class labels on L2
    est_mapping_l3 : Dictionary
        mapping of estimator representation of the class (begins with 0) to real class labels on L3

    Returns
    -------
    list
        object's ground truth + object_id, sorted probabilities on L1, sorted probabilities on L2, sorted probabilities on L3
    """
    clf_1 = model_stack[0][0]
    if len(row.shape) == 2:
        x = row.drop(get_descriptive_col_names_with_predictions(), axis=1).values
    else:
        x = row.drop(get_descriptive_col_names_with_predictions()).values

    classes_votes_l1 = get_classification_probs_per_level(x, clf_1)
    predicted_l1 = int(classes_votes_l1[0]['value_l1'])-1
    classes_votes_l2 = get_classification_probs_per_level(x, model_stack[1][predicted_l1], est_mapping_l2[predicted_l1], value="value_l2")
    predicted_l2 = int(classes_votes_l2[0]['value_l2'])-1
    classes_votes_l3 = get_classification_probs_per_level(x, model_stack[2][predicted_l1][predicted_l2], est_mapping_l3[predicted_l1][predicted_l2], value="value_l3")
    return [row[get_descriptive_col_names()].values, classes_votes_l1, classes_votes_l2, classes_votes_l3]

def get_mappings_on_misclassifications(df_res, n_classes=15):
    counters = [100 for i in range(n_classes)]
    class_dict = {}; approx_dict = {}
    L1_preds = []; L2_preds = []; obj_ids = []; L1s = []; L2s = []
    for i,r in df_res[(df_res["first_lvl_pivot_id"] != df_res["first_lvl_pivot_id_pred"])].iterrows():
        L1_pred = int(r['first_lvl_pivot_id_pred']); L1 = int(r['first_lvl_pivot_id']); L2 = int(r['second_lvl_pivot_id'])
        if (L1, L2, L1_pred) in class_dict:
            c = class_dict[(L1, L2, L1_pred)]
        else:
            counters[L1_pred] += 1
            c = counters[L1_pred]
            class_dict[(L1, L2, L1_pred)] = counters[L1_pred]
        #print(f"{L1_pred}.{101 + counters[L1_pred]} : {L1}.{L2}")
        L1_preds.append(L1_pred); L2_preds.append(c); L1s.append(L1); L2s.append(L2); obj_ids.append(int(r["object_id"]))
        approx_dict[f"M.1.{L1_pred}.{c}"] = f"M.1.{L1}.{L2}"

    return (obj_ids, L1_preds, L2_preds, L1s, L2s, approx_dict)

def create_overlaps_approx_dict(class_comb_dict, df_res_l1_miss, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    approx_dict = {}
    L1_pred_label = f"{L1_label}_pred"
    for n,g in df_res_l1_miss.groupby([f"{L1_label}_pred"]):
        for i, obj in g.iterrows():
            group_c = class_comb_dict[(int(obj[f"{L1_label}_pred"]), int(obj[f"{L2_label}_pred"]),int(obj[f"{L1_label}"]))]
            approx_dict[f"M.1.{int(obj[L1_pred_label])}.{group_c}"] = f"M.1.{int(obj[L1_label])}.{int(obj[L2_label])}"
    return approx_dict

def correct_training_labels(df_res, df_res_l1_miss, class_comb_dict, n_objects_in_new_classes, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    capacity_dict = get_avg_capacities(df_res, L1_label, L2_label)
    mismatched_objs = df_res_l1_miss[["object_id", f"{L1_label}_pred", f"{L2_label}_pred", f"{L1_label}"]].values
    split_dfs = [df_res[df_res[f"{L1_label}_pred"] == i] for i in range(df_res[f"{L1_label}_pred"].max()+1)]
    for m_o in mismatched_objs:
        r = split_dfs[m_o[1]].loc[split_dfs[m_o[1]]["object_id"] == m_o[0]].copy()
        r[L2_label] = class_comb_dict[(m_o[1], m_o[2], m_o[3])]
        r[L1_label] = m_o[1]
        r_all = np.array([r.values[0],]*int((capacity_dict[m_o[1]])/n_objects_in_new_classes[(m_o[1], m_o[2])]))
        r_all_df = pd.DataFrame(r_all, columns=df_res.columns)
        print(m_o, int((capacity_dict[m_o[1]])/n_objects_in_new_classes[(m_o[1], m_o[2])]))
        split_dfs[m_o[1]] = pd.concat([split_dfs[m_o[1]], r_all_df])
    return split_dfs

def create_overlaps_L12(df_res, stack_l1, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    df_res_l1_miss = df_res[df_res[f"{L1_label}_pred"] != df_res[f"{L1_label}"]]
    
    class_comb_dict = {}; n_objects_in_new_classes = {}
    counters = [(df_res[L2_label].max() + 1) for i in range(df_res[L1_label].max() + 1)]
    for class_comb in np.unique(np.array(list(df_res_l1_miss.groupby([f"{L1_label}_pred", f"{L2_label}_pred",  f"{L1_label}"]).groups.keys())), axis=0):
        class_comb_dict[(class_comb[0], class_comb[1], class_comb[2])] = counters[class_comb[0]]
        n_objects_in_new_classes[(class_comb[0], class_comb[1])] = df_res_l1_miss[(df_res_l1_miss[f"{L1_label}_pred"] == class_comb[0]) & (df_res_l1_miss[f"{L2_label}_pred"] == class_comb[1])].shape[0]
        counters[class_comb[0]] += 1
    approx_dict = create_overlaps_approx_dict(class_comb_dict, df_res_l1_miss, L1_label, L2_label)
    split_dfs_corrected = correct_training_labels(df_res, df_res_l1_miss, class_comb_dict, n_objects_in_new_classes, L1_label, L2_label)
    splits = create_splits(split_dfs_corrected)
    models_to_retrain = np.unique(np.array(list(class_comb_dict.keys()))[:, 0], axis=0)
    for i in models_to_retrain:
        print(f"Training model {i}")
        stack_l1[i-1].fit(splits[i-1]['X'], splits[i-1]["y_2"])    
    
    return approx_dict, stack_l1, splits

def correct_training_labels_L23_leaf(df_res, df_res_l2_miss, class_comb_dict, n_objects_in_new_classes, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    mismatched_objs = df_res_l2_miss[["object_id", f"{L1_label}_pred", f"{L2_label}_pred", f"{L1_label}"]].values
    capacity_dict = get_avg_capacities(df_res)
    df_res_c = df_res.copy()
    for m_o in mismatched_objs:
        r = df_res_c.loc[df_res_c["object_id"] == m_o[0]].copy()
        print(f"Cap for {m_o[1]}, {m_o[2]} == {capacity_dict[(m_o[1], m_o[2])]}")
        """
        print(n_objects_in_new_classes[(m_o[1], m_o[2], m_o[2], m_o[3])])
        r[L2_label] = class_comb_dict[(m_o[1], m_o[2], m_o[2], m_o[3])]
        for i in range(int((capacity_dict[(m_o[1], m_o[2])])/n_objects_in_new_classes[(m_o[1], m_o[2], m_o[2], m_o[3])])):
            df_res_c = df_res_c.append(r)
        #df_res_c.loc[df_res_c["object_id"] == m_o[0], L2_label] = class_comb_dict[(m_o[1], m_o[2], m_o[2])]
        """
        print(class_comb_dict[(m_o[1], m_o[2],  m_o[2])])
        r[L2_label] = class_comb_dict[(m_o[1], m_o[2],  m_o[2])]
        #r[L1_label] = m_o[1]
        r_all = np.array([r.values[0],]*int((capacity_dict[(m_o[1], m_o[2])])/n_objects_in_new_classes[(m_o[1],  m_o[2], m_o[2])]))
        r_all_df = pd.DataFrame(r_all, columns=df_res.columns)
        print(m_o, int((capacity_dict[(m_o[1], m_o[2])])/n_objects_in_new_classes[(m_o[1], m_o[2], m_o[2])]))
        df_res = pd.concat([df_res, r_all_df])
    return df_res_c

def create_overlaps_approx_dict_L23_leaf(class_comb_dict, df_res_l1_miss, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    approx_dict = {}
    L1_pred_label = f"{L1_label}_pred"; L2_pred_label = f"{L2_label}_pred"
    for n,g in df_res_l1_miss.groupby([f"{L1_label}_pred", f"{L2_label}_pred"]):
        for i, obj in g.iterrows():
            group_c = class_comb_dict[(int(obj[f"{L1_label}_pred"]), int(obj[f"{L2_label}_pred"]),int(obj[f"{L2_label}_pred"]))]
            #if L1_pred_label == L1_label:
            approx_dict[f"C.1.{int(obj[L1_pred_label])}.{int(obj[L2_pred_label])}.{group_c}"] = f"C.1.{int(obj[L1_label])}.{int(obj[L2_label])}.{int(obj[L2_label])}"
    return approx_dict

def create_splits_groupby(df_res_c, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    split_data = []; split_data_mapping = {}
    L1_pred_label = f"{L1_label}_pred"; L2_pred_label = f"{L2_label}_pred"
    split_dfs = df_res_c.groupby([L1_pred_label, L2_pred_label])
    for i, g in enumerate(split_dfs.groups):
        X = df_res_c[(df_res_c[L1_pred_label] == g[0]) & (df_res_c[L2_pred_label] == g[1])].drop(get_descriptive_col_names_with_predictions(), axis=1, errors='ignore')
        #assert X.shape[1] == 282
        y = df_res_c[(df_res_c[L1_pred_label] == g[0]) & (df_res_c[L2_pred_label] == g[1])][get_descriptive_col_names()].values
        split_data.append({'X': X, 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,2]})
        split_data_mapping[g] = i
    return split_data, split_data_mapping

def create_overlaps_L23_leaf(df_res, stack_l2, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    df_res_l2_miss = df_res[df_res[f"{L2_label}_pred"] != df_res[f"{L2_label}"]]
    
    class_comb_dict = {}; n_objects_in_new_classes = {}
    counters = [[] for i in range(df_res[L1_label].max() + 1)]
    for i,c in enumerate(counters):
        counters[i] = [int(df_res[L2_label].max()) + 1 for j in range(df_res[L2_label].max()+1)]
    #print(counters)
    for class_comb in np.unique(np.array(list(df_res_l2_miss.groupby([f"{L1_label}_pred", f"{L2_label}_pred"]).groups.keys())), axis=0):
        # class_comb[0], class_comb[1], class_comb[1]) -> repeating the same class, cause it's a leaf
        #print(class_comb)
        class_comb_dict[(class_comb[0], class_comb[1], class_comb[1])] = counters[class_comb[0]][class_comb[1]]
        n_objects_in_new_classes[(class_comb[0], class_comb[1], class_comb[1])] = df_res_l2_miss[(df_res_l2_miss[f"{L1_label}_pred"] == class_comb[0]) & (df_res_l2_miss[f"{L2_label}_pred"] == class_comb[1])].shape[0]
        #print(class_comb[0], class_comb[1])
        counters[class_comb[0]][class_comb[1]] += 1
    approx_dict = create_overlaps_approx_dict_L23_leaf(class_comb_dict, df_res_l2_miss, L1_label, L2_label)
    df_res_corrected = correct_training_labels_L23_leaf(df_res, df_res_l2_miss, class_comb_dict, n_objects_in_new_classes, L1_label, L2_label)
    splits, split_data_mapping = create_splits_groupby(df_res_corrected)
    models_to_retrain = np.unique(np.array(list(class_comb_dict.keys()))[:, :2], axis=0)
    for i in models_to_retrain:
        #print(f"Training model {i[0]}, {i[1]} using stack[2][{i[0]-1}][{i[0]-1}]")
        stack_l2[i[0]-1][i[1]-1].fit(splits[split_data_mapping[(i[0], i[1])]]['X'], splits[split_data_mapping[(i[0], i[1])]]["y_2"])    
    
    return approx_dict, stack_l2, splits

def get_avg_capacities(df_res, L1_label="first_lvl_pivot_id", L2_label="second_lvl_pivot_id"):
    cap_dict = {}
    for v in df_res[f"{L1_label}_pred"].unique():
        cap_dict[v] = int(np.unique(df_res[df_res[f"{L1_label}_pred"] == v][f"{L2_label}_pred"].values, return_counts=True)[1].mean())

    for name, g in df_res.groupby([f"{L1_label}_pred", f"{L2_label}_pred"]):
        cap_dict[name] = g[f"{L2_label}_pred"].value_counts().values[0]
    
    return cap_dict