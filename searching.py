from classification import *
from data_handling import *
#from network import *
import time # perf measurements
logging.basicConfig(datefmt='%d-%m-%y %H:%M', format='%(asctime)-15s%(levelname)s: %(message)s', level=logging.INFO)

def get_classification_probs_per_model(model, x, est_mapping=[], debug=False):
    """Computes each estimator's vote on the predicted class for a single object

    Parameters
    ----------
    model : RF model
    x: numpy array
        single row (without labels)
    est_mapping (optional): Dictionary
        mapping of estimator representation of the class (begins with 0) to real class labels

    Returns
    -------
    numpy array
        array of predictions (size: num_estimators)
    """
    p = []
    to_be_predicted = x.reshape(1, -1)
    for e in model.estimators_:
        p.append(model.classes_[int(e.predict(to_be_predicted)[0])])
    return np.array(p)

def get_classification_probs_for_vector(model, x, debug=False):
    p = []
    to_be_predicted = x#.reshape(1, -1)
    for e in model.estimators_:
        p.append(model.classes_[e.predict(to_be_predicted)])
    return np.array(p)


def get_classification_probs_per_level_(x, cached, model, est_mapping=[], value='value_l1', debug=False):
    """Collects classification probabilities for an object on 1 level

    Parameters
    ----------
    x: numpy array
        single row (without labels)
    model : RF model
    est_mapping : Dictionary
        mapping of estimator representation of the class (begins with 0) to real class labels
    value : String
        label of the level

    Returns
    -------
    list
        list of sorted dict values: label: percentage of votes
    """
    classes_votes = []
    if cached is None:
        probs_2 = get_classification_probs_per_model(model, x, est_mapping, debug)
    else:
        probs_2 = cached
    n_estimators_2 = probs_2.shape[0]
    for c in np.unique(probs_2):
        classes_votes.append({value: int(c), 'votes_perc': np.where(probs_2 == c)[0].shape[0] / n_estimators_2})
    zero_prob_classes = list(set(model.classes_).difference(set(probs_2)))
    shuffle(zero_prob_classes)
    for c in zero_prob_classes:
        classes_votes.append({value: int(c), "votes_perc": 0})
    classes_votes_l2 = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
    return classes_votes_l2
    classes_votes_l2 = sorted(classes_votes, key=(lambda i: i['votes_perc']), reverse=True)
    return classes_votes_l2


def get_classification_probs(model_stack, row, est_mapping_l2, est_mapping_l3, debug=False):
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
        x = row.drop((get_descriptive_col_names_with_predictions()), axis=1).values
    else:
        x = row.drop(get_descriptive_col_names_with_predictions()).values
    classes_votes_l1 = get_classification_probs_per_level(x, clf_1)
    predicted_l1 = int(classes_votes_l1[0]['value_l1']) - 1
    classes_votes_l2 = get_classification_probs_per_level(x, (model_stack[1][predicted_l1]), [], value='value_l2')
    predicted_l2 = int(classes_votes_l2[0]['value_l2']) - 1
    classes_votes_l3 = get_classification_probs_per_level(x, (model_stack[2][predicted_l1][predicted_l2]), [], value='value_l3')
    return [
     row[get_descriptive_col_names()].values, classes_votes_l1, classes_votes_l2, classes_votes_l3]

def get_mappings_on_misclassifications_L12(df_res):
    class_dict = {}
    approx_dict = {}
    L1_preds = []
    L2_preds = []
    obj_ids = []
    L1s = []
    L2s = []
    for i, r in df_res[(df_res['first_lvl_pivot_id'] != df_res['first_lvl_pivot_id_pred'])].iterrows():
        L1_pred = int(r['first_lvl_pivot_id_pred'])
        L1 = int(r['first_lvl_pivot_id'])
        L2 = int(r['second_lvl_pivot_id'])
        if (L1, L2, L1_pred) in class_dict:
            c = class_dict[(L1, L2, L1_pred)]
        else:
            counters[L1_pred] += 1
            c = counters[L1_pred]
            class_dict[(L1, L2, L1_pred)] = counters[L1_pred]
        L1_preds.append(L1_pred)
        L2_preds.append(c)
        L1s.append(L1)
        L2s.append(L2)
        obj_ids.append(int(r['object_id']))
        approx_dict[f"M.1.{L1_pred}.{c}"] = f"M.1.{L1}.{L2}"

    return (
     obj_ids, L1_preds, L2_preds, L1s, L2s, approx_dict)

def get_mappings_on_misclassifications_(df_res, n_classes=15):
    counters = [100 for i in range(n_classes)]
    class_dict = {}
    approx_dict = {}
    L1_preds = []
    L2_preds = []
    obj_ids = []
    L1s = []
    L2s = []
    for i, r in df_res[(df_res['first_lvl_pivot_id'] != df_res['first_lvl_pivot_id_pred'])].iterrows():
        L1_pred = int(r['first_lvl_pivot_id_pred'])
        L1 = int(r['first_lvl_pivot_id'])
        L2 = int(r['second_lvl_pivot_id'])
        if (L1, L2, L1_pred) in class_dict:
            c = class_dict[(L1, L2, L1_pred)]
        else:
            counters[L1_pred] += 1
            c = counters[L1_pred]
            class_dict[(L1, L2, L1_pred)] = counters[L1_pred]
        L1_preds.append(L1_pred)
        L2_preds.append(c)
        L1s.append(L1)
        L2s.append(L2)
        obj_ids.append(int(r['object_id']))
        approx_dict[f"M.1.{L1_pred}.{c}"] = f"M.1.{L1}.{L2}"

    return (
     obj_ids, L1_preds, L2_preds, L1s, L2s, approx_dict)


def get_mappings_on_misclassifications_L23(df_res, n_classes=15, n_classes_2=100):
    counters = [[100 for j in range (n_classes_2+1)] for i in range(n_classes+1)]
    class_dict = {}
    approx_dict = {}
    L1_preds = []
    L2_preds = []
    obj_ids = []
    L1s = []
    L2s = []
    for i, r in df_res[(df_res['second_lvl_pivot_id'] != df_res['second_lvl_pivot_id_pred'])].iterrows():
        L_prev_pred = int(r['first_lvl_pivot_id_pred'])
        L1_pred = int(r['second_lvl_pivot_id_pred'])
        L1 = int(r['second_lvl_pivot_id'])
        #L2 = int(r['second_lvl_pivot_id'])
        if (L1, L1, L1_pred) in class_dict:
            c = class_dict[(L1, L1, L1_pred)]
        else:
            counters[L_prev_pred][L1_pred] += 1
            c = counters[L_prev_pred][L1_pred]
            class_dict[(L1, L1, L1_pred)] = counters[L_prev_pred][L1_pred]
        L1_preds.append(L1_pred)
        L2_preds.append(c)
        L1s.append(L1)
        L2s.append(L1)
        obj_ids.append(int(r['object_id']))
        approx_dict[f"C.1.{L_prev_pred}.{L1_pred}.{c}"] = f"C.1.{L_prev_pred}.{L1}.{L1}"

    return (
     obj_ids, L1_preds, L2_preds, L1s, L2s, approx_dict)


def add_level_to_queue(priority_q, probs, from_l1_level=None, from_l2_level=None, pop=True):
    """ Trains split of the data on L3 (multiple classifiers of multiple parents).

    Parameters
    ----------
    priority_q: List of Dicts
        Queue with sorted (based on probabilities) models to be visited 
    probs: List of dicts
        Probabilities on each level of the object
    n_prev_levels: int
        Previous levels
    Returns
    -------
    clfs_2: List of trained (nested) RFs
    df_2: Pandas DataFrame
        DF of predictions
    """
    if len(priority_q) != 0 and pop:
        priority_q.pop()
    for i in probs:
        if from_l1_level != None and from_l2_level != None:
            priority_q.append({f"C.1.{from_l1_level}.{from_l2_level}." + str(i['value_l3']): i['votes_perc']})
            #print(str((f"C.1.{from_l1_level}.{from_l2_level}." + str(i['value_l3']), i['votes_perc']))+",")
        elif from_l1_level != None:
            priority_q.append({f"M.1.{from_l1_level}." + str(i['value_l2']): i['votes_perc']})
            #print(str((f"M.1.{from_l1_level}." + str(i['value_l2']), i['votes_perc']))+",")
        else:
            priority_q.append({'M.1.' + str(i['value_l1']): i['votes_perc']})
            #print(str(('M.1.' + str(i['value_l1']), i['votes_perc']))+",")

    return priority_q

def process_node(priority_q, x, model_stack, gts, mapping, custom_classifier=None, debug=False):
    """ Performs node processing in context of approximate search.
        - pops the node from queue
        - decides, whether it's a 'hit' - equals the ground truth of the object on either level
        - collects next probabilities followed from the node
    
    Parameters
    ----------
    priority_q: List of Dicts
        Queue with sorted (based on probabilities) models to be visited 
    x: numpy array
        Data of Object being processed (minus labels)
    model_stack: (Nested) List of RFs
    gts: List of ground truths (L1, L2)
    debug: boolean

    Returns
    -------
    priority_q: Priority Queue with popped node and its successors added
    popped: Popped node
    is_L1_hit, is_L2_hit, is_L3_hit: Boolean
        - bools representing hit on each level
    """
    is_L1_hit = False
    is_L2_hit = False
    popped = priority_q.pop(0)
    node_to_process = list(popped.keys())[0]
    processed_node = node_to_process
    if debug:
        print(f"Popped {processed_node}")
    model_label = node_to_process.split('.')
    if len(model_label) == 3:
        if gts[0] == int(model_label[(-1)]):
            is_L1_hit = True
    if len(model_label) == 4:
        if gts[0] == int(model_label[(-2)]):
            if gts[1] == int(model_label[(-1)]):
                is_L2_hit = True
    if len(model_label) == 3:
        if custom_classifier == "random":
            probs = get_classification_probs_per_level(x, int(model_label[-1]), [], custom_classifier=custom_classifier, value='value_l2')
            priority_q = add_level_to_queue(priority_q, probs, (model_label[(-1)]), pop=False)
        elif len(model_stack[1]) > int(model_label[(-1)])-1:
            preds_index = [int(model_label[-1])]
            #print(preds_index)
            #print(mapping[0])
            if preds_index in mapping[0]:
                stack_index = mapping[0].index(preds_index)
                model = model_stack[1][stack_index]
                probs = get_classification_probs_per_level(x, model, [], custom_classifier=custom_classifier, value='value_l2')
                #print(probs)
                priority_q = add_level_to_queue(priority_q, probs, (model_label[(-1)]), pop=False)
    elif len(model_label) == 4:
        processed_node =  "C" + processed_node[1:]

    priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
    if debug:
        print(f"L[2-3] added - PQ: {priority_q}\n")
    return (priority_q, processed_node, is_L1_hit, is_L2_hit)

def process_node_3(priority_q, x, model_stack, gts, mapping, custom_classifier=None, debug=False):
    """ Performs node processing in context of approximate search.
        - pops the node from queue
        - decides, whether it's a 'hit' - equals the ground truth of the object on either level
        - collects next probabilities followed from the node
    
    Parameters
    ----------
    priority_q: List of Dicts
        Queue with sorted (based on probabilities) models to be visited 
    x: numpy array
        Data of Object being processed (minus labels)
    model_stack: (Nested) List of RFs
    gts: List of ground truths (L1, L2)
    debug: boolean

    Returns
    -------
    priority_q: Priority Queue with popped node and its successors added
    popped: Popped node
    is_L1_hit, is_L2_hit, is_L3_hit: Boolean
        - bools representing hit on each level
    """
    is_L1_hit = False
    is_L2_hit = False
    is_L3_hit = False
    popped = priority_q.pop(0)
    node_to_process = list(popped.keys())[0]
    processed_node = node_to_process
    if debug:
        print(f"Popped {processed_node}")
    model_label = node_to_process.split('.')
    if len(model_label) == 3:
        if gts[0] == int(model_label[(-1)]):
            is_L1_hit = True
    if len(model_label) == 4:
        if gts[0] == int(model_label[(-2)]):
            if gts[1] == int(model_label[(-1)]):
                is_L2_hit = True
    if len(model_label) == 3:
        if len(model_stack[1]) > int(model_label[(-1)])-1:
            preds_index = [int(model_label[-1])]
            #print(f"preds_index: {preds_index}")
            if preds_index in mapping[0]:
                stack_index = mapping[0].index(preds_index)
                model = model_stack[1][stack_index]
                probs = get_classification_probs_per_level(x, model, [], custom_classifier=custom_classifier, value='value_l2')
                priority_q = add_level_to_queue(priority_q, probs, (model_label[(-1)]), pop=False)
    elif len(model_label) == 4:
        if len(model_stack[2]) > int(model_label[(-1)])-1:
            preds_index = None
            if len(gts) >= 2: preds_index = [int(model_label[-2]), int(model_label[-1])]
            #print(f"{preds_index} not in {mapping[1]}")
            #print(f"preds_index: {preds_index}")
            if preds_index in mapping[1]:
                stack_index = mapping[1].index(preds_index)
                model = model_stack[2][stack_index]
                probs = get_classification_probs_per_level(x, model, [], custom_classifier=custom_classifier, value='value_l3')
                priority_q = add_level_to_queue(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False)
    elif len(model_label) == 5:
        processed_node =  "C" + processed_node[1:]

    priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
    if debug:
        print(f"L[2-3] added - PQ: {priority_q}\n")
    return (priority_q, processed_node, is_L1_hit, is_L2_hit, is_L3_hit)

def process_node_ref(priority_q, x, model_stack, gts, cached_probs, over_approx_dict):
    is_L1_hit = False
    is_L2_hit = False
    is_leaf = 0
    popped = priority_q.pop(0)
    processed_node = list(popped.keys())[0]
    if over_approx_dict and processed_node in over_approx_dict:
        if over_debug and debug == False: 
            debug=True
        if over_debug: print(gts)
        is_L1_hit = True
        is_L2_hit = True
        old_node_to_process = node_to_process
        old_popped = popped
        node_to_process = over_approx_dict[node_to_process]
        processed_node = [node_to_process]
        prev_models = "."
        for node in node_to_process.split(".")[2:-1]:
            processed_node.append(f"M.1{prev_models}{node}")
            prev_models += node + "."
    model_label = processed_node.split('.')
    if len(model_label) == 3:
        if gts[0] == int(model_label[(-1)]):
            is_L1_hit = True
    if len(model_label) == 4:
        if gts[0] == int(model_label[(-2)]):
            if gts[1] == int(model_label[(-1)]):
                is_L2_hit = True
    """
    if len(model_label) == 5:
        if gts[1] == int(model_label[(-1)]):
            if gts[1] == int(model_label[(-2)]):
                if gts[0] == int(model_label[(-3)]):
                    is_L3_hit = True
    """
    if len(model_label) == 3:
        probs = get_classification_probs_per_level_(x, cached_probs[1], model_stack[1][(int(model_label[(-1)]) - 1)], [], value='value_l2')
        priority_q = add_level_to_queue(priority_q, probs, model_label[(-1)], pop=False)
    
    """
    elif len(model_label) == 4:
        l1 = int(model_label[(-2)]) - 1
        l2 = int(model_label[(-1)]) - 1
        probs = get_classification_probs_per_level_(x, cached_probs[2], (model_stack[2][l1][l2]), [], value='value_l3')
        priority_q = add_level_to_queue(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False)
        is_leaf = 1
    """

    priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
    return (priority_q, processed_node, is_L1_hit, is_L2_hit, is_leaf)

from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower

def is_iterable(obj):
    return isinstance(obj, Iterable)

def remove_merged_paths(merged, popped, debug=False):
    for parent in merged:
        base_node_label = "M.1."
        for p in parent[0]:
            base_node_label += f"{p}."
        base_node_label = base_node_label[:-1]
        if base_node_label == popped:
            print(f"Popped {base_node_label}")
            return None
    return popped
    
def approximate_search(model_stack, df_res, object_id, mapping, labels=get_descriptive_col_names_with_predictions(), steps_limit=None, steps_limit_leaf=None, end_on_exact_hit=True, custom_classifier=None, debug=False):
    """ Implements the approximate search algorithm:
        - Given an object, its predictions and ground truths on L1, L2, find out the number of models visited in order to classify him.
        - Includes models on L1, L2 and leaf nodes levels.
        - Is 3 for objects classified correctly both on L1 and L2 (most object).
        - In other cases, collects the probabilities of other probable classes represented by models, sorts them and "hops" inbetween
        until a "hit" is made on L1 and L2.
        - Operates with
        - pops the node from queue
        - decides, whether it's a 'hit' - equals the ground truth of the object on either level
        - collects next probabilities followed from the node
    
    Parameters
    ----------
    priority_q: List of Dicts
        Queue with sorted (based on probabilities) models to be visited 
    x: numpy array
        Data of Object being processed (minus labels)
    model_stack: (Nested) List of RFs
    est_class_mappings_L2, est_class_mappings_L3: List of Dicts
        - mappings of classes and estimators' representation of them, for L2, L3 respectively
    debug: boolean

    Returns
    -------
    priority_q: Priority Queue with popped node and its successors added
    popped: Popped node
    is_L1_hit, is_L2_hit, is_L3_hit: Boolean
        - bools representing hit on each level
    """
    row = df_res[(df_res['object_id'] == object_id)]
    n_steps_global = 0
    n_leaf_steps_global = 0
    if custom_classifier is not None and is_iterable(custom_classifier) and "random" in custom_classifier:
        x = object_id
    else:
        x = row.drop((labels), axis=1, errors='ignore').values
    #assert x.shape[1] == 282
    gts = row[labels[:2]].values[0]
    #l1 = get_classification_probs_nn(x, model_stack[0])
    #s = time.time()
    l1 = get_classification_probs_per_level(x, model_stack[0], custom_classifier=custom_classifier)
    #e = time.time()
    #print(e-s)
    priority_q = [{'M.1': 1.0}]
    if debug:
        print(f"Step 0: M.1 added - PQ: {priority_q}\n")
    priority_q = add_level_to_queue(priority_q, l1)
    if debug:
        print(f"Step 1: L1 added - PQ: {priority_q[:5]}, ...\n")
    is_L1_hit = False
    is_L2_hit = False
    popped_nodes = []
    #print(l1)
    while (end_on_exact_hit and not (is_L1_hit and is_L2_hit) and len(priority_q) != 0) or (not end_on_exact_hit and len(priority_q) != 0):
        #print(popped_nodes)
        if steps_limit != None and steps_limit <= n_steps_global or steps_limit_leaf != None and steps_limit_leaf <= n_leaf_steps_global:
            #print("here")
            #merged_paths
            return {'id':object_id,  'steps_to_hit':n_steps_global,  'is_hit':is_L1_hit and is_L2_hit,  'gt_L1':gts[0],  'gt_L2':gts[1],  'popped_nodes':popped_nodes}
        else:
            if debug:
                print(f"Step {n_steps_global + 2} - Model visit {n_steps_global + 1}: ")
            priority_q, popped, is_curr_L1_hit, is_curr_L2_hit = process_node(priority_q, x, model_stack, gts, mapping=mapping, custom_classifier=custom_classifier, debug=debug)
            if type(popped) is list:
                popped_nodes.extend(popped)
            else: popped_nodes.append(popped)
            popped_nodes = list(set(popped_nodes))
            popped_nodes = sorted(popped_nodes, key=len)
            n_leaf_steps_global = get_number_of_unique_leaf_node_models(popped_nodes)
            if is_curr_L1_hit:
                is_L1_hit = True
            if is_curr_L2_hit:
                is_L2_hit = True
        n_steps_global += 1
    return {'id':object_id,  'steps_to_hit':n_steps_global,  'is_hit':is_L1_hit and is_L2_hit,  'gt_L1':gts[0],  'gt_L2':gts[1],  'popped_nodes':popped_nodes}

def approximate_search_3(model_stack, df_res, object_id, mapping, labels=get_descriptive_col_names_with_predictions(), steps_limit=None, steps_limit_leaf=None, end_on_exact_hit=True, custom_classifier=None, merged_paths=None, debug=False):
    """ Implements the approximate search algorithm:
        - Given an object, its predictions and ground truths on L1, L2, find out the number of models visited in order to classify him.
        - Includes models on L1, L2 and leaf nodes levels.
        - Is 3 for objects classified correctly both on L1 and L2 (most object).
        - In other cases, collects the probabilities of other probable classes represented by models, sorts them and "hops" inbetween
        until a "hit" is made on L1 and L2.
        - Operates with
        - pops the node from queue
        - decides, whether it's a 'hit' - equals the ground truth of the object on either level
        - collects next probabilities followed from the node
    
    Parameters
    ----------
    priority_q: List of Dicts
        Queue with sorted (based on probabilities) models to be visited 
    x: numpy array
        Data of Object being processed (minus labels)
    model_stack: (Nested) List of RFs
    est_class_mappings_L2, est_class_mappings_L3: List of Dicts
        - mappings of classes and estimators' representation of them, for L2, L3 respectively
    debug: boolean

    Returns
    -------
    priority_q: Priority Queue with popped node and its successors added
    popped: Popped node
    is_L1_hit, is_L2_hit, is_L3_hit: Boolean
        - bools representing hit on each level
    """
    row = df_res[(df_res['object_id'] == object_id)]
    n_steps_global = 0
    n_leaf_steps_global = 0
    if custom_classifier is not None and is_iterable(custom_classifier) and "random" in custom_classifier:
        x = object_id
    else:
        x = row.drop((labels), axis=1, errors='ignore').values
    gts = row[labels].values[0]
    l1 = get_classification_probs_per_level(x, model_stack[0], custom_classifier=custom_classifier)
    priority_q = [{'M.1': 1.0}]
    if debug:
        print(f"Step 0: M.1 added - PQ: {priority_q}\n")
    priority_q = add_level_to_queue(priority_q, l1)
    if debug:
        print(f"Step 1: L1 added - PQ: {priority_q[:5]}, ...\n")
    is_L1_hit = False; is_L2_hit = False; is_L3_hit = False
    popped_nodes = []
    #print(l1)
    while (end_on_exact_hit and not (is_L1_hit and is_L2_hit and is_L3_hit) and len(priority_q) != 0) or (not end_on_exact_hit and len(priority_q) != 0):
        #print(popped_nodes)
        if steps_limit != None and steps_limit <= n_steps_global or steps_limit_leaf != None and steps_limit_leaf <= n_leaf_steps_global:
            return {'id':object_id, 'steps_to_hit':n_steps_global, 'is_hit':is_L1_hit and is_L2_hit and is_L3_hit, 'gt_L1':gts[0], 'gt_L2':gts[1], 'popped_nodes':popped_nodes}
        else:
            if debug:
                print(f"Step {n_steps_global + 2} - Model visit {n_steps_global + 1}: ")
            priority_q, popped, is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit = process_node_3(priority_q, x, model_stack, gts, custom_classifier=custom_classifier, mapping=mapping, debug=debug)
            if merged_paths is not None:
                popped = remove_merged_paths(merged_paths, popped)
            if (merged_paths and popped is not None) or (not merged_paths):
                if type(popped) is list:
                    popped_nodes.extend(popped)
                else: popped_nodes.append(popped)
                popped_nodes = list(set(popped_nodes))
                popped_nodes = sorted(popped_nodes, key=len)
                n_leaf_steps_global = get_number_of_unique_leaf_node_models(popped_nodes)
                if is_curr_L1_hit:
                    is_L1_hit = True
                if is_curr_L2_hit:
                    is_L2_hit = True
                if is_curr_L3_hit:
                    is_L3_hit = True
                n_steps_global += 1
    return {'id':object_id, 'steps_to_hit':n_steps_global, 'is_hit':is_L1_hit and is_L2_hit and is_L3_hit, 'gt_L1':gts[0], 'gt_L2':gts[1], 'gt_L3':gts[2], 'popped_nodes':popped_nodes}


def approximate_search_mtree(df_orig, query_id, struct_df, max_visited_models=2, bucket_level=2, labels=get_descriptive_col_names(), is_profi=True, debug=False):
    n_visited_models = 0
    priority_queue = []
    region_id = ""
    visited_models = []
    query_row = df_orig[df_orig["object_id"] == query_id]
    regions = struct_df[~(struct_df['IP'].str.contains("\."))]
    priority_queue = get_mtree_distance(priority_queue, regions, query_row, df_orig, labels, is_profi=is_profi)
    #time_checkpoints = []; popped_nodes_checkpoints = []; steps_checkpoints = []
    while len(visited_models) < max_visited_models and len(priority_queue) > 0:
        popped = priority_queue.pop(0)
        region_id = popped[0]
        if not region_id in visited_models:
            if debug: print(f"Popped: {popped}")
            visited_models.append(region_id)
            if (not "." in region_id) and bucket_level >= 2:
                regions = struct_df[struct_df['IP'].str.contains(f"{region_id}\.")]
                priority_queue = get_mtree_distance(priority_queue, regions, query_row, df_orig, labels, is_profi=is_profi)
            elif bucket_level == 3 and len(region_id.split('.')) == 2:
                regions = struct_df[struct_df['IP'].str.contains(f"{region_id}\.")]
                priority_queue = get_mtree_distance(priority_queue, regions, query_row, df_orig, labels, is_profi=is_profi)
    
    
    v = {}; v['popped_nodes'] = []
    for m in visited_models:
        v['popped_nodes'].append(f"M.1.{str(m)}")
    return v

def approximate_search_mindex(df_orig, query_id, struct_df, L1_only_pivots, existing_regions, existing_regions_dict, max_visited_models=2, bucket_level=2, labels=["L1", "L2", "object_id"], is_profi=True, debug=False):
    n_visited_models = 0
    region_id = ""
    visited_models = []
    priority_queue = []
    query_row = df_orig[df_orig["object_id"] == query_id]
    pivot_ids = struct_df["object_id"].values
    pivot_descriptors = struct_df.drop(["object_id"], axis=1).values
    if not is_profi:
        pow_list = [pow(0.75, i) for i in range(8)]
    s = time.time()
    priority_queue = get_mindex_distance(pivot_ids, pivot_descriptors, query_row, df_orig, labels, L1_only_pivots, is_profi=is_profi)
    
    e = time.time()
    total = 0
    pivot_distances = priority_queue.copy()
    while len(visited_models) < max_visited_models:
        popped = priority_queue.pop(0)
        if len(priority_queue) > max_visited_models:
            priority_queue = priority_queue[:max_visited_models] 
        if not popped in visited_models:
            if debug: print(f"Popped: {popped}")
            region_id = popped[0]
            #print(popped)
            visited_models.append(region_id)
            #print(region_id)
            if (not "." in region_id) or ("." in region_id and len(region_id.split(".")) < bucket_level):
                if is_profi:
                    priority_queue = get_wspd(priority_queue, pivot_ids, pivot_distances, df_orig, labels, L1_only_pivots, existing_regions, popped, is_profi=is_profi)
                else:
                    #s = time.time()
                    priority_queue, t = get_wspd_cophir(priority_queue, pivot_ids, pivot_distances, df_orig, labels, existing_regions, existing_regions_dict, pow_list, popped)
                    #e = time.time()
                    total += t
    v = {}; v['popped_nodes'] = []
    for m in visited_models:
        v['popped_nodes'].append(f"M.1.{str(m)}")
    return v

def approximate_search_ref(model_stack, x, gts, cached_probs, steps_limit_leaf=None, steps_limit=None, end_on_exact_hit=True, over_approx_dict=None):
    n_steps_global = 0
    n_leaf_steps_global = 0
    priority_q = [{'M.1': 1.0}]
    l1 = get_classification_probs_per_level_(x, cached_probs[0], model_stack[0])
    priority_q = add_level_to_queue(priority_q, l1)
    is_L1_hit = False
    is_L2_hit = False
    popped_nodes = []
    while end_on_exact_hit and not (is_L1_hit and is_L2_hit) and len(priority_q) != 0 or not end_on_exact_hit and len(priority_q) != 0:
        if steps_limit != None and steps_limit <= n_steps_global or steps_limit_leaf != None and steps_limit_leaf <= n_leaf_steps_global:
            #print(priority_q)
            return {'steps_to_hit':n_steps_global,  'is_hit':is_L1_hit and is_L2_hit,  'gt_L1':gts[0],  'gt_L2':gts[1],  'popped_nodes':popped_nodes}
        else:
            #start = time.clock()
            priority_q, popped, is_curr_L1_hit, is_curr_L2_hit, is_leaf = process_node_ref(priority_q, x, model_stack, gts, cached_probs, over_approx_dict)
            #end = time.clock()
            #print(end-start)
            if type(popped) is list:
                popped_nodes.extend(popped)
            else: popped_nodes.append(popped)
            popped_nodes = list(set(popped_nodes)) # remove duplicates
            popped_nodes = sorted(popped_nodes, key=len)
            n_leaf_steps_global += is_leaf
            if is_curr_L1_hit:
                is_L1_hit = True
            if is_curr_L2_hit:
                is_L2_hit = True
        n_steps_global += 1
    #print(priority_q)
    return {'steps_to_hit':n_steps_global, 'is_hit':is_L1_hit and is_L2_hit and is_L3_hit, 'gt_L1':gts[0], 'gt_L2':gts[1], 'popped_nodes':popped_nodes}

def get_number_of_unique_leaf_node_models(popped_nodes):
    set_ = set([e for e in popped_nodes if e[0] == 'C'])
    return len(set_) 

def knn_search(df_1k, stop_cond_leaf, knns, model_stack, df_res, mapping, stop_cond_model=None, labels=get_descriptive_col_names_with_predictions(), n_objects=1000, row=None, knn=30, mtree=False, struct_df=None, custom_classifier=None, df_orig=None, debug=False):
    """ Implements the knn search algorithm:
        - given a dictionary of ground truth of k nearest neighbors (`knn`)
        - for `n_objects` of objects finds the estimated neighbourhood of objects (using approximate_seach with `stop_cond_leaf`)
        - finds out all the objects in the dataset (`df_res`) belonging to the estimated neighbourhood
        - evaluates the number of "hits" out of these objects in relation to the ground truth (i.e., how many objects out of the ground truth belong to estimated neighbourhood as well)
    Parameters
    ----------
    df_1k: Dataframe of main objects to search for
    stop_cond_leaf: Parameter to approximate_search, controls size of the neighborhood
    knns: dictionary of ground truth of k nearest neighbors
    model_stack: stack of trained models
    df_res: dataset of the objects
    est_class_mappings_L2, est_class_mappings_L3: List of Dicts
        - mappings of classes and estimators' representation of them, for L2, L3 respectively
    pointers: parameter to approximate_search, used if overlaps are considered
    n_candidates: if present controls number of chosen NN with the most probable neighborhood strategy 
    debug: boolean

    Returns
    -------
    priority_q: Priority Queue with popped node and its successors added
    popped: Popped node
    is_L1_hit, is_L2_hit, is_L3_hit: Boolean
        - bools representing hit on each level
    """
    stats = []; intersects = []; times = []
    if row is not None:
        iterate = row
    else:
        iterate = df_1k[:n_objects]
    #print(len(iterate))
    for i, o in iterate.iterrows():
        o_df_1 = int(o['object_id'])
        #print(o_df_1)
        c_L1_L2_L3 = 0
        c_L1_L2 = 0
        c_L1 = 0
        if debug:
            print(f"Orig object: {o_df_1}")
        if mtree:
            s = time.time()
            search_res = approximate_search_mtree(df_orig, o_df_1, struct_df, max_visited_models=stop_cond_model, is_profi=False, debug=False)
            times.append(time.time() - s)
        else:
            if stop_cond_model is not None:
                #print(stop_cond_model)
                s = time.time()
                search_res = approximate_search(model_stack, df_res, o_df_1, mapping, steps_limit=stop_cond_model, end_on_exact_hit=False, custom_classifier=custom_classifier, debug=False)
                times.append(time.time() - s)
            else:
                search_res = approximate_search(model_stack, df_res, o_df_1, mapping, steps_limit_leaf=stop_cond_leaf, end_on_exact_hit=False, debug=False)
            if search_res['is_hit']:
                c_L1_L2_L3 += 1
        if debug:
            print(f"\nTrying to hit {search_res['popped_nodes']} n_leaf_models: {get_number_of_unique_leaf_node_models(search_res['popped_nodes'])}")

        #popped_keys = [list(p_n.keys())[0] for p_n in search_res['popped_nodes']]
        popped_keys = [p_n for p_n in search_res['popped_nodes']]
        popped_keys.sort(key = lambda s: len(s))
        gts = []
        gts_L1 = []
        n_last_level = 0
        for p in popped_keys:
            model_label = p.split('.')
            if len(model_label) == 3:
                gts_L1.append(int(model_label[(-1)]))
            if len(model_label) == 4:
                if int(model_label[(-2)]) in gts_L1:
                    gts.append((int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 5 and (int(model_label[(-3)]), int(model_label[(-2)])) in gts and int(model_label[(-2)]) == int(model_label[(-1)]):
                    n_last_level += 1

        if debug:
            print(f"\nIdentified pool of ids to hit: {gts} | {n_last_level}")
        df_subset = [pd.DataFrame([])]
        for gt in gts:
            if debug:
                print(f"Appending {(gt[0], gt[1])}")
            if mtree:
                df_subset.append(df_orig[((df_orig['first_lvl_pivot_id'] == gt[0]) & (df_orig['second_lvl_pivot_id'] == gt[1]))])
            else:
                df_subset.append(df_res[((df_res['first_lvl_pivot_id_pred'] == gt[0]) & (df_res['second_lvl_pivot_id_pred'] == gt[1]))])
        if stop_cond_model != 1:
            df_subset = pd.concat(df_subset).drop_duplicates()

            nn_object_ids = np.array((list(knns[str(o_df_1)].keys())), dtype=(np.int64))
            if df_subset.shape[0] != 0:
                intersect = np.intersect1d(df_subset['object_id'].values, nn_object_ids)
                intersects.append(intersect)
                stats.append((o_df_1, intersect.shape[0] / knn, gts))
            else:
                stats.append((o_df_1, 0, gts))
        else:
            stats.append((o_df_1, 0, None))
    return stats, np.array(times).mean()

def knn_search_3(df_1k, stop_cond_leaf, knns, model_stack, df_res, mapping, stop_cond_model=None, n_objects=1000, row=None, knn=30, mtree=False, struct_df=None, custom_classifier=None, df_orig=None, labels=["L1", "L2", "L3", "L1_pred", "L2_pred", "L3_pred", "object_id"], merged_paths=None, debug=False):
    stats = []; intersects = []; times = []
    if row is not None:
        iterate = row
    else:
        iterate = df_1k[:n_objects]
    #print(len(iterate))
    for n, (i, o) in enumerate(iterate.iterrows()):
        if mtree: logging.info(f"{n}/{n_objects}")
        o_df_1 = int(o['object_id'])
        #print(o_df_1)
        c_L1_L2_L3 = 0
        c_L1_L2 = 0
        c_L1 = 0
        if debug:
            print(f"Orig object: {o_df_1}")
        if mtree:
            s = time.time()
            search_res = approximate_search_mtree(df_orig, o_df_1, struct_df, max_visited_models=stop_cond_model, labels=labels, is_profi=False, debug=False)
            times.append(time.time() - s)
        else:
            if stop_cond_model is not None:
                #print(stop_cond_model)
                s = time.time()
                search_res = approximate_search_3(model_stack, df_res, o_df_1, mapping=mapping, steps_limit=stop_cond_model, end_on_exact_hit=False, labels=labels, custom_classifier=custom_classifier, merged_paths=merged_paths, debug=False)
                times.append(time.time() - s)
            else:
                search_res = approximate_search_3(model_stack, df_res, o_df_1, mapping=mapping, steps_limit_leaf=stop_cond_leaf, end_on_exact_hit=False, debug=False)
            if search_res['is_hit']:
                c_L1_L2_L3 += 1
        if debug:
            print(f"\nTrying to hit {search_res['popped_nodes']} n_leaf_models: {get_number_of_unique_leaf_node_models(search_res['popped_nodes'])}")

        #popped_keys = [list(p_n.keys())[0] for p_n in search_res['popped_nodes']]
        popped_keys = [p_n for p_n in search_res['popped_nodes']]
        popped_keys.sort(key = lambda s: len(s))
        gts = []; gts_L1 = []; gts_L2 = []
        n_last_level = 0
        for p in popped_keys:
            model_label = p.split('.')
            if len(model_label) == 3:
                gts_L1.append(int(model_label[(-1)]))
            if len(model_label) == 4:
                if int(model_label[(-2)]) in gts_L1:
                    gts_L2.append((int(model_label[(-1)])))
                if len(model_label) == 5 and (int(model_label[(-3)]), int(model_label[(-2)])) in gts and int(model_label[(-2)]) == int(model_label[(-1)]):
                    n_last_level += 1
            if len(model_label) == 5:
                if int(model_label[(-3)]) in gts_L1 and int(model_label[(-2)]) in gts_L2:
                    gts.append((int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                else:
                    print(f"Got bucket {model_label} and either his parent or his parent's parent hasn't hit: debug")

        if debug:
            print(f"\nIdentified pool of ids to hit: {gts} | {n_last_level}")
        df_subset = [pd.DataFrame([])]
        for gt in gts:
            if debug:
                print(f"Appending {(gt[0], gt[1])}")
            if mtree:
                df_subset.append(df_orig[((df_orig['L1'] == gt[0]) & (df_orig['L2'] == gt[1]))])
            else:
                df_subset.append(df_res[((df_res['L1_pred'] == gt[0]) & (df_res['L2_pred'] == gt[1]))])
        if stop_cond_model != 1:
            df_subset = pd.concat(df_subset).drop_duplicates()

            nn_object_ids = np.array((list(knns[str(o_df_1)].keys())), dtype=(np.int64))
            if df_subset.shape[0] != 0:
                intersect = np.intersect1d(df_subset['object_id'].values, nn_object_ids)
                intersects.append(intersect)
                stats.append((o_df_1, intersect.shape[0] / knn, gts))
            else:
                stats.append((o_df_1, 0, gts))
        else:
            stats.append((o_df_1, 0, None))
    return stats, np.array(times).mean()

def knn_search_ref(X, gts_all, object_ids, stop_cond_leaf, knns, model_stack, df_res, cached_probs, over_approx_dict=None, knn=30):
    stats = []
    start = time.clock()
    for x, gts, o_df_1 in zip(X, gts_all, object_ids):
        #o_df_1 = int(o['object_id'])
        #def approximate_seach_ref(model_stack, x, gts, steps_limit_leaf=None, end_on_exact_hit=True, over_approx_dict=None, debug=False, over_debug=False):
        #start = time.clock()
        search_res = approximate_search_ref(model_stack, x, gts, cached_probs, over_approx_dict=over_approx_dict, steps_limit_leaf=stop_cond_leaf, end_on_exact_hit=False)
        #end = time.clock()
        #print(end-start)
        popped_keys = [p_n for p_n in search_res['popped_nodes']]
        popped_keys.sort(key = lambda s: len(s))
        gts = []; gts_L1 = []
        for p in popped_keys:
            model_label = p.split('.')
            if len(model_label) == 3:
                gts_L1.append(int(model_label[(-1)]))
            if len(model_label) == 4:
                if int(model_label[(-2)]) in gts_L1:
                    gts.append((int(model_label[(-2)]), int(model_label[(-1)])))
        df_subset = []
        end = time.clock()
        print(end-start)
        for gt in gts:
            #start = time.clock()
            df_subset.append(df_res[((df_res['first_lvl_pivot_id_pred'] == gt[0]) & (df_res['second_lvl_pivot_id_pred'] == gt[1]))])
            #end = time.clock()
            #print(end-start)
        df_subset = pd.concat(df_subset)
        nn_object_ids = np.array((list(knns[str(o_df_1)].keys())), dtype=(np.int64))
        intersect = np.intersect1d(df_subset['object_id'].values, nn_object_ids)
        stats.append((o_df_1, intersect.shape[0] / knn, gts))
    return stats

def prepare_data_for_search(df_1k):
    X = df_1k.values[:, :282]
    gts = np.vstack([df_1k["first_lvl_pivot_id"].values, df_1k["second_lvl_pivot_id"].values]).T
    object_ids = df_1k["object_id"].values
    return X, gts, object_ids