from collections import defaultdict
import math 

class MonsterClassificationAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, samples, new_monster):
        #Add your code here!
        #
        #The first parameter to this method will be a labeled list of samples in the form of
        #a list of 2-tuples. The first item in each 2-tuple will be a dictionary representing
        #the parameters of a particular monster. The second item in each 2-tuple will be a
        #boolean indicating whether this is an example of this species or not.
        #
        #The second parameter will be a dictionary representing a newly observed monster.
        #
        #Your function should return True or False as a guess as to whether or not this new
        #monster is an instance of the same species as that represented by the list.

        positive_data = [data for data, label in samples if label]
        negative_data = [data for data, label in samples if not label]

        positive_count = len(positive_data)
        negative_count = len(negative_data)

        positive_prior = positive_count / len(samples)
        negative_prior = negative_count / len(samples)

        positive_distribution = get_distribution(positive_data)
        negative_distribution = get_distribution(negative_data)

        log_prob_positive = math.log(positive_prior)
        log_prob_negative = math.log(negative_prior)
        
        for key, value in new_monster.items():
            
            log_prob_positive += math.log(likelihood(positive_distribution, positive_count, key, value) + 1e-5)
            log_prob_negative += math.log(likelihood(negative_distribution, negative_count, key, value) + 1e-5)

        return log_prob_positive > log_prob_negative
       

def get_distribution(data):
    distribute_dict = defaultdict(lambda: defaultdict(int))
    for features in data:
        for name, value in features.items():
            distribute_dict[name][value] += 1
    return distribute_dict

def likelihood(distribute_dict, total, feature_name, value):
    return distribute_dict[feature_name][value] / total