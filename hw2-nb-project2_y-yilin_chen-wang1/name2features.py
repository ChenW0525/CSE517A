#!/usr/bin/python
import numpy as np
import sys
import random

def name2features(name):
    """
    Take a name and create a feature. 
    The feature can be any length and can be binary, integer or real numbers. 
    But every name must generate the same length feature. 
    """
    
    d = 2048 # number of hashing buckets
    # dim = 2
    # v = np.zeros([d] * dim)
    v = np.zeros(d)
    v_1 = np.zeros(d)
    v_len = np.zeros(d)
    v_bigram = np.zeros(d)
    name=name.lower() 
    
    # hash prefixes & suffixes - alexander -> [prefixa, prefixal, prefixale, suffixr, suffixer, suffixder]
    
    prefix_max = 3
    for m in range(prefix_max):
        prefix_string='prefix'+name[0:min(m+1,len(name))]
        random.seed(prefix_string)
        # prefix_index = [int(random.randint(0,d-1)) for _ in range(dim)]
        prefix_index = int(random.randint(0, d-1))
        v[prefix_index] = 1
    
    suffix_max = 3
    for m in range(suffix_max):
        suffix_string='suffix'+name[-1:-min(m+2,len(name)+1):-1]
        random.seed(suffix_string)
        suffix_index = int(random.randint(0, d-1))
        v_1[suffix_index] = 1
        
    # v = v.flatten()

    # vowels = 'aeiou'
    # numV = 0
    # for i, char in enumerate(name):
    #     if char in vowels:
    #         numV += 1
    # VCRatio = numV / (len(name) - numV)
    # random.seed(f'ratio{VCRatio}')
    # ratioIndex = int(random.randint(0, d-1))
    # v_vowels[ratioIndex] = 1
    random.seed(f'len{len(name)}')
    lenIndex = int(random.randint(0, d-1))
    v_len[lenIndex] = 1

    for i in range(len(name)-1):
        bigram = name[i:i+2]
        random.seed('bigram' + bigram)
        bi_index = int(random.randint(0, d-1))
        v_bigram[bi_index] += 1 

    # bi_combi = ['ia', 'na', 'la', 'ly', 'y', 'ie', 'lle', 'ette']
    # for i in bi_combi:
    #     if i in name:
    #         random.seed(f'bicomb{i}')
    #         biIndex = int(random.randint(0, d-1))
    #         v_bigram[biIndex] = 1
    v = np.concatenate([v, v_1])
    v_combined = np.concatenate([v, v_len])
    v_combined = np.concatenate([v_combined, v_bigram])

    return v_combined
