import os 
import json

def read_word2id(entityPath_list, wordPath_list, entity2id):
    word2id = {}  #key: string of word, value: int of id 
    for i in range(len(entityPath_list)): 
        print(i)
        with open(entityPath_list[i]) as f1, open(wordPath_list[i]) as f2:
            for x, y in zip(f1, f2):
                #x is the line with number, y is the line with words 
                x, y = x.strip(), y.strip()
                numList, wordList= x.split(), y.split()
                #first word 
                head = wordList[0].split(".")[0]
                tail = wordList[-1].split(".")[0]
                if (head not in word2id): 
                    word2id[head] = entity2id[numList[0]]
                if (tail not in word2id): 
                    word2id[tail] = entity2id[numList[-1]]
    with open("word2id.json", "w") as outfile:
        json.dump(word2id, outfile)
        

def read_entity2id(entity2id_path):
    entity2id = {}
    entity2id_file = open(entity2id_path, 'r')
    lines = entity2id_file.readlines()[1:]
    for l in lines: 
        entity_id = l.split()
        entity2id[entity_id[0]] = int(entity_id[1])
    return entity2id


entityPath_list = ['/Users/minglan/Desktop/WN18RR/original/test.txt', '/Users/minglan/Desktop/WN18RR/original/train.txt', 
'/Users/minglan/Desktop/WN18RR/original/valid.txt']
wordPath_list = ['/Users/minglan/Desktop/WN18RR/text/test.txt', '/Users/minglan/Desktop/WN18RR/text/train.txt', 
'/Users/minglan/Desktop/WN18RR/text/valid.txt']
entity2id_path = '/Users/minglan/Desktop/WN18RR/entity2id.txt'

entity2id = read_entity2id(entity2id_path)
read_word2id(entityPath_list, wordPath_list, entity2id)