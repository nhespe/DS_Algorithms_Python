""" HMM Model w/ Viterbi Decoding for HW6

"""
#builtins
import argparse
import collections
import re

from scorer import score

import numpy as np
from jellyfish import jaro_winkler, levenshtein_distance

SPECIAL_CHARS = '[()\[\]{},./?><*&^%$#@!;|\\""''``]'

class HMM:
    default_small_value = 0.00001

    def __init__(self):
        self.all_trigram_pos = None
        self.all_bigram_pos = None
        self.trigram_pos_count_dict = None
        self.bigram_pos_count_dict = None
        self.word_pos_counts = None
        self.word_counts_dict = None
        self.word_pos_dict = collections.defaultdict(set)
        self.bigram_pos_dict = collections.defaultdict(set)

    def clean_tuple(self, labeled_word):
        # First remove specials
        if labeled_word and len(labeled_word)>1:
            word, pos = labeled_word

            #Matches on all numbers 
            if re.match(r"-?\d+(,\d+)*(\.\d+(e\d+)?)?$", word.strip()):
                return ("#", pos.strip())

            # timestamp
            if re.match(r"-?\d+(,\d+)*(\:\d+(e\d+)?)?$", word):
                return ("TIMESTAMP", pos.strip())

            # Matches on all numbers #(th, rd, st, nd)
            elif re.match(r"(?i)(?<!\S)\d+(?:th|rd|st|nd)\b", word):
                return ("#th", pos.strip()) 

            return (word.strip(), pos.strip())

        return None

    def clean_word(self, word):
        """ Cleans a word by processing it into its respective bins """

        if word:
            #Matches on all numbers 
            if re.match(r"-?\d+(,\d+)*(\.\d+(e\d+)?)?$", word.strip()):
                return "#"

            # timestamp
            if re.match(r"-?\d+(,\d+)*(\:\d+(e\d+)?)?$", word):
                return "TIMESTAMP"

            # Matches on all numbers #(th, rd, st, nd)
            elif re.match(r"(?i)(?<!\S)\d+(?:th|rd|st|nd)\b", word):
                return "#th"

            return word.strip()

        return None

    def generate_trigrams(self, sentence):
        """ Given a sentence, return a list of trigrams using a sliding window approach """
        working_list = ["START", "START"] + sentence
        output_list = []

        for i in range(len(sentence)):
            output_list.append(tuple(working_list[i:i+3]))
        return output_list

    def generate_bigrams(self, sentence):
        """ Given a sentence, return a list of biigrams using a sliding window approach """
        working_list = ["START", "START"] + sentence
        output_list = []
        
        for i in range(len(sentence)):
            output_list.append(tuple(working_list[i:i+2]))
        return output_list

    def load_data(self, file_name, words_only=False, dont_clean=False):
        """ """
        all_sentences = []
        with open(file_name) as inpt:
            sentence = []
            for line in inpt:
                if line=="\n":
                    all_sentences.append(list(filter(lambda x:x not in [None, ""], sentence)))
                    sentence = [] # reset sentence

                if words_only and not dont_clean:
                    sentence.append(self.clean_word(line.strip()))
                elif words_only and dont_clean:
                    sentence.append(line.strip())
                else:
                    sentence.append(self.clean_tuple(line.split("\t")))

            all_sentences.append(list(filter(lambda x:x not in [None, ""], sentence)))
        return all_sentences

    def get_max_prob_(self, tag2, tag1, word1):
        """
        This performs the viterbi step for any tag1, tag2, word1

        """
        # If we dont have the word, try to find it
        if word1 not in self.word_counts_dict:
            word1 = self.find_word(word1)

        # Parse hello-world to hello world
        if word1 not in self.word_counts_dict and "-" in word1:
            word1 = self.find_word(word1.split("-")[-1])

        # bins for the outcomes, and the corresponding tags
        outcomes, target = [], []
        for possible_tag in self.bigram_pos_dict.get((tag2, tag1), []):
            trigram_count = self.trigram_pos_count_dict.get((tag2, tag1, possible_tag), 1)
            bigram_count = self.bigram_pos_count_dict.get((tag2, tag1), np.inf)
            prob_pos_trig = trigram_count/bigram_count

            outcomes.append(prob_pos_trig * self.get_prob_tag_given_word(word1, possible_tag)), 
            target.append(possible_tag)

        if outcomes:
            maxindex = np.argmax(outcomes)
            return target[maxindex]

        return "NNP" # return NNP as default
        
    def find_word(self, word):
        """ IF WE ARE MISSING THE WORD - SEE IF ITS STEM IS IN THE LIST """
        if len(word) > 3 and word[-3:] == "ing" and word[:-3] in self.word_counts_dict:
            return word[:-3]
        elif word[:-1] in self.word_counts_dict: # s ending check
            return word[:-1]
        elif word[:-2] in self.word_counts_dict: # -ed ending
            return word[:-2]

        new_word = self.find_mispelled_word_jw(word)
        
        if new_word:
            return new_word
        return word

    def find_mispelled_word_lev(self, word):
        """ """
        for target in self.word_counts_dict:
            if levenshtein_distance(word, target) < 2:
                return target
        return None

    def find_mispelled_word_jw(self, word):
        """ """
        outcomes, scores = [], []
        words = list(self.word_counts_dict)
        
        for target in words:
            local_score = jaro_winkler(word, target)
            if local_score >= 0.95:
                scores.append(local_score)
                outcomes.append(target)
        if outcomes:
            return outcomes[np.argmax(scores)]
        return None

    def get_prob_tag_given_word(self, word, pos_tag):
        """ Given a word, return the tag with the highest probability """
        if not word in self.word_counts_dict:
            return self.pos_default_occurences.get(pos_tag, self.default_small_value)
        return self.word_pos_counts.get((word, pos_tag), 0) / self.pos_counts_dict.get(pos_tag, np.inf)

    def viterbi_decode(self, sentence):
        splitsentence = [self.clean_word(x) for x in sentence.split(" ")]
        pos_tags = ["START", "START"]
        
        for word in splitsentence:
            pos_tags.append(self.get_max_prob_(*pos_tags[-2:], word))
        return pos_tags[2:] 

    def tag_sentence(self, sentences, words_only=False):
        scores, labels, sentence = [], [], []
        for i in range(len(sentences)):   
            if not words_only:
                output = self.viterbi_decode(" ".join([x[0] for x in sentences[i]]))
                prop_match = sum([(x==y) for x,y in zip(output, [x[1] for x in sentences[i]])])/len(output)
                scores.append(prop_match)
                labels.append([(x,y) for x,y in zip(output, [x[1] for x in sentences[i]])])
                sentence.append(sentences[i])
            else:
                output = self.viterbi_decode(" ".join(sentences[i]))
                prop_match = -1
                scores.append(prop_match)
                labels.append([(x,y) for x,y in zip(output, [x for x in sentences[i]])])
                sentence.append(sentences[i])
            
        return scores, labels, sentence

    def train(self, all_sentences):
        # A list of all Trigrams and Bigrams
        self.all_trigram_pos = [word for sentence in all_sentences for word in self.generate_trigrams([x[1] for x in sentence])]
        self.all_bigram_pos = [word for sentence in all_sentences for word in self.generate_bigrams([x[1] for x in sentence])]

        # A list of the counts of all trigrams and Bigrams, respectively
        self.trigram_pos_count_dict = collections.Counter(self.all_trigram_pos)
        self.bigram_pos_count_dict = collections.Counter(self.all_bigram_pos)
        
        # A list of individual counts (word, pos), word, pos
        self.word_pos_counts = collections.Counter([word for sentence in all_sentences for word in sentence])
        self.word_counts_dict = collections.Counter([word[0] for sentence in all_sentences for word in sentence])
        self.pos_counts_dict = collections.Counter([word[1] for sentence in all_sentences for word in sentence])

        # For every word, get all of the pos tags that its involved
        for pair in [word for sentence in all_sentences for word in sentence]:
            word, pos = pair
            self.word_pos_dict[word].add(pos)
            
        # For every pos pair, get the pos tag that is left
        for trigram in self.all_trigram_pos:
            bigram = (trigram[0], trigram[1])
            next_pos = trigram[2]
            self.bigram_pos_dict[bigram].add(next_pos)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        type=str,
        help="Source input file for training data"
    )
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        type=str,
        help="Target input file for classification data"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default="output.txt",
        type=str,
        help="Target input file for classification data"
    )
    parser.add_argument(
        "-f",
        "--target_format",
        required=False,
        action="store_true",
        default=False,
        help="How many elements per line?"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    hmm = HMM()

    print(f"Training HMM with data from '{args.source}'")
    training_sentences = hmm.load_data(args.source)
    hmm.train(training_sentences)

    print(f"Starting classification with data from '{args.target}'")
    test_sentences = hmm.load_data(args.target, words_only=args.target_format)
    test_sentences_unclean = hmm.load_data(args.target, words_only=args.target_format, dont_clean=True)
    
    print(f"\tFound {len(test_sentences)} sentences in the file")
    scores, labels, sentences = hmm.tag_sentence(test_sentences, words_only=args.target_format)

    print(f"Writing Output to '{args.output}'")
    with open(args.output, "w") as outfile:
        for idx, tup in enumerate(zip(labels, test_sentences_unclean)):
            labeled_sent, sentence = tup
            
            # Start Sentence
            for label, word in zip(labeled_sent, sentence):
                if not args.target_format:
                    outfile.write(f"{word[0]}\t{label[0]}\n")
                    continue
                outfile.write(f"{word}\t{label[0]}\n")
            # End Sentence
            if not idx == (len(sentences)-1):
                outfile.write(f"\n")
    if not args.target_format:
        print(score(args.target, args.output))

if __name__=="__main__":
    main()
