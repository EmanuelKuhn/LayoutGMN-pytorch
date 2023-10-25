# File to define a dataset based on a list of test triplets

import os
import pickle

import pandas as pd

from inference.sg_dataset import SGDataset

class TestTripletDataset:

    def __init__(self, sg_dataset: SGDataset, test_triplets_csv: str) -> None:
        
        test_triplets = pd.read_csv(test_triplets_csv).to_dict(orient="records")

        self.test_triplets_dict = self.make_test_triplets_dict(test_triplets)

        self.sg_dataset = sg_dataset

    @staticmethod
    def make_test_triplets_dict(test_triplets):
        """Filter test triplets based on whether the corresponding npy files exist"""

        test_triplets_dict = {"anchor": [], "positive": [], "negative": []}

        for test_triplet in test_triplets:
            anchor = test_triplet["anchor"]
            pos = test_triplet["positive"]
            neg = test_triplet["negative"]
            try:
                for id in [anchor, pos, neg]:
                    assert os.path.exists(f'../training_scripts/fp_data/geometry-directed/{id}.npy')
            except:
                continue
            
            test_triplets_dict["anchor"].append(anchor)
            test_triplets_dict["positive"].append(pos)
            test_triplets_dict["negative"].append(neg)
        
        assert len(test_triplets_dict["anchor"]) > 0, "No valid test triplets found"

        return test_triplets_dict
    
    def __len__(self):
        return len(self.test_triplets_dict["anchor"])
    
    def get_triplets_batch(self, anchors, positives, negatives):
        """Get a batch of triplets specified by the anchor, positive and negative ids"""

        assert len(anchors) == len(positives) == len(negatives), "Number of anchors, positives and negatives must be the same"

        anchors = [str(anchor) for anchor in anchors]
        positives = [str(positive) for positive in positives]
        negatives = [str(negative) for negative in negatives]


        queries = self.sg_dataset.get_batch(anchors)
        positives = self.sg_dataset.get_batch(positives)
        negatives = self.sg_dataset.get_batch(negatives)

        data = {
            "sg_data_a": queries,
            "sg_data_p": positives,
            "sg_data_n": negatives
        }

        assert len(data['sg_data_a']) == len(data['sg_data_p']) == len(data['sg_data_n'])
        assert len(data['sg_data_a']) > 0, f"sg_data_a is empty for; {data['sg_data_a']=}"

        return data

    def get_batch_of_all_triplets(self):
        """Get a batch of all test triplets"""

        return self.get_triplets_batch(self.test_triplets_dict["anchor"], self.test_triplets_dict["positive"], self.test_triplets_dict["negative"])