"""Regression checks for row alignment and the confidence-interval contract."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import data
from metrics import bootstrap_ci, compute_metrics


class AlignmentTests(unittest.TestCase):
    def test_labels_follow_feature_record_order(self):
        labels = pd.DataFrame({'RecordID':[10,20,30], 'split':['train']*3, 'In-hospital_death':[1,0,0]})
        features = pd.DataFrame({'RecordID':[30,10,20], 'measurement':[3.,1.,2.]})
        with patch.object(data, 'load_data', return_value=(labels, features, {'train':features})):
            x, y = data.get_split('train')
        self.assertEqual(x.measurement.tolist(),[3.,1.,2.])
        self.assertEqual(y.tolist(),[0,1,0])
        self.assertNotIn('RecordID', x.columns)

    def test_wrong_split_membership_is_rejected(self):
        labels = pd.DataFrame({'RecordID':[10,20], 'split':['train','test'], 'In-hospital_death':[1,0]})
        features = pd.DataFrame({'RecordID':[10,20], 'measurement':[1.,2.]})
        with patch.object(data, 'load_data', return_value=(labels, features, {'train':features})):
            with self.assertRaisesRegex(ValueError,'RecordID mismatch'):
                data.get_split('train')


class MetricTests(unittest.TestCase):
    def test_bootstrap_contract_is_point_low_high(self):
        y=np.array([0]*10+[1]*10)
        score=np.array([.1,.4,.3,.7,.2,.8,.6,.5,.1,.3,.8,.9,.4,.7,.2,.6,.9,.8,.5,.7])
        point, low, high=bootstrap_ci(y,score,roc_auc_score,n_bootstrap=100)
        self.assertEqual(point,roc_auc_score(y,score))
        self.assertLess(low,point)
        self.assertGreater(high,point)
        self.assertEqual((point,low,high),bootstrap_ci(y,score,roc_auc_score,n_bootstrap=100))

    def test_threshold_confusion_counts(self):
        m=compute_metrics([0,0,1,1],[.2,.9,.7,.1],threshold=.5)
        self.assertEqual(m['sensitivity'],.5)
        self.assertEqual(m['ppv'],.5)
        self.assertEqual(m['specificity'],.5)


if __name__ == '__main__':
    unittest.main()
