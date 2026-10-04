"""Hand-derived tests guarding scientific semantics and held-out isolation."""
import unittest
import numpy as np
try:
    import prediction_core as core
except ModuleNotFoundError:
    core = None

class PredictionContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, 'prediction_core implementation is absent')

    def test_student_active_fraction_and_ordinal_mean(self):
        np.testing.assert_allclose(core.features('StudentLife', np.array([0.,0.,1.,2.])), [.5,.75])

    def test_extra_magnitude_units_and_population_spread(self):
        np.testing.assert_allclose(core.features('ExtraSensory', np.array([0.,2.,4.])), [2.,np.sqrt(8/3),2.,3.6])

    def test_unknown_student_state_is_rejected_not_stationary(self):
        with self.assertRaises(ValueError):
            core.features('StudentLife', np.array([0.,3.,1.]))

    def test_nonfinite_and_negative_magnitudes_are_rejected(self):
        for bad in [np.array([1.,np.nan]),np.array([1.,-2.])]:
            with self.assertRaises(ValueError):core.features('ExtraSensory', bad)

    def test_matched_deletion_count_and_contiguous_index_run(self):
        for n in [5,11,30,61]:
            random, block = core.mask_pair(n, ('fixed',n,2))
            self.assertEqual(len(random),int(n*.2))
            self.assertEqual(len(block),int(n*.2))
            self.assertEqual(len(np.unique(random)),len(random))
            np.testing.assert_array_equal(np.diff(block),np.ones(len(block)-1))
            self.assertTrue(np.all(random<n) and np.all(block<n))

    def test_masks_repeat_with_same_identity_without_global_rng(self):
        a = core.mask_pair(100,('pid','win',4))
        np.random.seed(9);np.random.uniform(size=200)
        b = core.mask_pair(100,('pid','win',4))
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_equal_participant_total_weight_despite_unequal_rows(self):
        w=core.participant_weights(np.array(['a','a','a','b']))
        np.testing.assert_allclose(w,[2/3,2/3,2/3,2.])
        self.assertAlmostEqual(w[:3].sum(),w[3])

    def test_losses_match_hand_calculation(self):
        m=core.metrics(np.array([0,1]),np.array([.25,.75]))
        self.assertAlmostEqual(m['log_loss'],-np.log(.75))
        self.assertAlmostEqual(m['brier'],.0625)
        self.assertEqual(m['auc'],1.)
        self.assertEqual(m['balanced_accuracy'],1.)

    def test_single_class_metrics_keep_losses_but_mark_auc_and_ba_unavailable(self):
        m=core.metrics(np.array([1,1]),np.array([.8,.8]))
        self.assertTrue(np.isnan(m['auc']) and np.isnan(m['balanced_accuracy']))
        self.assertAlmostEqual(m['brier'],.04)
        self.assertAlmostEqual(m['accuracy'],1.)

    def test_probability_drift_is_not_ground_truth_loss(self):
        y=np.array([0,1]);p=np.array([.1,.9]);q=np.array([.9,.1])
        m=core.change_metrics(y,p,q)
        self.assertAlmostEqual(m['drift'],.8)
        self.assertEqual(m['flip_rate'],1.)
        self.assertGreater(m['log_loss'],core.metrics(y,p)['log_loss'])

    def test_heldout_label_changes_cannot_change_fit_or_prior(self):
        x=np.array([[0.],[1.],[0.2],[.8],[.1],[.9]])
        y=np.array([0,1,0,1,0,1]);g=np.array(['a','a','b','b','c','c'])
        first=core.fit_fold(x,y,g,'c')
        changed=y.copy();changed[-2:]=1-changed[-2:]
        second=core.fit_fold(x,changed,g,'c')
        self.assertEqual(first['fingerprint'],second['fingerprint'])
        self.assertEqual(first['prior'],second['prior'])
        self.assertEqual(first['train_participants'],['a','b'])
        np.testing.assert_array_equal(first['test_indices'],[4,5])

    def test_scoring_does_not_refit_scaler_or_model(self):
        x=np.array([[0.],[1.],[.2],[.8],[.1],[.9]])
        y=np.array([0,1,0,1,0,1]);g=np.array(['a','a','b','b','c','c'])
        fold=core.fit_fold(x,y,g,'c');before=core.fingerprint(fold)
        core.predict(fold,np.array([[100.],[200.]]))
        self.assertEqual(core.fingerprint(fold),before)

    def test_constant_comparator_has_zero_drift_but_no_discrimination(self):
        y=np.array([0,1]);p=np.array([.3,.3])
        m=core.change_metrics(y,p,p)
        self.assertEqual(m['drift'],0.)
        self.assertEqual(m['flip_rate'],0.)
        self.assertEqual(m['auc'],.5)

if __name__=='__main__':unittest.main(verbosity=2)
