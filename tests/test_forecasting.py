import copy
import unittest
import numpy as np
import pandas as pd
from match_predictor.data import validate_matches
from match_predictor.features import build_features, feature_columns
from match_predictor.models import MatchModel, goal_grid, outcomes, reconcile_outcomes, temperature_scale
from match_predictor.predict import predict_match
from match_predictor.training import chronological_partitions

def sample_matches(n=160):
    rng=np.random.default_rng(14)
    rows=[]
    for i in range(n):
        h,a=rng.choice(['A','B','C','D'],2,replace=False)
        rows.append({'date':pd.Timestamp('2023-01-01',tz='UTC')+pd.Timedelta(days=i*2), 'home_team':h,'away_team':a,'home_goals':int(rng.poisson(1.6)),'away_goals':int(rng.poisson(1.2))})
    return validate_matches(pd.DataFrame(rows))

class LeakageTests(unittest.TestCase):
    def test_future_results_do_not_change_earlier_features(self):
        matches=sample_matches()
        original,_=build_features(matches)
        altered=matches.copy()
        altered.loc[80:,['home_goals','away_goals']]=[8,7]
        altered.loc[80:,['home_shots','away_shots','home_sot','away_sot']]=[40,35,20,15]
        changed,_=build_features(altered)
        pd.testing.assert_frame_equal(original.loc[:80,feature_columns(original)],changed.loc[:80,feature_columns(changed)])
    def test_prefix_equals_full_history_features(self):
        matches=sample_matches()
        full,_=build_features(matches)
        prefix,_=build_features(matches.iloc[:70])
        pd.testing.assert_frame_equal(full.iloc[:70],prefix)
    def test_same_day_results_are_not_visible(self):
        matches=sample_matches(4)
        matches.loc[1,'date']=matches.loc[0,'date']
        before,_=build_features(matches)
        matches.loc[0,'home_goals']=20
        after,_=build_features(matches)
        pd.testing.assert_frame_equal(before.loc[:1,feature_columns(before)],after.loc[:1,feature_columns(after)])
    def test_training_and_live_features_agree(self):
        matches=sample_matches()
        frame,_=build_features(matches)
        _,engine=build_features(matches.iloc[:60])
        row=matches.iloc[60]
        engine.prepare_season(row.season)
        live=engine.snapshot(row.home_team,row.away_team,row.date)
        for key,value in live.items(): self.assertEqual(value,frame.loc[60,key])
    def test_date_partitions_never_overlap(self):
        frame,_=build_features(sample_matches())
        dev,test,folds,_=chronological_partitions(frame)
        self.assertLess(frame.loc[dev].date.max(),frame.loc[test].date.min())
        for train,valid in folds:
            self.assertLess(frame.loc[train].date.max(),frame.loc[valid].date.min())
            self.assertLess(frame.loc[valid].date.max(),frame.loc[test].date.min())
    def test_incomplete_new_season_is_not_test_set(self):
        chunks=[]
        for season in range(2018,2024):
            chunk=sample_matches(220)
            chunk['date']=pd.date_range(f'{season}-07-01',periods=220,tz='UTC')
            chunk['season']=season
            chunks.append(chunk)
        latest=sample_matches(20)
        latest['date']=pd.date_range('2024-08-01',periods=20,tz='UTC')
        latest['season']=2024
        frame,_=build_features(pd.concat(chunks+[latest],ignore_index=True))
        dev,test,folds,_=chronological_partitions(frame)
        self.assertEqual(set(frame.loc[test].season),{2023})
        self.assertLess(frame.loc[dev].season.max(),2023)

class ProbabilityTests(unittest.TestCase):
    def test_grid_mass_and_orientation(self):
        grid=goal_grid([[3,.5],[.5,3],[8,8],[.05,.05]])
        np.testing.assert_allclose(grid.sum(axis=(1,2)),1)
        self.assertTrue((grid>=0).all())
        p=outcomes(grid)
        self.assertGreater(p[0,0],p[0,2])
        self.assertGreater(p[1,2],p[1,0])
        np.testing.assert_allclose(p.sum(axis=1),1)
    def test_score_and_outcome_probabilities_agree(self):
        grid=goal_grid([[2,1]])
        p=np.array([[.6,.25,.15]])
        updated=reconcile_outcomes(grid,p)
        np.testing.assert_allclose(outcomes(updated),p)
        np.testing.assert_allclose(temperature_scale(updated,1),updated)
        self.assertLess(outcomes(temperature_scale(updated,1.5)).max(),p.max())

class DataAndPredictionTests(unittest.TestCase):
    def test_unplayed_knockout_placeholders_are_omitted(self):
        matches=sample_matches(5)
        placeholder=pd.DataFrame([{'date':None,'home_team':None,'away_team':None,'home_goals':np.nan,'away_goals':np.nan}])
        result=validate_matches(pd.concat([matches,placeholder],ignore_index=True))
        self.assertEqual(len(result),5)
        matches.loc[0,'home_team']=None
        with self.assertRaises(ValueError): validate_matches(matches)
    def test_duplicate_and_partial_results_rejected(self):
        matches=sample_matches(5)
        with self.assertRaises(ValueError): validate_matches(pd.concat([matches,matches.iloc[:1]]))
        matches.loc[0,'home_goals']=np.nan
        with self.assertRaises(ValueError): validate_matches(matches)
    def test_future_match_uses_requested_opponent_and_leaves_state_unchanged(self):
        frame,engine=build_features(sample_matches())
        cols=feature_columns(frame)
        model=MatchModel('rolling_poisson',cols).fit(frame)
        bundle={'schema_version':2,'teams':['A','B','C','D'],'recent_teams':['A','B','C','D'],'columns':cols,'engine':engine,'models':{'rolling_poisson':model},'weights':{'rolling_poisson':1.0},'temperature':1.,'last_match_date':str(frame.date.max().date())}
        date=frame.date.max()+pd.Timedelta(days=3)
        before=copy.deepcopy(engine.teams['A'])
        result=predict_match(bundle,'A','B',date)
        second=predict_match(bundle,'A','C',date)
        self.assertAlmostEqual(sum(result['probabilities'].values()),1)
        self.assertNotEqual(result['probabilities'],second['probabilities'])
        self.assertEqual(engine.teams['A'].elo,before.elo)
        self.assertEqual(engine.teams['A'].matches,before.matches)
        with self.assertRaises(ValueError): predict_match(bundle,'A','A',date)
        with self.assertRaises(ValueError): predict_match(bundle,'A','Unknown',date)
        with self.assertRaises(ValueError): predict_match(bundle,'A','B',frame.date.max())

if __name__=='__main__': unittest.main()
