import copy
import unittest
import numpy as np
import pandas as pd
from match_predictor.data import validate_matches
from match_predictor.features import build_features, feature_columns
from match_predictor.models import MatchModel, blend_market, goal_grid, outcomes, reconcile_outcomes, temperature_scale
from match_predictor.predict import predict_match
from match_predictor.selection import apply_policy, double_chance_metrics, double_chance_pick, fit_policy
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
    def test_market_blend_preserves_score_outcome_consistency(self):
        grid=goal_grid([[2,1]])
        frame=pd.DataFrame({"market_home":[.7],"market_draw":[.2],"market_away":[.1]})
        original=outcomes(grid)[0]
        updated=blend_market(grid,frame,.5)
        np.testing.assert_allclose(outcomes(updated)[0],.5*original+.5*np.array([.7,.2,.1]))
        np.testing.assert_allclose(updated.sum(),1)

class SelectionTests(unittest.TestCase):
    def test_double_chance_excludes_only_the_least_likely_outcome(self):
        pick=double_chance_pick([.52,.29,.19])
        self.assertEqual(pick["code"],"1X")
        self.assertEqual(pick["included_classes"],[0,1])
        self.assertAlmostEqual(pick["probability"],.81)
        frame=pd.DataFrame({"actual_class":[0,1,0],"p_stats_home":[.6,.5,.6],
            "p_stats_draw":[.3,.2,.3],"p_stats_away":[.1,.3,.1]})
        measured=double_chance_metrics(frame,"stats")
        self.assertEqual(measured["selected_matches"],3)
        self.assertEqual(measured["correct"],2)
        self.assertEqual(measured["coverage"],1.0)

    def test_policy_abstains_below_threshold_and_without_market(self):
        policy={"threshold":.625,"min_team_history":10,"require_agreement":True}
        frame=pd.DataFrame({"confidence":[.7,.6,.8],"home_experience":[20,20,20],
                            "away_experience":[20,20,20],"eligible":[True,True,False],
                            "agreement":[True,True,True]})
        np.testing.assert_array_equal(apply_policy(frame,policy),[True,False,False])
    def test_threshold_is_fit_from_supplied_development_rows(self):
        frame=pd.DataFrame({"confidence":np.r_[np.repeat(.65,240),np.repeat(.55,160)],
            "predicted_class":0,"actual_class":np.r_[np.zeros(200),np.ones(40),np.zeros(80),np.ones(80)],
            "home_experience":30,"away_experience":30,"eligible":True})
        policy=fit_policy(frame,target=.70,min_matches=120)
        self.assertIsNotNone(policy["threshold"])
        self.assertGreater(policy["threshold"],.55)

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
        bundle={'schema_version':3,'teams':['A','B','C','D'],'recent_teams':['A','B','C','D'],'columns':cols,'engine':engine,'models':{'rolling_poisson':model},'weights':{'rolling_poisson':1.0},'temperature':1.,'market_weight':.8,'selective_policies':{},'league':'test','last_match_date':str(frame.date.max().date())}
        date=frame.date.max()+pd.Timedelta(days=3)
        before=copy.deepcopy(engine.teams['A'])
        result=predict_match(bundle,'A','B',date)
        second=predict_match(bundle,'A','C',date)
        self.assertAlmostEqual(sum(result['probabilities'].values()),1)
        self.assertAlmostEqual(result['double_chance']['probability'],
                               1-min(result['probabilities'].values()))
        self.assertNotEqual(result['probabilities'],second['probabilities'])
        self.assertEqual(engine.teams['A'].elo,before.elo)
        self.assertEqual(engine.teams['A'].matches,before.matches)
        assisted=predict_match(bundle,'A','B',date,[1.5,4.5,7])
        self.assertEqual(assisted['prediction_mode'],'assisted')
        self.assertIsNotNone(assisted['market_probabilities'])
        with self.assertRaises(ValueError): predict_match(bundle,'A','B',date,[1.0,4,5])
        with self.assertRaises(ValueError): predict_match(bundle,'A','A',date)
        with self.assertRaises(ValueError): predict_match(bundle,'A','Unknown',date)
        with self.assertRaises(ValueError): predict_match(bundle,'A','B',frame.date.max())

if __name__=='__main__': unittest.main()
