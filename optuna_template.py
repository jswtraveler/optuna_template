
import optuna
import xgboost as xgb
import sklearn
import datetime
from sklearn.metrics import mean_squared_error
def objective(trial):
    
    
    param = {
        #'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        "objective": 'reg:squarederror',
        'eval_metric':'mae',
        'reg_lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'reg_alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'gamma':trial.suggest_loguniform('gamma', 1e-3,1),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [i/10.0 for i in range(4,11)]),
        'subsample': trial.suggest_categorical('subsample', [i/10.0 for i in range(4,11)]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02,0.300000012]),
        'n_estimators': trial.suggest_int('n_estimators',100,500),
        'max_depth': trial.suggest_categorical('max_depth', [5,6,7,9,11,13,15,17,20]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBRegressor(**param)  
    
    model.fit(X,Y)#,eval_set=[(X,Y)],early_stopping_rounds=100,verbose=False)
    
    preds = model.predict(X)#X_test)
    
    rmse = mean_squared_error(Y,preds,squared=False)#xgb_base_test_y[tar], preds,squared=False)
    
    return rmse

pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study(pruner=pruner, direction="minimize")
study.enqueue_trial(default_params)
start_time = datetime.datetime.now()
#study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=800)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
end_time = datetime.datetime.now()
print(str(end_time-start_time))
