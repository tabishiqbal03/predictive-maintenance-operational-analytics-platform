-- Latest fleet distribution, maintained as a view by the pipeline
SELECT * FROM fleet_status;
-- Maintenance review queue
SELECT engine_id, cycle, predicted_rul, maintenance_status
FROM maintenance_alerts ORDER BY predicted_rul;
-- Lowest RUL assets, including monitor/healthy where relevant
SELECT * FROM current_predictions ORDER BY predicted_rul LIMIT 10;
-- Historical run metadata and prediction summaries
SELECT r.run_id, r.created_at, COUNT(p.engine_id) AS assets,
       AVG(p.predicted_rul) AS mean_rul, AVG(ABS(p.error)) AS endpoint_mae
FROM model_runs r JOIN model_predictions p USING(run_id)
GROUP BY r.run_id ORDER BY r.created_at DESC;
