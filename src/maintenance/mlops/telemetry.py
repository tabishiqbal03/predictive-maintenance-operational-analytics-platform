from maintenance.mlops.store import connection, initialize, now


def record_event(root, request_id, version, status_code, latency_ms, prediction=None, error_type=None):
    initialize(root)
    prediction = prediction or {}
    with connection(root) as db:
        db.execute(
            """INSERT INTO inference_events
                   (request_id,timestamp,model_version,status_code,latency_ms,predicted_rul,maintenance_status,error_type)
                   VALUES (?,?,?,?,?,?,?,?)""",
            (
                request_id,
                now(),
                version,
                status_code,
                latency_ms,
                prediction.get("predicted_rul"),
                prediction.get("maintenance_status"),
                error_type,
            ),
        )
