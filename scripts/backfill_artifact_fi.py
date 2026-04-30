"""Backfill feature_importances on existing model_artifacts rows."""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db as DB
from model.predict import _artifact_feature_importances


def main():
    DB.init_model_artifacts_table()
    conn = DB.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, artifact_bytes, feature_columns, feature_importances "
                "FROM model_artifacts ORDER BY id"
            )
            rows = cur.fetchall()
        updated = 0
        for row_id, blob, feat_cols, existing_fi in rows:
            if existing_fi:
                continue
            try:
                bundle = pickle.loads(bytes(blob))
                clf = bundle["clf"]
                feats = list(bundle.get("active_feats") or feat_cols or [])
                fi = _artifact_feature_importances(clf, feats)
            except Exception as e:
                print(f"  id={row_id} skipped: {e}")
                continue
            if not fi:
                print(f"  id={row_id} no FI extractable")
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE model_artifacts SET feature_importances = %s::jsonb WHERE id = %s",
                    (json.dumps(fi), row_id),
                )
            conn.commit()
            updated += 1
            print(f"  id={row_id} updated ({len(fi)} features)")
        print(f"Done. Updated {updated}/{len(rows)}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
