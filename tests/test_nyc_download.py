"""Download profiles must keep provenance for previously acquired context files."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_nyc_bike


class NYCDownloadTests(unittest.TestCase):
    def test_training_profile_retains_existing_optional_file_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            optional = {"path": "Data/AQI/previous.csv", "sha256": "previous", "bytes": 42}
            manifest_path = root / "download_manifest.json"
            manifest_path.write_text(json.dumps({
                "commit": download_nyc_bike.COMMIT, "files": [optional],
            }), encoding="utf-8")

            def downloaded(folder, relative):
                self.assertEqual(folder, root)
                return {"path": relative, "sha256": "current", "bytes": 10}

            arguments = ["download_nyc_bike.py", "--profile", "training", "--output-dir", str(root)]
            with patch("sys.argv", arguments), \
                    patch.object(download_nyc_bike, "download", side_effect=downloaded) as fetch, \
                    contextlib.redirect_stdout(io.StringIO()):
                download_nyc_bike.main()

            requested = {call.args[1] for call in fetch.call_args_list}
            self.assertEqual(requested, set(download_nyc_bike.TRAINING_PATHS))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = {record["path"]: record for record in manifest["files"]}
            self.assertEqual(records[optional["path"]], optional)
            self.assertEqual(set(records), requested | {optional["path"]})
            self.assertEqual(manifest["profile"], "training")


if __name__ == "__main__":
    unittest.main()
