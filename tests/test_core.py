"""Dependency-light regression tests for Digital Eye's core services."""
from contextlib import closing
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import database
import email_alerts
import retention


class DatabaseTests(unittest.TestCase):
    def test_camera_name_is_saved_per_incident(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "test.db")
            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.add_incident(
                    "2026-11-21 12:00:00", 1, 0, "snapshot.jpg",
                    "intrusion", "Front door", "event.mp4",
                    camera="Front Gate",
                )
                with closing(sqlite3.connect(db_path)) as db:
                    result = db.execute(
                        "SELECT camera, alert_type, details, clip_path FROM incidents"
                    ).fetchone()
            self.assertEqual(
                result, ("Front Gate", "intrusion", "Front door", "event.mp4")
            )


class RetentionTests(unittest.TestCase):
    def test_only_old_supported_media_is_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            incidents = Path(directory) / "incidents"
            recordings = Path(directory) / "recordings"
            incidents.mkdir()
            recordings.mkdir()
            old_image = incidents / "old.jpg"
            new_image = incidents / "new.jpg"
            old_text = incidents / "keep.txt"
            for item in (old_image, new_image, old_text):
                item.write_bytes(b"test")
            now = time.time()
            os.utime(old_image, (now - 16 * 86400, now - 16 * 86400))
            with patch.object(retention, "INCIDENTS_DIR", str(incidents)), \
                 patch.object(retention, "RECORDINGS_DIR", str(recordings)):
                removed = retention.cleanup_expired_media(15, now=now)
            self.assertIn(str(old_image), removed)
            self.assertFalse(old_image.exists())
            self.assertTrue(new_image.exists())
            self.assertTrue(old_text.exists())

    def test_protected_media_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            incidents = Path(directory) / "incidents"
            recordings = Path(directory) / "recordings"
            incidents.mkdir()
            recordings.mkdir()
            protected = incidents / "active.mp4"
            protected.write_bytes(b"active")
            old = time.time() - 20 * 86400
            os.utime(protected, (old, old))
            with patch.object(retention, "INCIDENTS_DIR", str(incidents)), \
                 patch.object(retention, "RECORDINGS_DIR", str(recordings)):
                removed = retention.cleanup_expired_media(
                    15, protected_paths=[str(protected)], now=time.time()
                )
            self.assertEqual(removed, [])
            self.assertTrue(protected.exists())


class EmailTests(unittest.TestCase):
    def test_missing_email_configuration_fails_gracefully(self):
        with patch.object(email_alerts, "email_is_configured", return_value=False):
            self.assertFalse(email_alerts.send_email("subject", "body"))


class CameraConfigTests(unittest.TestCase):
    def test_config_round_trip_and_independent_slots(self):
        from camera_config import CameraConfigStore
        with tempfile.TemporaryDirectory() as directory:
            store = CameraConfigStore(Path(directory) / "cameras.json")
            store.save(2, name="Front Gate", source="rtsp://camera/stream", enabled=True)
            store.save(3, name="Cash Counter", source="1", enabled=True)
            self.assertEqual(store.get(2)["name"], "Front Gate")
            self.assertEqual(store.get(2)["source"], "rtsp://camera/stream")
            self.assertEqual(store.get(3)["name"], "Cash Counter")
            self.assertEqual(store.get(4)["source"], "")

    def test_invalid_camera_slot_is_rejected(self):
        from camera_config import CameraConfigStore
        with tempfile.TemporaryDirectory() as directory:
            store = CameraConfigStore(Path(directory) / "cameras.json")
            with self.assertRaises(ValueError):
                store.save(5, name="Invalid")

    def test_values_are_clamped(self):
        from camera_config import CameraConfigStore
        with tempfile.TemporaryDirectory() as directory:
            store = CameraConfigStore(Path(directory) / "cameras.json")
            config = store.save(2, width=99999, height=1, fps=100, confidence=4)
            self.assertEqual(config["width"], 3840)
            self.assertEqual(config["height"], 120)
            self.assertEqual(config["fps"], 60)
            self.assertEqual(config["confidence"], 0.99)

    def test_numeric_saved_source_is_restored_as_camera_index(self):
        from camera_config import CameraConfigStore
        from camera_config import parse_camera_source
        with tempfile.TemporaryDirectory() as directory:
            store = CameraConfigStore(Path(directory) / "cameras.json")
            store.save(2, source="1", enabled=True)
            self.assertEqual(parse_camera_source(store.get(2)["source"]), 1)


if __name__ == "__main__":
    unittest.main()
