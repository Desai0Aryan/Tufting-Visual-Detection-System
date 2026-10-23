import unittest

import numpy as np

from coordinates import (CoordinateError, CoordinateSystem, LocationError, TuftLocations, auto_locations)


class CoordinateSystemTests(unittest.TestCase):
    def test_origin_axis_scale_and_orientation(self):
        cs = CoordinateSystem.from_origin_axis((100, 500), (600, 500), 250, "mm")
        w = cs.to_world([(100, 500), (600, 500), (100, 400)])
        np.testing.assert_allclose(w, [[0, 0], [250, 0], [0, 50]], atol=1e-9)   # Y is up on screen
        self.assertAlmostEqual(cs.units_per_pixel(), 0.5)

    def test_y_down_option(self):
        cs = CoordinateSystem.from_origin_axis((0, 0), (100, 0), 100, y_up=False)
        np.testing.assert_allclose(cs.to_world([(0, 50)]), [[0, 50]], atol=1e-9)

    def test_rotated_axis(self):
        cs = CoordinateSystem.from_origin_axis((0, 0), (0, -100), 10)            # X points up the picture
        np.testing.assert_allclose(cs.to_world([(0, -100)]), [[10, 0]], atol=1e-9)
        v = cs.to_world_vectors([(5, 5)], [(0, -1)])
        np.testing.assert_allclose(v, [[1, 0]], atol=1e-6)

    def test_bad_input_is_explained(self):
        with self.assertRaises(CoordinateError):
            CoordinateSystem.from_origin_axis((0, 0), (1, 1), 5)
        with self.assertRaises(CoordinateError):
            CoordinateSystem.from_origin_axis((0, 0), (100, 0), "abc")
        with self.assertRaises(CoordinateError):
            CoordinateSystem.from_origin_axis((0, 0), (100, 0), -3)
        with self.assertRaises(CoordinateError):
            CoordinateSystem.from_points([(0, 0, 0, 0)])
        with self.assertRaises(CoordinateError):
            CoordinateSystem.from_points([(0, 0, 0, 0), (10, 10, 1, 1), (20, 20, 2, 2)])

    def test_affine_and_perspective(self):
        cs3 = CoordinateSystem.from_points([(0, 0, 0, 0), (100, 0, 1, 0), (0, 100, 0, 1)])
        np.testing.assert_allclose(cs3.to_world([(50, 50)]), [[0.5, 0.5]], atol=1e-9)
        src = [(0, 0), (400, 20), (380, 300), (10, 280)]
        dst = [(0, 0), (1, 0), (1, 1), (0, 1)]
        cs4 = CoordinateSystem.from_points([(*s, *d) for s, d in zip(src, dst)])
        self.assertLess(max(cs4.residuals()), 1e-9)
        np.testing.assert_allclose(cs4.to_world([(200, 150)]), [[0.50788141, 0.48007387]], atol=1e-6)
        np.testing.assert_allclose(cs4.to_pixel(cs4.to_world([(123, 77)])), [[123, 77]], atol=1e-6)

    def test_vectors_are_unit_and_follow_perspective(self):
        cs = CoordinateSystem.from_origin_axis((0, 0), (100, 0), 1)
        v = cs.to_world_vectors([(10, 10), (20, 20)], [(1, 0), (0.6, 0.8)])
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-9)

    def test_round_trip_dict(self):
        cs = CoordinateSystem.from_origin_axis((10, 20), (210, 20), 40, "cm", y_up=False, z_plane=7)
        cs2 = CoordinateSystem.from_dict(cs.to_dict())
        np.testing.assert_allclose(cs2.to_world([(55, 70)]), cs.to_world([(55, 70)]), atol=1e-9)
        self.assertEqual((cs2.units, cs2.z_plane, cs2.defined), ("cm", 7.0, True))
        self.assertFalse(CoordinateSystem.from_dict({}).defined)


class LocationTests(unittest.TestCase):
    def test_set_get_resolve(self):
        loc = TuftLocations()
        loc.set(3, 1, None, "2.5")
        self.assertEqual(loc.get(3), (1.0, None, 2.5))
        xyz, src = loc.resolve(3, (10, 20, 30))
        self.assertEqual(xyz, (1.0, 20.0, 2.5))
        self.assertEqual(src, ("user", "auto", "user"))
        loc.set(3)                                    # all blank -> back to automatic
        self.assertEqual(len(loc), 0)

    def test_rejects_non_numbers(self):
        with self.assertRaises(LocationError):
            TuftLocations().set(1, "abc")
        with self.assertRaises(LocationError):
            TuftLocations().set(1, float("nan"))

    def test_csv_round_trip_and_aliases(self):
        loc = TuftLocations.from_csv("﻿Tuft ID,X,Y,Z\n3,1,2,\n4,,,5\n")
        self.assertEqual(loc.items, {3: (1.0, 2.0, None), 4: (None, None, 5.0)})
        again = TuftLocations.from_csv(loc.to_csv())
        self.assertEqual(again.items, loc.items)
        with self.assertRaises(LocationError):
            TuftLocations.from_csv("foo,bar\n1,2\n")
        with self.assertRaises(LocationError):
            TuftLocations.from_csv("tuft_id,x\nabc,1\n")

    def test_auto_locations_uses_median_root(self):
        cs = CoordinateSystem.from_origin_axis((0, 0), (100, 0), 100, y_up=False, z_plane=4)
        got = auto_locations({1: [(10, 10), (12, 10), (400, 10)]}, cs)
        self.assertEqual(got[1], (12.0, 10.0, 4.0))


if __name__ == "__main__":
    unittest.main()
