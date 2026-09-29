"""Headless tests for Weight Transfer (magic_fit/robust_transfer).

Ported from the standalone add-on's tests/test_blender.py. They need SciPy and robust-laplacian:
tools/vendor_wheels.py unpacks them into deps/, which this script puts on sys.path (or pass another
folder after --, such as the site-packages an installed extension's wheels went to). Run from the
repository root:
    blender -b --factory-startup --python tests/test_transfer_core.py [-- deps folder]
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else os.path.join(ROOT, 'deps')
if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import magic_fit  # noqa: E402

# Registering must neither touch sys.path (extensions mustn't) nor load SciPy (the brushes shouldn't
# wait for it). Checked in AddonTests.
PATH_BEFORE_REGISTER = list(sys.path)
magic_fit.register()
PATH_AFTER_REGISTER = list(sys.path)
SCIPY_AFTER_REGISTER = 'scipy' in sys.modules

from magic_fit.robust_transfer import dependencies, operators, seams, transfer, util  # noqa: E402
from magic_fit.robust_transfer import weighttransfer as wt  # noqa: E402


def split_surface():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                  [1, 0, 0], [2, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]])
    w = np.tile([0.25, 0.75], (8, 1))
    matched = np.array([True] * 4 + [False] * 4)
    return v, f, w, matched


def grid(size, length=1.0):
    """A wavy triangulated grid along x, as vertices, faces and x coordinates."""
    xs, ys = np.meshgrid(np.linspace(0, length, size), np.linspace(0, 1, size), indexing='ij')
    x, y = xs.ravel(), ys.ravel()
    v = np.column_stack([x, y, 0.05 * np.sin(3 * x) * np.cos(2 * y)])
    index = np.arange(size * size).reshape(size, size)
    f = np.vstack([
        np.column_stack([index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel()]),
        np.column_stack([index[:-1, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()])])
    return v, f, x


def flat_grid(xs, ys, z=0.0):
    """A flat triangulated grid over the coordinates xs and ys, as vertices and faces."""
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    v = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    index = np.arange(X.size).reshape(X.shape)
    f = np.vstack([
        np.column_stack([index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel()]),
        np.column_stack([index[:-1, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()])])
    return v, f


class NumericalTests(unittest.TestCase):
    def test_partial_distance_strength(self):
        distance = np.array([0, 0.5, 0.8, 0.85, 0.9, 0.95, 1, 2])
        np.testing.assert_allclose(transfer.distance_strength(distance ** 2, 1, 20),
                                   [1, 1, 1, 0.84375, 0.5, 0.15625, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(transfer.distance_strength(distance ** 2, 1, 0),
                                      [1, 1, 1, 1, 1, 1, 1, 0])
        np.testing.assert_allclose(transfer.distance_strength(np.array([0, .25, 1, 4]), 1, 100),
                                   [1, .5, 0, 0])
        for falloff in [0, 20, 100]:
            np.testing.assert_array_equal(transfer.distance_strength(distance ** 2, 0, falloff),
                                          [1, 0, 0, 0, 0, 0, 0, 0])
        for radius, falloff in [(-1, 20), (np.nan, 20), (1, 101)]:
            with self.assertRaises(ValueError):
                transfer.distance_strength(distance ** 2, radius, falloff)

    def test_partial_solve_does_not_use_rejected_constraints(self):
        v, f, w, matched = split_surface()
        matched[1:4] = False
        w[1:4] = [1, 0]
        w[4:] = [0, 1]
        domain = wt.prepare_inpainting(v, f, w, matched, False)
        with self.assertRaises(wt.InpaintingError):
            wt.solve_inpainting(domain)
        result = wt.solve_inpainting(domain, skip_rejected=True)
        np.testing.assert_allclose(result[:4], np.tile([.25, .75], (4, 1)), atol=1e-6)
        np.testing.assert_array_equal(domain.rejected, [False] * 4 + [True] * 4)
        for mode in wt.INPAINT_MODES:
            empty = wt.prepare_inpainting(v, f, w, np.zeros(8, bool), mode)
            self.assertTrue(np.all(np.isfinite(wt.solve_inpainting(empty, skip_rejected=True))))

    def test_groups_without_weights_are_not_solved(self):
        v, f, x = grid(9)
        zero = np.zeros(len(v))
        w = np.column_stack([zero, 1 - x, zero, x, zero])
        matched = (x < 0.25) | (x > 0.75)
        with patch.object(wt, '_solve_constrained', wraps=wt._solve_constrained) as solve:
            result = wt.solve_inpainting(wt.prepare_inpainting(v, f, w, matched, 'SURFACE'))
        # The two groups with weights and the all-ones column.
        self.assertEqual(solve.call_args.args[4].shape[1], 3)
        np.testing.assert_array_equal(result[:, [0, 2, 4]], 0)
        alone = wt.solve_inpainting(wt.prepare_inpainting(v, f, w[:, [1, 3]], matched, 'SURFACE'))
        np.testing.assert_allclose(result[:, [1, 3]], alone, atol=1e-7)
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)

    def test_match_confidence_fades_between_limits(self):
        distances = np.array([0, 0, 0, 0, 0, 0.2, 0])
        angles = np.array([0, 30, 45, 60, 90, 10, np.nan])
        np.testing.assert_allclose(wt.match_confidence(distances, angles, 0.1, 30, 90),
                                   [1, 1, 0.84375, 0.5, 0, 0, 0])
        # Without a soft band, confidence is the old binary match.
        np.testing.assert_array_equal(wt.match_confidence(distances, angles, 0.1, 30, 30),
                                      [1, 1, 0, 0, 0, 0, 0])

    def test_closest_surface_angles_fold_flipped_normals(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        f = np.array([[0, 1, 2]])
        n = np.tile([0, 0, 1.0], (3, 1))
        points = np.array([[0.2, 0.2, 0.1], [0.2, 0.2, 0.1], [0.2, 0.2, 0.1], [0.2, 0.2, 0.1]])
        normals = np.array([[0, 0, 1], [0, 0, -1], [0, np.sin(np.pi / 6), np.cos(np.pi / 6)], [0, 0, 0]])
        weights = np.eye(3)
        folded = wt.match_closest_surface(v, f, n, points, normals, weights, True)
        np.testing.assert_allclose(folded.angles[:3], [0, 0, 30], atol=1e-6)
        self.assertTrue(np.isnan(folded.angles[3]))
        np.testing.assert_allclose(folded.distances, 0.1)
        np.testing.assert_allclose(folded.weights, np.tile([0.6, 0.2, 0.2], (4, 1)), atol=1e-6)
        oriented = wt.match_closest_surface(v, f, n, points, normals, weights, False)
        np.testing.assert_allclose(oriented.angles[:3], [0, 180, 30], atol=1e-6)

    def test_bridge_trust_follows_the_gap(self):
        v, f = flat_grid(np.linspace(0, 0.2, 11), np.linspace(0, 0.2, 11))
        index = int(np.argmin(np.linalg.norm(v - [0.1, 0.1, 0], axis=1)))
        trust = []
        for gap in (0, 0.002, 0.004, 0.01, 0.03):
            distances = np.full(len(v), gap)
            trust.append(wt.bridge_confidence(v, f, distances, np.ones(len(v)))[index])
        # Touching and 2 mm away count alike, then trust falls with the gap down to the floor.
        self.assertAlmostEqual(trust[0], trust[1])
        self.assertTrue(np.all(np.diff(trust[1:]) < 0))
        self.assertGreater(trust[0], 0.9)
        self.assertLess(trust[0], 1)
        self.assertAlmostEqual(trust[-1], wt.BRIDGE_FLOOR)
        # The solver pulls with area * t / (1 - t) / L^4 = area / (softness * d)^4, whatever L is.
        for length in (0.005, 0.0125, 0.05):
            t = wt.bridge_confidence(v, f, np.full(len(v), 0.004), np.ones(len(v)), length)[index]
            np.testing.assert_allclose(t / (1 - t) / length ** 4, 1 / (wt.BRIDGE_SOFTNESS * 0.004) ** 4, rtol=1e-6)
        # A wider softness lowers the trust at the same gap; Max Distance cuts it off.
        wide = wt.bridge_confidence(v, f, np.full(len(v), 0.004), np.ones(len(v)), softness=6.0)[index]
        self.assertLess(wide, trust[2])
        cut = wt.bridge_confidence(v, f, np.full(len(v), 0.004), np.ones(len(v)), max_distance=0.003)
        np.testing.assert_array_equal(cut, wt.BRIDGE_FLOOR)
        with self.assertRaises(ValueError):
            wt.bridge_confidence(v, f, np.zeros(len(v)), np.ones(len(v)), 0.0)

    def test_bridge_trust_fades_with_the_normals_either_way(self):
        v, f = flat_grid(np.linspace(0, 0.2, 11), np.linspace(0, 0.2, 11))
        index = int(np.argmin(np.linalg.norm(v - [0.1, 0.1, 0], axis=1)))
        trust = {}
        for angle in (0, 5, 11, 18, 25, 40, 180):
            cosines = np.full(len(v), np.abs(np.cos(np.radians(angle))))
            trust[angle] = wt.bridge_confidence(v, f, np.zeros(len(v)), cosines)[index]
        self.assertAlmostEqual(trust[0], trust[5])
        self.assertAlmostEqual(trust[0], trust[11])
        self.assertAlmostEqual(trust[0], trust[180])
        self.assertTrue(trust[11] > trust[18] > trust[25])
        self.assertAlmostEqual(trust[25], wt.BRIDGE_FLOOR)
        self.assertAlmostEqual(trust[40], wt.BRIDGE_FLOOR)
        # Undefined normals count as disagreeing.
        undefined = wt.bridge_confidence(v, f, np.zeros(len(v)), np.full(len(v), np.nan))
        np.testing.assert_allclose(undefined[index], wt.BRIDGE_FLOOR)

    def test_bridge_trust_is_shared_by_coincident_vertices(self):
        # Two loose parts meeting at a split seam, with different normals on either side.
        left, lf = flat_grid(np.linspace(-0.1, 0, 6), np.linspace(0, 0.1, 6))
        right, rf = flat_grid(np.linspace(0, 0.1, 6), np.linspace(0, 0.1, 6))
        v = np.vstack([left, right])
        f = np.vstack([lf, rf + len(left)])
        cosines = np.r_[np.ones(len(left)), np.full(len(right), np.cos(np.radians(20)))]
        trust = wt.bridge_confidence(v, f, np.full(len(v), 0.003), cosines)
        seam_left = np.flatnonzero(np.isclose(left[:, 0], 0))
        seam_right = len(left) + np.flatnonzero(np.isclose(right[:, 0], 0))
        self.assertFalse(np.allclose(cosines[seam_left], cosines[seam_right]))
        np.testing.assert_allclose(trust[seam_left], trust[seam_right])

    def test_free_borders_keep_more_trust_but_seams_dont_count(self):
        left, lf = flat_grid(np.linspace(-0.1, 0, 11), np.linspace(0, 0.1, 11))
        right, rf = flat_grid(np.linspace(0, 0.1, 11), np.linspace(0, 0.1, 11))
        v = np.vstack([left, right])
        f = np.vstack([lf, rf + len(left)])
        merge = wt.find_vertex_merge_map(v, wt.BRIDGE_WELD)
        border = wt.free_border_distance(v, f, merge)
        # The seam at x == 0 isn't a border: its middle lies about 5 cm from the nearest real one.
        seam = np.isclose(v[:, 0], 0) & np.isclose(v[:, 1], 0.05)
        self.assertTrue(np.all((border[seam] > 0.049) & (border[seam] < 0.052)))
        np.testing.assert_allclose(border[np.isclose(v[:, 0], -0.1)], 0)
        np.testing.assert_array_equal(wt.free_border_distance(v, np.zeros((0, 3), int), merge), np.inf)
        # 1 cm off the body, a vertex on a hem keeps more trust than one 5 cm inside.
        trust = wt.bridge_confidence(v, f, np.full(len(v), 0.01), np.ones(len(v)))
        hem = np.flatnonzero(np.isclose(v[:, 0], -0.1) & np.isclose(v[:, 1], 0.05))[0]
        inside = np.flatnonzero(seam)[0]
        self.assertGreater(trust[hem], 2 * trust[inside])

    def test_groups_are_limited_without_jumps(self):
        rows = np.array([[.3, .2, .15, .1, .08, .07, .05, .03, .015, .005],
                         [.3, .2, .15, .1, .08, .07, .05, .016, .024, .01],
                         [.3, .2, .15, .1, .08, .07, .05, .024, .016, .01]])
        limited = wt.limit_groups_smoothly(rows, 8)
        self.assertTrue(np.all(np.count_nonzero(limited, axis=1) <= 8))
        np.testing.assert_allclose(limited.sum(axis=1), rows.sum(axis=1))
        # Rows 1 and 2 swap their 8th and 9th groups: both fade out near the swap, so the rows stay close.
        self.assertLess(np.abs(limited[1] - limited[2]).max(), 0.015)
        # A tie between the groups around the cut keeps the largest ones instead of emptying the vertex.
        for row, limit in (([0.5, 0.5], 1), ([0.5005, 0.4995], 1), ([0.2] * 5, 4),
                           ([0.3, 0.2995, 0.2991, 0.1014], 2)):
            out = wt.limit_groups_smoothly(np.array([row]), limit)
            self.assertAlmostEqual(out.sum(), sum(row))
            self.assertLessEqual(np.count_nonzero(out), limit)
            self.assertGreater(np.count_nonzero(out), 0)
        # Weights under half an 8-bit step go, and totals are kept.
        tiny = wt.limit_groups_smoothly(np.array([[0.999, 0.001], [0.5, 0.5]]), 8)
        np.testing.assert_allclose(tiny, [[1, 0], [0.5, 0.5]])
        np.testing.assert_allclose(wt.limit_groups_smoothly(np.zeros((2, 3)), 2), 0)

    def test_folded_rims_are_free_borders_but_seams_are_not(self):
        # A panel whose back-face layer folds over at the hem (y == 0): both layers end there.
        front, ff = flat_grid(np.linspace(0, 0.1, 6), np.linspace(0, 0.1, 6))
        back = front.copy()
        back[:, 2] = -0.002 * (back[:, 1] / 0.1)
        v = np.vstack([front, back])
        f = np.vstack([ff, ff[:, ::-1] + len(front)])
        merge = wt.find_vertex_merge_map(v, wt.BRIDGE_WELD)
        rim = np.flatnonzero(np.isclose(v[:, 1], 0))
        self.assertTrue(np.isin(rim, wt.fold_aware_free_borders(v, f, merge)).all())
        # The simple rule sees the two layers' rims as a seam.
        self.assertFalse(np.isin(rim, wt.free_border_vertices(v, f, merge)).any())
        # Two panels continuing each other across a split seam (x == 0.1) don't end there.
        right, rf = flat_grid(np.linspace(0.1, 0.2, 6), np.linspace(0, 0.1, 6))
        v = np.vstack([front, right])
        f = np.vstack([ff, rf + len(front)])
        merge = wt.find_vertex_merge_map(v, wt.BRIDGE_WELD)
        seam = np.flatnonzero(np.isclose(v[:, 0], 0.1) & (v[:, 1] > 0.01) & (v[:, 1] < 0.09))
        self.assertFalse(np.isin(seam, wt.fold_aware_free_borders(v, f, merge)).any())
        self.assertFalse(np.isin(seam, wt.free_border_vertices(v, f, merge)).any())

    def test_forward_kinematics_turns_children_but_scales_only_the_bone(self):
        rest = np.tile(np.eye(4), (2, 1, 1))
        rest[1, :3, 3] = (0, 0, -0.4)
        quarter = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        deform = wt.forward_kinematics(rest, [-1, 0], {0: quarter}, {0: (2, 2, 2)})
        knee = deform[1] @ [0, 0, -0.4, 1]
        np.testing.assert_allclose(knee[:3], quarter @ [0, 0, -0.4], atol=1e-12)
        np.testing.assert_allclose(deform[0][:3, :3], quarter * 2, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(deform[1][:3, :3]), 1, atol=1e-12)

    def test_strides_need_both_thighs(self):
        names = ['j_kosi', 'j_asi_a_l', 'j_asi_a_r', 'j_asi_b_l']
        rest = np.tile(np.eye(4), (4, 1, 1))
        rest[1, :3, 3], rest[2, :3, 3], rest[3, :3, 3] = (0.1, 0, 0.9), (-0.1, 0, 0.9), (0.1, 0, 0.5)
        poses = wt.stride_poses(names, rest, [-1, 0, 0, 1], (0, -1, 0))
        self.assertEqual(len(poses), 4)
        # The first stride swings the left leg forward (towards -Y): its knee moves forward, the right one back.
        knee = poses[0][3] @ [0.1, 0, 0.5, 1]
        self.assertLess(knee[1], -0.1)
        self.assertEqual(wt.stride_poses(names[:2], rest[:2], [-1, 0], (0, -1, 0)), [])
        # Turned as a whole (a posed body), the strides turn with it.
        turn = np.eye(4)
        turn[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        turned = wt.stride_poses(names, turn @ rest, [-1, 0, 0, 1], turn[:3, :3] @ [0, -1, 0])
        for pose, turned_pose in zip(poses, turned):
            np.testing.assert_allclose(turned_pose, turn @ pose @ np.linalg.inv(turn), atol=1e-12)
        # The thighs get thicker across the leg (towards the knee), not longer.
        scaled = poses[2][1] @ [0.1, 0, 0.5, 1]
        np.testing.assert_allclose(np.linalg.norm(scaled[:3] - [0.1, 0, 0.9]), 0.4, atol=1e-9)

    def test_reweighted_domain_solves_like_a_new_one(self):
        v, f, w, matched = split_surface()
        confidence = np.where(matched, 1.0, 0.3)
        domain = wt.prepare_inpainting(v, f, w, matched, 'COMBINED', 0.0, confidence, 0.5)
        other = np.tile([0.4, 0.6], (len(v), 1))
        fresh = wt.prepare_inpainting(v, f, other, matched, 'COMBINED', 0.0, confidence * 0.5, 0.5)
        again = wt.reweight_domain(domain, v, f, other, matched, confidence * 0.5)
        np.testing.assert_allclose(wt.solve_inpainting(again), wt.solve_inpainting(fresh), atol=1e-6)

    def test_hem_follow_pulls_a_hem_that_would_let_the_leg_through(self):
        # A thigh's front as a flat patch (bone 0 of two) and a hem 5 mm in front of it whose solved
        # weights stay with the hip (bone 1): swinging the thigh forward would push it through.
        body, bf = flat_grid(np.linspace(-0.05, 0.05, 11), np.linspace(-0.1, 0.1, 21))
        hem, hf = flat_grid(np.linspace(-0.04, 0.04, 9), np.linspace(-0.06, 0.02, 9), z=0.005)
        tree = wt.build_surface_bvh(body, bf)
        points = hem.copy()
        points[:, 2] = 0
        point_normals = np.tile([0.0, 0.0, 1.0], (len(hem), 1))
        normals = point_normals.copy()
        closest = np.tile([1.0, 0.0], (len(hem), 1))
        solved = np.tile([0.0, 1.0], (len(hem), 1))
        swing = np.tile(np.eye(4), (2, 1, 1))
        swing[0, 2, 3] = 0.02  # the thigh moves 2 cm out, the hip stays
        merge = wt.find_vertex_merge_map(hem, wt.BRIDGE_WELD)
        distances = np.full(len(hem), 0.005)
        pull = wt.hem_follow(hem, hf, merge, normals, points, point_normals, distances, closest, solved,
                             np.ones(len(hem)), [swing], tree)
        near = np.isclose(hem[:, 1], -0.06) & (np.abs(hem[:, 0]) < 0.02)
        # 4 cm from every border: halfway into the hem band.
        middle = np.isclose(hem[:, 0], 0) & np.isclose(hem[:, 1], -0.02)
        self.assertTrue(np.all(pull[near] > 0.5))
        self.assertTrue(np.all(pull <= 1))
        # Hems already following the leg, off the legs, or without strides stay as they are.
        np.testing.assert_array_equal(wt.hem_follow(hem, hf, merge, normals, points, point_normals, distances,
                                                    closest, closest, np.ones(len(hem)), [swing], tree), 0)
        np.testing.assert_array_equal(wt.hem_follow(hem, hf, merge, normals, points, point_normals, distances,
                                                    closest, solved, np.zeros(len(hem)), [swing], tree), 0)
        np.testing.assert_array_equal(wt.hem_follow(hem, hf, merge, normals, points, point_normals, distances,
                                                    closest, solved, np.ones(len(hem)), [], tree), 0)
        trust = wt.hem_follow_trust(np.full(len(hem), 0.01), pull)
        self.assertTrue(np.all(trust[near] > 0.9))
        self.assertTrue(np.all(trust <= wt.HEM_TRUST))
        self.assertLess(pull[middle].max(), pull[near].min())
        self.assertGreater(pull[middle].min(), 0.2)

    def test_soft_matches_anchor_otherwise_rejected_parts(self):
        v, f, w, matched = split_surface()
        w[4:] = [0.9, 0.1]
        confidence = np.array([1] * 4 + [0.3, 0.2, 0.3, 0.2], dtype=float)
        self.assertTrue(wt.prepare_inpainting(v, f, w, matched, False).rejected[4:].all())
        domain = wt.prepare_inpainting(v, f, w, matched, False, confidence=confidence, soft_length=0.1)
        self.assertFalse(domain.rejected.any())
        # The right part only has weak matches, all agreeing: they define it.
        np.testing.assert_allclose(wt.solve_inpainting(domain), w, atol=1e-6)
        # Weak matches need a positive smoothing length to take effect.
        self.assertTrue(wt.prepare_inpainting(v, f, w, matched, False,
                                              confidence=confidence).rejected[4:].all())

    def test_soft_matches_yield_to_direct_matches(self):
        v, f, x = grid(15, length=3)
        matched = x < 0.5
        w = np.where(matched[:, np.newaxis], [1.0, 0.0], [0.0, 1.0])
        results = []
        for trust in [0.1, 0.5, 0.9]:
            confidence = np.where(matched, 1.0, trust)
            domain = wt.prepare_inpainting(v, f, w, matched, False, confidence=confidence,
                                           soft_length=0.25)
            result = wt.solve_inpainting(domain).astype(np.float64)
            np.testing.assert_array_equal(result[matched], w[matched])
            np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)
            self.assertGreaterEqual(result.min(), 0)
            results.append(result[~matched, 1].mean())
        # Weak matches pull harder the more they are trusted, and never win
        # completely next to direct matches.
        self.assertLess(results[0], results[1])
        self.assertLess(results[1], results[2])
        self.assertLess(results[2], 1)

    def test_point_mode_welds_coincident_layers(self):
        # Double-sided cloth duplicates every vertex at the same position. The
        # point-cloud Laplacian degenerates on such points and the solve used to
        # return weights like -5 or 12 without any warning.
        for size in [8, 12, 20]:
            with self.subTest(size=size):
                v, f, x = grid(size)
                v2 = np.vstack([v, v])
                f2 = np.vstack([f, f[:, ::-1] + len(v)])
                x2 = np.concatenate([x, x])
                matched = (x2 < 0.2) | (x2 > 0.8)
                w = np.column_stack([x2 <= 0.5, x2 > 0.5]).astype(float)
                result = wt.solve_inpainting(wt.prepare_inpainting(v2, f2, w, matched, True))
                np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)
                self.assertGreaterEqual(result.min(), 0)
                self.assertLessEqual(result.max(), 1 + 1e-6)
                np.testing.assert_array_equal(result[:len(v)], result[len(v):])

    def test_unstable_solve_falls_back_to_harmonic(self):
        v, f, x = grid(10)
        matched = (x < 0.2) | (x > 0.8)
        w = np.column_stack([x <= 0.5, x > 0.5]).astype(float)
        original = wt._solve_constrained

        def broken_biharmonic(*args):
            solution = original(*args)
            return solution if args[-1] else solution * 3

        with patch.object(wt, '_solve_constrained', side_effect=broken_biharmonic):
            domain = wt.prepare_inpainting(v, f, w, matched, False)
            result = wt.solve_inpainting(domain)
        self.assertTrue(domain.harmonic_fallback)
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)
        with patch.object(wt, '_solve_constrained', side_effect=lambda *args: original(*args) * 3):
            with self.assertRaises(wt.InpaintingError):
                wt.solve_inpainting(wt.prepare_inpainting(v, f, w, matched, False))

    def test_negative_overshoot_is_clipped_keeping_row_totals(self):
        weights = np.array([[1.2, -0.2, 0.0], [0.7, 0.5, -0.2], [0.5, 0.5, 0.0]])
        clipped = wt._clip_negative_weights(weights)
        self.assertGreaterEqual(clipped.min(), 0)
        np.testing.assert_allclose(clipped.sum(axis=1), 1)
        np.testing.assert_allclose(clipped[1], [0.7 / 1.2, 0.5 / 1.2, 0])
        np.testing.assert_array_equal(clipped[2], weights[2])

    def test_disconnected_rejected_at_every_scale(self):
        v, f, w, matched = split_surface()
        for scale in [0.01, 0.1, 1, 3, 10, 100]:
            with self.subTest(scale=scale):
                domain = wt.prepare_inpainting(v * scale, f, w, matched, False)
                np.testing.assert_array_equal(domain.rejected, ~matched)
                self.assertFalse(wt.inpaint(v * scale, f, w, matched, False)[0])

    def test_virtual_merge_rescues_every_mode(self):
        v, f, w, matched = split_surface()
        for mode in wt.INPAINT_MODES:
            domain = wt.prepare_inpainting(v, f, w, matched, mode, 0.0001)
            self.assertFalse(domain.rejected.any())
            np.testing.assert_allclose(wt.solve_inpainting(domain), w, atol=1e-6)

    def test_all_and_no_matches(self):
        v, f, w, matched = split_surface()
        w[4:] = [0.9, 0.1]
        ok, out = wt.inpaint(v, f, w, np.ones(8, bool), False, 0.0001)
        self.assertTrue(ok)
        np.testing.assert_allclose(out, w)
        domain = wt.prepare_inpainting(v, f, w, np.zeros(8, bool), True, 0.0001)
        self.assertTrue(domain.rejected.all())

    def test_small_point_graph_and_diagnostic(self):
        v, f, w, matched = split_surface()
        v, f, w = v[:4], f[:2], w[:4]
        matched = np.array([True, False, False, False])
        domain = wt.prepare_inpainting(v, f, w, matched, True)
        self.assertFalse(domain.rejected.any())
        np.testing.assert_allclose(wt.solve_inpainting(domain), w, atol=1e-6)

    def test_point_graph_can_connect_surface_components(self):
        v, f, w, matched = split_surface()
        v[4:, 0] += 0.2
        surface = wt.prepare_inpainting(v, f, w, matched, False)
        self.assertTrue(surface.rejected[4:].all())
        for mode in [True, 'COMBINED']:
            domain = wt.prepare_inpainting(v, f, w, matched, mode)
            self.assertFalse(domain.rejected.any())
            np.testing.assert_allclose(wt.solve_inpainting(domain), w, atol=1e-6)

    def test_combined_mode_keeps_weights_from_crossing_gaps(self):
        # A U-shaped strip whose arms lie one cell apart, matched at their open ends to a group each.
        v, f = flat_grid(np.linspace(0, 1.1, 56), np.linspace(0, 0.22, 12))
        centers = v[f].mean(axis=1)
        f = f[~((centers[:, 0] < 1) & (centers[:, 1] > 0.1) & (centers[:, 1] < 0.12))]
        lower = v[:, 1] < 0.11
        matched = v[:, 0] < 0.2
        w = np.column_stack([lower, ~lower]).astype(float)
        solved = {mode: wt.solve_inpainting(wt.prepare_inpainting(v, f, w, matched, mode))
                  for mode in wt.INPAINT_MODES}
        # Within a loose part, Combined is Surface: weights go around the slot. Point lets them
        # leak across it.
        np.testing.assert_allclose(solved['COMBINED'], solved['SURFACE'], atol=1e-6)
        self.assertGreater(np.abs(solved['POINT'] - solved['COMBINED']).max(), 0.1)
        # Separate arms two cells apart don't touch, so they aren't linked.
        v, f = flat_grid(np.linspace(0, 1, 51), np.linspace(0, 0.1, 6))
        v, f = np.vstack([v, v + [0, 0.14, 0]]), np.vstack([f, f + len(v)])
        lower = np.arange(len(v)) < len(v) // 2
        w = np.column_stack([lower, ~lower]).astype(float)
        combined = wt.solve_inpainting(wt.prepare_inpainting(v, f, w, v[:, 0] < 0.2, 'COMBINED'))
        np.testing.assert_allclose(combined, w, atol=1e-6)

    def test_combined_mode_links_touching_and_unmatched_parts(self):
        panel = flat_grid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
        parts = [
            panel,
            # A lining 2 mm under the panel, only matched along one end.
            (panel[0] - [0, 0, 0.002], panel[1]),
            # An unmatched button 1 cm above the panel.
            flat_grid(np.linspace(0.45, 0.55, 4), np.linspace(0.45, 0.55, 4), z=0.01),
            # An unmatched part far away from everything.
            flat_grid(np.linspace(0, 1, 6), np.linspace(5, 6, 6)),
        ]
        starts = np.cumsum([0] + [len(part[0]) for part in parts])
        v = np.vstack([part[0] for part in parts])
        f = np.vstack([part[1] + start for part, start in zip(parts, starts)])
        x = v[:, 0]
        matched = np.zeros(len(v), bool)
        matched[:starts[1]] = (x[:starts[1]] < 0.25) | (x[:starts[1]] > 0.75)
        matched[starts[1]:starts[2]] = x[starts[1]:starts[2]] < 0.25
        w = np.column_stack([x < 0.5, x >= 0.5]).astype(float)
        domain = wt.prepare_inpainting(v, f, w, matched, 'COMBINED')
        np.testing.assert_array_equal(domain.rejected, np.arange(len(v)) >= starts[3])
        result = wt.solve_inpainting(domain, skip_rejected=True)
        panel_w, lining_w, button_w = (result[start:end] for start, end in zip(starts, starts[1:4]))
        # The lining moves with the panel, and the button takes the weights under it.
        self.assertLess(np.abs(lining_w - panel_w).max(), 0.02)
        under = np.argmin(np.linalg.norm(v[:starts[1], None, :2] - v[None, starts[2]:starts[3], :2], axis=2),
                          axis=0)
        self.assertLess(np.abs(button_w - panel_w[under]).max(), 0.05)
        # Split seams are welded, like in Point mode.
        v, f, w, matched = split_surface()
        self.assertFalse(wt.prepare_inpainting(v, f, w, matched, 'COMBINED').rejected.any())
        with self.assertRaises(wt.InpaintingError):
            wt.prepare_inpainting(v, f, w, matched, 'FAST')

    def test_final_sync_repairs_smoothing_regression(self):
        v, f, w, matched = split_surface()
        w[:4] = [[1, 0], [0, 1], [0.8, 0.2], [0.2, 0.8]]
        ok, solved = wt.inpaint(v, f, w, matched, False, 0.0001)
        self.assertTrue(ok)
        edges = np.unique(np.sort(np.vstack([f[:, :2], f[:, 1:], f[:, [2, 0]]]), axis=1), axis=0)
        adjacency = wt.sp.sparse.csr_array((np.ones(len(edges) * 2),
                                           (edges.ravel(), edges[:, ::-1].ravel())), shape=(8, 8))
        adjacency.setdiag(1)
        lists = [adjacency[[i]].indices.tolist() for i in range(8)]
        smoothed = np.asarray(wt.smooth_weigths(v, solved, matched, adjacency, lists, 4, 0.2, 5))
        self.assertGreater(np.max(np.abs(smoothed[[1, 3]] - smoothed[[4, 6]])), 0.05)
        final, synced, skipped = seams.synchronize_weights(
            smoothed, np.ones_like(smoothed, bool), np.zeros(8, bool),
            [np.array([1, 4]), np.array([3, 6])], 1)
        np.testing.assert_array_equal(final[[1, 3]], final[[4, 6]])
        np.testing.assert_array_equal(final[[0, 2, 5, 7]], smoothed[[0, 2, 5, 7]])
        self.assertEqual((synced, skipped), (2, 0))

    def test_duplicate_and_collapsed_faces(self):
        v, f, w, matched = split_surface()
        v[4:] = v[:4]
        ok, out = wt.inpaint(v, np.vstack([f, f[:, ::-1]]), w, matched, False, 0.0001)
        self.assertTrue(ok)
        np.testing.assert_allclose(out, w, atol=1e-6)
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                      [2, 0, 0], [2.00001, 0, 0], [2, 0.00001, 0]])
        f = np.array([[0, 1, 2], [3, 4, 5]])
        w = np.tile([0.25, 0.75], (6, 1))
        matched = np.array([True, False, False, True, True, True])
        self.assertTrue(wt.inpaint(v, f, w, matched, False, 0.0001)[0])
        matched[3:] = False
        self.assertFalse(wt.inpaint(v, f, w, matched, False, 0.0001)[0])

    def test_invalid_data(self):
        v, f, w, matched = split_surface()
        for value in [np.nan, np.inf]:
            broken = v.copy(); broken[0, 0] = value
            self.assertFalse(wt.inpaint(broken, f, w, matched, False)[0])
            broken = w.copy(); broken[0, 0] = value
            self.assertFalse(wt.inpaint(v, f, broken, matched, False)[0])
        self.assertFalse(wt.inpaint(v, f + 100, w, matched, False)[0])

    def test_seam_detection_boundaries_scope_and_zero(self):
        v, f, w, matched = split_surface()
        edges = np.unique(np.sort(np.vstack([f[:, :2], f[:, 1:], f[:, [2, 0]]]), axis=1), axis=0)
        boundary, labels = seams.boundary_components(len(v), f, edges)
        clusters, excluded = seams.find_seam_clusters(v, boundary, labels, np.zeros(8), 0)
        self.assertEqual([c.tolist() for c in clusters], [[1, 4], [3, 6]])
        self.assertEqual(excluded, 0)
        clusters, excluded = seams.find_seam_clusters(v, boundary, labels, labels, 0)
        self.assertEqual(clusters, [])
        self.assertEqual(excluded, 2)
        # Closed tetrahedra have no open borders, even with duplicate geometry.
        tetra = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
        boundary, _ = seams.boundary_components(4, tetra, np.array([[0, 1], [1, 2], [2, 3]]))
        self.assertFalse(boundary.any())

    def test_seam_clusters_avoid_density_chaining(self):
        # A seam sampled far more densely on one side puts a single sparse-side
        # vertex within range of several dense-side vertices. Without a
        # reciprocal-nearest check, connected components chains the whole run
        # into one oversized cluster (the entire seam, in this example)
        # instead of many local pairs, smearing weights across positions that
        # were never actually close together.
        front_y = np.linspace(0, 1, 100)
        back_y = np.linspace(0, 1, 10)
        vertices = np.vstack([
            np.column_stack([np.zeros(100), front_y, np.zeros(100)]),
            np.column_stack([np.full(10, 0.001), back_y, np.zeros(10)]),
        ])
        boundary = np.ones(110, dtype=bool)
        components = np.array([0] * 100 + [1] * 10)
        clusters, excluded = seams.find_seam_clusters(
            vertices, boundary, components, np.zeros(110), 0.07)
        self.assertEqual(excluded, 0)
        self.assertEqual(len(clusters), 10)
        self.assertTrue(all(len(cluster) == 2 for cluster in clusters))

    def test_seam_limit_protections_and_stable_ties(self):
        w = np.eye(6)[:2]
        w[0] = [0.25, 0.25, 0.25, 0.25, 0, 0]
        w[1] = [0, 0, 0.25, 0.25, 0.25, 0.25]
        editable = np.ones_like(w, bool)
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)], 4)
        np.testing.assert_array_equal(out[0], out[1])
        self.assertLessEqual(np.count_nonzero(out[0]), 4)
        self.assertAlmostEqual(out[0].sum(), 1)
        self.assertEqual((synced, skipped), (1, 0))
        editable[:, 0] = False
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)], 4)
        np.testing.assert_array_equal(out, w)
        self.assertEqual(skipped, 1)
        out, _, skipped = seams.synchronize_weights(w, np.ones_like(w, bool), np.array([True, False]), [np.arange(2)])
        np.testing.assert_array_equal(out, w)
        self.assertEqual(skipped, 1)

    def test_confidence_prevents_matched_weight_dilution(self):
        # A double-sided mesh duplicates every vertex as a front- and a
        # back-facing copy. The front copy commonly gets a clean direct
        # surface match; its back-facing twin, unmatched, only reaches a
        # (weaker) weight through inpainting. Averaging the two blindly drags
        # the well-matched side down at every seam vertex around the
        # silhouette. Confidence-weighting the merge toward the matched side
        # avoids that dilution.
        w = np.array([[1.0, 0.0], [0.3, 0.7]])
        editable = np.ones_like(w, bool)
        out, synced, skipped = seams.synchronize_weights(
            w, editable, np.zeros(2, bool), [np.arange(2)], confidence=np.array([1.0, 0.0]))
        np.testing.assert_allclose(out, [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual((synced, skipped), (1, 0))
        # With no confidence signal, the merge falls back to a plain mean.
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)])
        np.testing.assert_allclose(out, [[0.65, 0.35], [0.65, 0.35]])
        # All-zero confidence within a cluster also falls back to a plain mean.
        out, synced, skipped = seams.synchronize_weights(
            w, editable, np.zeros(2, bool), [np.arange(2)], confidence=np.zeros(2))
        np.testing.assert_allclose(out, [[0.65, 0.35], [0.65, 0.35]])

    def test_fixed_contributions_count_toward_limit(self):
        w = np.array([[0.2, 0.6, 0.2], [0.2, 0.1, 0.7]])
        editable = np.array([[False, True, True]] * 2)
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)], 2)
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_array_equal(out[:, 0], w[:, 0])
        self.assertEqual(np.count_nonzero(out[0]), 2)
        self.assertEqual((synced, skipped), (1, 0))

    def test_infeasible_fixed_weights_and_limits_are_skipped(self):
        for fixed, limit in [(1.1, 4), (0.2, 1)]:
            w = np.array([[fixed, 0.5, 0.3], [fixed, 0.1, 0.7]])
            out, synced, skipped = seams.synchronize_weights(
                w, np.array([[False, True, True]] * 2), np.zeros(2, bool), [np.arange(2)], limit)
            np.testing.assert_array_equal(out, w)
            self.assertEqual((synced, skipped), (0, 1))


class AddonTests(unittest.TestCase):
    def test_register_leaves_sys_path_and_scipy_alone(self):
        self.assertEqual(PATH_AFTER_REGISTER, PATH_BEFORE_REGISTER)
        self.assertFalse(SCIPY_AFTER_REGISTER)

    def test_missing_dependencies_disable_transfer(self):
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        with patch.object(dependencies, '_missing', ['SciPy']):
            self.assertIn('SciPy', operators.check_transfer_ready(bpy.context))
            self.assertFalse(bpy.ops.magic_fit.transfer_weights.poll())
        self.assertEqual(dependencies.missing(), [])

    def test_names_do_not_clash_with_standalone_addon(self):
        # The standalone Robust Weight Transfer may be enabled at the same time.
        def registered(idname):
            module, name = idname.split('.')
            try:
                getattr(getattr(bpy.ops, module), name).get_rna_type()
            except KeyError:
                return False
            return True

        self.assertFalse(hasattr(bpy.types.Scene, 'robust_weight_transfer_settings'))
        self.assertFalse(hasattr(bpy.types.Object, 'robust_weight_transfer_settings'))
        self.assertTrue(registered('magic_fit.transfer_weights'))
        for name in ('skin_weight_transfer', 'select_non_matched', 'rwt_inpaint', 'smooth_limit_weights',
                     'rbt_reset_scene_settings'):
            self.assertFalse(registered('object.' + name), name)

    def test_register_cycle(self):
        for _ in range(2):
            magic_fit.unregister()
            magic_fit.register()
        self.assertTrue(hasattr(bpy.types.Scene, 'magic_fit_transfer'))
        self.assertTrue(hasattr(bpy.types.Object, 'magic_fit_transfer'))
        self.assertTrue(hasattr(bpy.ops.magic_fit, 'transfer_weights'))
        self.assertTrue(hasattr(bpy.types, 'VIEW3D_PT_weight_transfer_sidebar'))


class TransferScene:
    """Scenes for the operator tests: a rig, meshes on it, a body."""

    @property
    def body(self):
        return bpy.context.scene.magic_fit.target

    @body.setter
    def body(self, obj):
        bpy.context.scene.magic_fit.target = obj

    def setUp(self):
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        self.settings = bpy.context.scene.magic_fit_transfer
        bpy.ops.magic_fit.reset_transfer_settings()
        self.settings.use_deformed_source = False
        self.settings.use_deformed_target = False
        self.settings.inpaint_mode = 'SURFACE'
        self.settings.max_distance = 0.1
        # The matching and inpainting underneath Crease Smoothing: direct matches by Max Distance and
        # Normal Angle, soft ones up to Soft Limit. CreaseSmoothingTests turns it on.
        self.settings.crease_smoothing = False
        # Tests turn these on where they check them.
        self.settings.virtual_merge = False
        self.settings.seam_sync = False
        self.rig = self.make_rig('Rig')
        self.v, self.f, _, _ = split_surface()

    def make_rig(self, name):
        data = bpy.data.armatures.new(name)
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        for index, name in enumerate([
                'A', 'B', 'C', 'D', 'E', 'F', 'Arm.L', 'Arm.R', 'Spine']):
            bone = data.edit_bones.new(name)
            bone.head = (index * 0.1, 0, -1)
            bone.tail = (index * 0.1, 0, 1)
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
        return obj

    def mesh(self, name, v, f, rig=True):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(np.asarray(v).tolist(), [], np.asarray(f).tolist())
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        if rig:
            mod = obj.modifiers.new('Armature', 'ARMATURE')
            mod.object = self.rig
        return obj

    def weights(self, obj, names, values):
        for name, column in zip(names, np.asarray(values).T):
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            for index, value in enumerate(column):
                if value > 0:
                    group.add([index], float(value), 'REPLACE')

    def select(self, active, *others):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in (active,) + others:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active

    def source(self):
        obj = self.mesh('Source', self.v, self.f)
        # Split source has an intentional weight jump at its duplicate border.
        values = np.array([[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4)
        self.weights(obj, ['A', 'B'], values)
        self.body = obj
        return obj

    def read(self, obj, names=('A', 'B')):
        return np.column_stack([util.get_group_arr(obj, name) if obj.vertex_groups.get(name)
                                else np.zeros(len(obj.data.vertices)) for name in names])

    def snapshot(self, obj):
        return ([(tuple(v.co), tuple(v.normal)) for v in obj.data.vertices],
                [tuple(e.vertices) for e in obj.data.edges],
                [tuple(p.vertices) for p in obj.data.polygons],
                [[tuple(d.uv) for d in uv.data] for uv in obj.data.uv_layers],
                [[tuple(p.co) for p in key.data] for key in obj.data.shape_keys.key_blocks]
                if obj.data.shape_keys else [],
                [(m.name, m.type) for m in obj.modifiers])

    def partial_scene(self, heights=(0, .75, 1, 2)):
        source = self.mesh('Source', [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
        self.weights(source, ['A', 'B'], np.tile([.9, .1], (3, 1)))
        vertices = [[x, y, z] for z in heights for x, y in [(0, 0), (1, 0), (0, 1)]]
        faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]
        target = self.mesh('Target', vertices, faces)
        self.weights(target, ['A', 'B'], np.tile([.2, .8], (len(vertices), 1)))
        self.body = source
        self.settings.partial_reweight = True
        self.settings.partial_reweight_falloff = 50
        self.settings.max_distance = 1
        self.select(target)
        return source, target

    def memberships(self, obj):
        return [tuple((obj.vertex_groups[g.group].name, g.weight) for g in v.groups)
                for v in obj.data.vertices]



class BlenderTests(TransferScene, unittest.TestCase):
    """Weight Transfer with Crease Smoothing off."""

    def test_partial_operator_falloff_preserves_outside_memberships(self):
        _, target = self.partial_scene()
        target.vertex_groups['A'].add([9], 1e-8, 'REPLACE')
        target.vertex_groups['B'].remove([10])
        target.vertex_groups['A'].remove([10])
        before = self.memberships(target)
        geometry = self.snapshot(target)
        self.settings.smoothing_enable = True
        self.settings.seam_sync = True
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target)[:6],
                                   [[.9, .1]] * 3 + [[.55, .45]] * 3, atol=1e-6)
        self.assertEqual(self.memberships(target)[6:], before[6:])
        self.assertEqual(self.snapshot(target), geometry)

    def test_partial_masks_locks_and_missing_groups(self):
        source, target = self.partial_scene()
        target.vertex_groups.remove(target.vertex_groups['B'])
        self.weights(source, ['C'], np.ones((3, 1)))
        self.weights(target, ['C', 'TargetOnly', 'Mask'], np.tile([.3, .4, .25], (12, 1)))
        target.vertex_groups['C'].lock_weight = True
        target.magic_fit_transfer.vertex_group = 'Mask'
        target.magic_fit_transfer.vertex_group_invert = True
        before = self.memberships(target)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target)[:6],
                                   [[.725, .075]] * 3 + [[.4625, .0375]] * 3, atol=1e-6)
        np.testing.assert_allclose(self.read(target, ['C', 'TargetOnly', 'Mask']),
                                   np.tile([.3, .4, .25], (12, 1)))
        self.assertEqual(self.memberships(target)[6:], before[6:])

    def test_partial_zero_radius_and_hard_cutoff(self):
        source, target = self.partial_scene()
        self.settings.partial_reweight_falloff = 0
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target)[:9], np.tile([.9, .1], (9, 1)), atol=1e-6)
        self.weights(target, ['A', 'B'], np.tile([.2, .8], (12, 1)))
        # Use exact BVH hits for the zero-radius case; triangle-interior
        # projections can have a nonzero floating-point residual.
        for vertex, original in zip(target.data.vertices[:3], source.data.vertices):
            vertex.co = original.co
        target.data.update()
        self.settings.max_distance = 0
        self.settings.partial_reweight_falloff = 100
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), [[.9, .1]] * 3 + [[.2, .8]] * 9, atol=1e-6)

    def test_partial_no_matches_and_selected_noop(self):
        _, target = self.partial_scene((2,))
        target.vertex_groups.remove(target.vertex_groups['B'])
        far = self.mesh('Far', self.v + [0, 0, 3], self.f)
        before = self.memberships(target)
        for mode in wt.INPAINT_MODES:
            self.settings.inpaint_mode = mode
            self.settings.apply_to_selected = True
            self.select(target, far)
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
            self.assertEqual(self.memberships(target), before)
            self.assertNotIn('B', target.vertex_groups)
            self.assertEqual(len(far.vertex_groups), 0)

    def test_partial_unsupported_normals_and_diagnostics(self):
        _, target = self.partial_scene((0, .5, 2))
        # Reject the middle part's normals, while the last part is outside.
        for polygon in target.data.polygons:
            if polygon.index == 1:
                polygon.flip()
        target.data.update()
        self.settings.flip_vertex_normal = False
        before = self.memberships(target)
        with patch.object(transfer, 'report_partial', wraps=transfer.report_partial) as report:
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        staged = report.call_args.args[1][0]
        self.assertEqual(np.count_nonzero((staged['strength'] > 0) & ~staged['supported']), 3)
        self.assertEqual(self.memberships(target)[3:], before[3:])
        # The first part trusts the body, the flipped one doesn't, and the last is out of range.
        self.assertTrue(np.all(staged['confidence'][:3] > .99))
        self.assertTrue(np.all(staged['confidence'][3:6] < .01))
        np.testing.assert_array_equal(staged['strength'][6:], 0)
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.magic_fit.select_rejected(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertEqual([v.select for v in target.data.vertices], [False] * 3 + [True] * 3 + [False] * 3)

    def test_partial_no_usable_normals_in_every_mode(self):
        _, target = self.partial_scene((.5,))
        target.data.polygons[0].flip()
        target.data.update()
        self.settings.flip_vertex_normal = False
        before = self.memberships(target)
        for mode in wt.INPAINT_MODES:
            self.settings.inpaint_mode = mode
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
            self.assertEqual(self.memberships(target), before)

    def test_partial_surface_distance_world_and_evaluated_shapes(self):
        source, target = self.partial_scene((.5,))
        # Local .5 becomes world distance 1, halfway through a radius-2 falloff.
        for obj in [source, target]:
            obj.location = (4, -3, 2)
            obj.scale = (2, 3, 2)
        self.settings.max_distance = 2
        self.settings.partial_reweight_falloff = 100
        bpy.context.view_layer.update()
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), np.tile([.55, .45], (3, 1)), atol=1e-6)
        source.shape_key_add(name='Basis')
        key = source.shape_key_add(name='Raised')
        for vertex in key.data:
            vertex.co.z = .5
        key.value = 1
        self.settings.use_deformed_source = True
        bpy.context.view_layer.update()
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), np.tile([.9, .1], (3, 1)), atol=1e-6)
        target.shape_key_add(name='Basis')
        key = target.shape_key_add(name='Far')
        for vertex in key.data:
            vertex.co.z += 2
        key.value = 1
        self.settings.use_deformed_target = True
        bpy.context.view_layer.update()
        before = self.memberships(target)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        self.assertEqual(self.memberships(target), before)

    def test_partial_virtual_cross_object_recovery(self):
        source = self.mesh('Source', self.v[:4], self.f[:2])
        self.weights(source, ['A', 'B'], np.tile([.9, .1], (4, 1)))
        self.body = source
        left = self.mesh('Left', self.v[:4], self.f[:2])
        right = self.mesh('Right', self.v[4:], (self.f[2:] - 4)[:, ::-1])
        self.weights(right, ['B', 'A'], np.tile([.8, .2], (4, 1)))
        self.settings.partial_reweight = True
        self.settings.partial_reweight_falloff = 0
        self.settings.max_distance = 2
        self.settings.flip_vertex_normal = False
        self.settings.apply_to_selected = True
        self.settings.seam_sync = True
        self.settings.seam_sync_across_objects = True
        self.settings.virtual_merge = True
        # Batch mode retains the existing convention of ignoring object masks.
        self.weights(right, ['Mask'], np.zeros((4, 1)))
        right.magic_fit_transfer.vertex_group = 'Mask'
        for mode in wt.INPAINT_MODES:
            self.settings.inpaint_mode = mode
            self.select(left, right)
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
            np.testing.assert_allclose(self.read(right), np.tile([.9, .1], (4, 1)), atol=1e-6)

    def test_partial_seams_skip_falloff_and_preserved_members(self):
        self.settings.seam_sync = True
        for strength in [0, .5]:
            targets = self.staged_pair()
            for target in targets:
                target['strength'] = np.full(4, strength)
                target['supported'] = np.ones(4, bool)
                transfer.stage_weights(target, ['A', 'B'], [True, True])
            before = [dict((k, v.copy()) for k, v in t['final'].items()) for t in targets]
            self.settings.apply_to_selected = True
            self.settings.seam_sync_across_objects = True
            self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B']), (0, 2, 0))
            for target, previous in zip(targets, before):
                for name, values in previous.items():
                    np.testing.assert_array_equal(target['final'][name], values)

    def test_partial_processing_excludes_unsupported_neighbors(self):
        # Connected triangles, but support stops at vertex 2. Unsupported weights
        # must not enter smoothing averages or influence-limit dilation.
        obj = self.mesh('Target', [[0, 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, 0]],
                        [[0, 1, 2], [1, 3, 2]])
        self.settings.smoothing_enable = True
        self.settings.max_distance = 1
        self.settings.num_limit_groups = 2
        outputs = []
        for unsupported in [[1, 0, 0, 0, 0], [.2, .2, .2, .2, .2]]:
            target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
            target.update(weights=np.array([[.7, .3, 0, 0, 0]] * 3 + [unsupported]),
                          matched=np.array([True, False, False, False]),
                          supported=np.array([True, True, True, False]))
            transfer.process_weights(target, self.settings)
            outputs.append(target['weights'][:3].copy())
        np.testing.assert_allclose(outputs[0], np.tile([.7, .3, 0, 0, 0], (3, 1)))
        np.testing.assert_array_equal(*outputs)

    def test_partial_numerical_failure_cancels_without_writes(self):
        _, target = self.partial_scene((0, .75))
        before = self.memberships(target)
        with patch.object(wt, 'solve_inpainting', side_effect=wt.InpaintingError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                bpy.ops.magic_fit.transfer_weights()
        self.assertEqual(self.memberships(target), before)

    def test_partial_defaults_reset_and_standalone_inpaint(self):
        self.assertFalse(self.settings.partial_reweight)
        self.assertEqual(self.settings.partial_reweight_falloff, 20)
        self.settings.partial_reweight = True
        self.settings.partial_reweight_falloff = 70
        bpy.ops.magic_fit.reset_transfer_settings()
        self.assertFalse(self.settings.partial_reweight)
        self.assertEqual(self.settings.partial_reweight_falloff, 20)
        self.settings.crease_smoothing = False
        self.settings.partial_reweight = True
        self.settings.max_distance = 0
        self.settings.inpaint_mode = 'SURFACE'
        self.test_standalone_inpaint_inversion_and_diagnostic()

    def test_operator_final_seams_and_geometry_and_bone_scaling(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        target.data.uv_layers.new(name='UVMap')
        target.shape_key_add(name='Basis')
        key = target.shape_key_add(name='Shape')
        key.data[0].co.z = 0.1
        self.weights(target, ['MaskData', 'C'], np.tile([0.33, 0.7], (8, 1)))
        # Source C exists but is zero everywhere: stale destination C must clear.
        self.body.vertex_groups.new(name='C')
        before = self.snapshot(target)
        self.settings.seam_sync = True
        self.settings.smoothing_enable = True
        self.settings.num_limit_groups = 1
        self.select(target)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        out = self.read(target)
        np.testing.assert_array_equal(out[[1, 3]], out[[4, 6]])
        self.assertTrue(np.all(np.count_nonzero(out[[1, 3]], axis=1) <= 1))
        np.testing.assert_allclose(out[[1, 3]].sum(axis=1), 1)
        np.testing.assert_array_equal(self.read(target, ['C']), 0)
        np.testing.assert_allclose(self.read(target, ['MaskData']), 0.33)
        self.assertEqual(before, self.snapshot(target))
        for name, scale, location, rotation in [('A', (1.7, 0.6, 1.3), (0.3, 0, 0.1), (0, 0.2, 0.3)),
                                                 ('B', (0.7, 1.6, 0.8), (0, 0.2, 0), (0.2, 0, 0))]:
            bone = self.rig.pose.bones[name]
            bone.scale = scale
            bone.location = location
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = rotation
        bpy.context.view_layer.update()
        evaluated = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
        coords = np.array([v.co[:] for v in evaluated.data.vertices])
        np.testing.assert_allclose(coords[[1, 3]], coords[[4, 6]], atol=1e-6)

    def test_cross_object_defaults_names_and_shared_solve(self):
        source = self.mesh('Source', self.v[:4], self.f[:2])
        self.weights(source, ['A', 'B'], np.tile([0.25, 0.75], (4, 1)))
        self.body = source
        left = self.mesh('Left', self.v[:4], self.f[:2])
        right = self.mesh('Right', self.v[4:], self.f[2:] - 4)
        # Different group ordering must not change the result.
        self.weights(right, ['B', 'A'], np.tile([0.1, 0.9], (4, 1)))
        self.settings.apply_to_selected = True
        self.settings.seam_sync = True
        self.settings.virtual_merge = True
        # Reject all right vertices by normals, including its duplicate seam.
        for polygon in right.data.polygons:
            polygon.flip()
        right.data.update()
        self.settings.flip_vertex_normal = False
        self.select(left, right)
        original_right = self.read(right).copy()
        with self.assertRaises(RuntimeError):
            bpy.ops.magic_fit.transfer_weights()
        np.testing.assert_array_equal(self.read(right), original_right)
        self.assertEqual(len(left.vertex_groups), 0)
        self.settings.seam_sync_across_objects = True
        self.select(right, left)
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.magic_fit.select_rejected(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertFalse(any(vertex.select for vertex in right.data.vertices))
        # Edit Mode also edits the active mesh when it isn't selected; it still shares the solve.
        right.select_set(False)
        for vertex in right.data.vertices:
            vertex.select = True
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.magic_fit.select_rejected(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertFalse(any(vertex.select for vertex in right.data.vertices))
        right.select_set(True)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(left), np.tile([0.25, 0.75], (4, 1)), atol=1e-6)
        np.testing.assert_allclose(self.read(right), self.read(left), atol=1e-6)

    def test_protected_seams_masks_and_locked_groups(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        self.weights(target, ['A', 'B'], np.array([[0.8, 0.2]] * 4 + [[0.3, 0.7]] * 4))
        target.vertex_groups['A'].lock_weight = True
        self.settings.seam_sync = True
        self.select(target)
        before_a = self.read(target, ['A']).copy()
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_array_equal(self.read(target, ['A']), before_a)
        self.assertNotEqual(self.read(target)[1, 0], self.read(target)[4, 0])
        target.vertex_groups['A'].lock_weight = False
        self.weights(target, ['TransferMask'], np.array([[0.0]] * 4 + [[0.5]] * 4))
        target.magic_fit_transfer.vertex_group = 'TransferMask'
        baseline = self.read(target).copy()
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_array_equal(self.read(target)[:4], baseline[:4])
        np.testing.assert_allclose(self.read(target)[5], 0.5 * baseline[5] + 0.5 * np.array([0.1, 0.9]), atol=1e-6)

    def test_standalone_inpaint_inversion_and_diagnostic(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        self.weights(target, ['A', 'B'], np.array([[0.25, 0.75]] * 4 + [[0.9, 0.1]] * 4))
        self.weights(target, ['Inpaint'], np.array([[1.0]] * 4 + [[0.0]] * 4))
        target.magic_fit_transfer.inpaint_group = 'Inpaint'
        target.magic_fit_transfer.inpaint_group_invert = True
        self.settings.virtual_merge = True
        self.settings.seam_sync = True
        self.select(target)
        self.assertEqual(bpy.ops.magic_fit.inpaint_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), np.tile([0.25, 0.75], (8, 1)), atol=1e-6)
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.magic_fit.select_rejected(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertFalse(any(v.select for v in target.data.vertices))
        self.settings.virtual_merge = False
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.magic_fit.select_rejected(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertEqual([v.select for v in target.data.vertices], [False] * 4 + [True] * 4)

    def test_inpaint_uses_the_rest_pose_like_the_transfer(self):
        self.source()
        self.settings.use_deformed_target = True
        v, f = flat_grid(np.linspace(0, 1, 11), np.linspace(0, 0.5, 6))
        target = self.mesh('Target', v, f)
        left = (v[:, 0] < 0.5).astype(float)
        self.weights(target, ['Inpaint'], ((v[:, 0] > 0.25) & (v[:, 0] < 0.75)).astype(float)[:, None])
        target.magic_fit_transfer.inpaint_group = 'Inpaint'
        self.select(target)
        states = [(mod.name, mod.show_viewport) for mod in target.modifiers]
        results = []
        # Lifting the left half stretches the faces in the middle, but a pose mustn't change the result.
        for lift in (0, 0.3):
            for name in ('A', 'B'):
                if name in target.vertex_groups:
                    target.vertex_groups.remove(target.vertex_groups[name])
            self.weights(target, ['A', 'B'], np.column_stack([left, 1 - left]))
            self.rig.pose.bones['A'].location = (0, lift, 0)
            bpy.context.view_layer.update()
            self.assertEqual(bpy.ops.magic_fit.inpaint_weights(), {'FINISHED'})
            results.append(self.read(target))
        self.rig.pose.bones['A'].location = (0, 0, 0)
        np.testing.assert_allclose(results[1], results[0], atol=1e-6)
        self.assertEqual([(mod.name, mod.show_viewport) for mod in target.modifiers], states)

    def staged_pair(self, right_rig=True):
        left = self.mesh('Left', self.v[:4], self.f[:2])
        right = self.mesh('Right', self.v[4:], self.f[2:] - 4, rig=right_rig)
        self.weights(right, ['B', 'A'], np.tile([0.9, 0.1], (4, 1)))
        targets = []
        for obj, values in [(left, [0.8, 0.2]), (right, [0.1, 0.9])]:
            target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
            target['weights'] = np.tile(values, (4, 1))
            transfer.stage_weights(target, ['A', 'B'], [True, True])
            targets.append(target)
        return targets

    def test_cross_scope_and_unbound_palette(self):
        self.settings.seam_sync = True
        self.settings.apply_to_selected = True
        targets = self.staged_pair(right_rig=False)
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (0, 0, 0))
        self.settings.seam_sync_across_objects = True
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (2, 0, 0))
        transfer.write_targets(targets)
        np.testing.assert_array_equal(self.read(targets[0]['obj'])[[1, 3]], self.read(targets[1]['obj'])[[0, 2]])

    def test_incompatible_armatures_and_target_only_weights(self):
        self.settings.seam_sync = self.settings.seam_sync_across_objects = True
        self.settings.apply_to_selected = True
        targets = self.staged_pair()
        other = self.make_rig('OtherRig')
        targets[1]['obj'].modifiers[0].object = other
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (0, 0, 2))
        targets[1]['obj'].modifiers[0].object = self.rig
        self.weights(targets[1]['obj'], ['C'], np.tile([0.2], (4, 1)))
        transfer.stage_weights(targets[1], ['A', 'B'], [True, True])
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (0, 2, 0))
        transfer.write_targets(targets)
        np.testing.assert_allclose(self.read(targets[1]['obj'], ['C']), 0.2)

    def test_seams_use_rest_positions_and_world_transforms(self):
        self.settings.seam_sync = self.settings.seam_sync_across_objects = True
        self.settings.apply_to_selected = True
        targets = self.staged_pair()
        right = targets[1]['obj']
        # Different local coordinates and object transforms, same world seam.
        for vertex in right.data.vertices:
            vertex.co.x -= 3
        right.location.x = 3
        bpy.context.view_layer.update()
        transfer.write_targets(targets)
        self.rig.pose.bones['A'].scale = (2, 1, 1)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        before = [util.get_obj_arrs_world(t['obj'].evaluated_get(depsgraph))[0] for t in targets]
        self.assertGreater(np.linalg.norm(before[0][1] - before[1][0]), 0.1)
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (2, 0, 0))
        transfer.write_targets(targets)
        bpy.context.view_layer.update()
        after = [util.get_obj_arrs_world(t['obj'].evaluated_get(bpy.context.evaluated_depsgraph_get()))[0]
                 for t in targets]
        np.testing.assert_allclose(after[0][[1, 3]], after[1][[0, 2]], atol=1e-6)

    def test_settings_defaults_and_reset(self):
        bpy.ops.magic_fit.reset_transfer_settings()
        self.assertTrue(self.settings.virtual_merge)
        self.assertTrue(self.settings.seam_sync)
        self.assertFalse(self.settings.seam_sync_across_objects)
        self.assertAlmostEqual(self.settings.seam_distance, 0.0001)
        self.assertFalse(self.settings.normalize_weights_after_transfer)
        self.assertAlmostEqual(np.degrees(self.settings.max_normal_angle_difference), 30, places=4)
        self.assertAlmostEqual(np.degrees(self.settings.soft_normal_limit), 90, places=4)
        self.assertTrue(self.settings.crease_smoothing)
        self.assertAlmostEqual(self.settings.crease_softness, 1.0)
        # FFXIV models take 8 bones per vertex.
        self.assertTrue(self.settings.enforce_four_bone_limit)
        self.assertEqual(self.settings.num_limit_groups, 8)
        self.settings.virtual_merge = False
        self.settings.seam_sync = False
        self.settings.seam_sync_across_objects = True
        self.settings.seam_distance = 0.01
        self.settings.normalize_weights_after_transfer = True
        self.settings.soft_normal_limit = 0.5
        self.settings.crease_smoothing = False
        self.settings.crease_softness = 1.5
        self.settings.num_limit_groups = 4
        bpy.ops.magic_fit.reset_transfer_settings()
        self.assertEqual(self.settings.inpaint_mode, 'COMBINED')
        # Files store each mode's number, so those must stay; the list shows Combined first.
        self.assertEqual({item.identifier: item.value
                          for item in self.settings.bl_rna.properties['inpaint_mode'].enum_items},
                         {'POINT': 0, 'SURFACE': 1, 'COMBINED': 2})
        self.assertEqual(self.settings.bl_rna.properties['inpaint_mode'].enum_items[0].identifier, 'COMBINED')
        self.assertEqual(self.settings.num_limit_groups, 8)
        self.assertTrue(self.settings.virtual_merge)
        self.assertTrue(self.settings.seam_sync)
        self.assertFalse(self.settings.seam_sync_across_objects)
        self.assertAlmostEqual(self.settings.seam_distance, 0.0001)
        self.assertFalse(self.settings.normalize_weights_after_transfer)
        self.assertAlmostEqual(np.degrees(self.settings.soft_normal_limit), 90, places=4)
        self.assertTrue(self.settings.crease_smoothing)
        self.assertAlmostEqual(self.settings.crease_softness, 1.0)

    def test_soft_normal_limit_rescues_steep_loose_part(self):
        xs, ys = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), indexing='ij')
        index = np.arange(25).reshape(5, 5)
        quads = [[index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]]
                 for i in range(4) for j in range(4)]
        source = self.mesh('Source', np.column_stack([xs.ravel(), ys.ravel(), np.zeros(25)]), quads)
        self.weights(source, ['A', 'B'], np.column_stack([(1 - xs.ravel()) / 2, (1 + xs.ravel()) / 2]))
        self.body = source
        # A flat part just above the source, and a separate part tilted by 60
        # degrees whose upper edge lies beyond Max Distance.
        rise = 0.1 * np.array([0, np.cos(np.pi / 3), np.sin(np.pi / 3)])
        bottom = [[0.3, 0, 0.02], [0.5, 0, 0.02]]
        target = self.mesh('Target', [[-0.5, 0, 0.02], [-0.3, 0, 0.02], [-0.3, 0.2, 0.02], [-0.5, 0.2, 0.02]]
                           + bottom + [np.add(bottom[1], rise), np.add(bottom[0], rise)],
                           [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.select(target)
        self.settings.soft_normal_limit = self.settings.max_normal_angle_difference
        with self.assertRaisesRegex(RuntimeError, 'No usable match'):
            bpy.ops.magic_fit.transfer_weights()
        self.assertEqual(len(target.vertex_groups), 0)
        bpy.ops.magic_fit.reset_transfer_settings()
        self.body = source
        self.settings.use_deformed_source = self.settings.use_deformed_target = False
        self.settings.inpaint_mode = 'SURFACE'
        self.settings.max_distance = 0.1
        self.settings.crease_smoothing = False
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        out = self.read(target)
        np.testing.assert_allclose(out[:4], [[.75, .25], [.65, .35], [.65, .35], [.75, .25]], atol=1e-6)
        np.testing.assert_allclose(out.sum(axis=1), 1, atol=1e-6)
        self.assertTrue(np.all((out[4:, 0] > 0.2) & (out[4:, 0] < 0.4)))
        # Full trust for direct matches, half for the 60 degree soft matches and
        # none where the tilted part leaves Max Distance.
        targets, _groups = operators.match_objects(bpy.context, self.settings, source, [target])
        np.testing.assert_allclose(targets[0]['confidence'], [1] * 4 + [.5, .5, 0, 0], atol=0.01)

    def test_transfer_clears_deform_groups_the_body_lacks(self):
        self.source()
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['A', 'C', 'D', 'Mask'], np.tile([.2, .5, .3, .5], (4, 1)))
        obj.vertex_groups['D'].lock_weight = True
        self.select(obj)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        # C is a bone the body has no weights for: a mod's skirt bone, say. Locked groups and groups
        # that aren't bones stay.
        np.testing.assert_allclose(self.read(obj, ['A', 'B']), np.tile([.9, .1], (4, 1)), atol=1e-6)
        np.testing.assert_allclose(self.read(obj, ['C', 'D', 'Mask']), np.tile([0, .3, .5], (4, 1)), atol=1e-6)
        # With a Transfer Mask, they fade out as far as the transfer reaches.
        self.weights(obj, ['C'], np.tile([.5], (4, 1)))
        obj.magic_fit_transfer.vertex_group = 'Mask'
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(obj, ['C']), .25, atol=1e-6)

    def test_max_groups_count_the_old_and_locked_weights(self):
        self.source()
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['C', 'D', 'E', 'Mask'], np.tile([.5, .3, .2, .5], (4, 1)))
        obj.magic_fit_transfer.vertex_group = 'Mask'
        self.settings.num_limit_groups = 3
        self.select(obj)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        # Half the body's A and B on half the old C, D and E would be five groups. The smallest fade
        # out and the others keep the total.
        bones = ['A', 'B', 'C', 'D', 'E']
        np.testing.assert_allclose(self.read(obj, bones), np.tile([.35, 0, .15, .05, 0], (4, 1)) / .55,
                                   atol=1e-6)
        np.testing.assert_allclose(self.read(obj, ['Mask']), .5, atol=1e-6)
        # Locked groups count against the limit and keep their weights.
        obj.magic_fit_transfer.vertex_group = ''
        self.weights(obj, ['D'], np.tile([.3], (4, 1)))
        obj.vertex_groups['D'].lock_weight = True
        self.settings.num_limit_groups = 2
        self.settings.normalize_weights_after_transfer = True
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(obj, bones), np.tile([.7, 0, 0, .3, 0], (4, 1)), atol=1e-6)

    def test_unrigged_target_loses_bone_groups_the_body_lacks(self):
        self.source()
        obj = self.mesh('Target', self.v[:4], self.f[:2], rig=False)
        self.weights(obj, ['A', 'C', 'Mask'], np.tile([.2, .5, .5], (4, 1)))
        self.select(obj)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        # C is a bone of the body's armature; Mask isn't a bone.
        np.testing.assert_allclose(self.read(obj, ['A', 'B', 'C', 'Mask']), np.tile([.9, .1, 0, .5], (4, 1)),
                                   atol=1e-6)

    def test_normalize_final_transfer_weights(self):
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['Arm.L', 'Arm.R', 'Spine', 'Mask'],
                     np.tile([.2, .1, .2, .8], (4, 1)))
        target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
        target['weights'] = np.tile([.5, .1], (4, 1))
        transfer.stage_weights(target, ['Arm.L', 'Arm.R'], [True, True])
        target['write_vertices'][:2] = False
        self.settings.normalize_weights_after_transfer = True
        counts = transfer.postprocess_transfer_targets([target], self.settings, ['Arm.L', 'Arm.R'])
        self.assertEqual(counts, {'normalization_failed': 0})
        transfer.write_targets([target])
        # The touched vertices' deform weights are scaled to add up to 1. The other vertices and the
        # non-deform Mask group are left alone.
        deform = self.read(obj, ['Arm.L', 'Arm.R', 'Spine'])
        np.testing.assert_allclose(deform[2:], np.tile([.5, .1, .2], (2, 1)) / .8, atol=1e-6)
        np.testing.assert_allclose(deform[:2], np.tile([.2, .1, .2], (2, 1)), atol=1e-6)
        np.testing.assert_allclose(self.read(obj, ['Mask']), .8, atol=1e-6)

    def test_normalize_preserves_locked_values_and_reports_infeasible_vertices(self):
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['Arm.L', 'Arm.R', 'Spine'], np.tile([.2, .2, .3], (4, 1)))
        for name in ['Arm.L', 'Arm.R', 'Spine']:
            obj.vertex_groups[name].lock_weight = True
        target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
        target['weights'] = np.tile([.9, .1], (4, 1))
        transfer.stage_weights(target, ['Arm.L', 'Arm.R'], [True, True])
        before = {name: values.copy() for name, values in target['final'].items()}
        self.settings.normalize_weights_after_transfer = True
        counts = transfer.postprocess_transfer_targets([target], self.settings, ['Arm.L', 'Arm.R'])
        self.assertEqual(counts['normalization_failed'], 4)
        for name, values in before.items():
            np.testing.assert_array_equal(target['final'][name], values)

    def test_normalize_repairs_writer_cutoff_residual(self):
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['Arm.L', 'Arm.R', 'Spine'],
                     np.tile([.2, .2, 1e-8], (4, 1)))
        target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
        target['weights'] = np.tile([.6, .4], (4, 1))
        transfer.stage_weights(target, ['Arm.L', 'Arm.R'], [True, True])
        self.settings.normalize_weights_after_transfer = True
        counts = transfer.postprocess_transfer_targets([target], self.settings, ['Arm.L', 'Arm.R'])
        self.assertEqual(counts['normalization_failed'], 0)
        transfer.write_targets([target])
        deform = self.read(obj, ['Arm.L', 'Arm.R', 'Spine'])
        np.testing.assert_allclose(deform.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(deform[:, 2], 0)

    def test_shared_mesh_is_rejected_without_mutation(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        linked = bpy.data.objects.new('Linked', target.data)
        bpy.context.collection.objects.link(linked)
        self.settings.seam_sync = True
        self.select(target)
        with self.assertRaises(RuntimeError):
            bpy.ops.magic_fit.transfer_weights()
        self.assertEqual(len(target.vertex_groups), 0)
        self.assertEqual(target.data, linked.data)

    RIGID_BONES = ('A', 'B', 'C', 'D', 'E', 'F')

    def rigid_scene(self, name='Target', offset=0.0):
        """A source whose weights run through bones A to F along x (bone k at x = 0.1 k), and a target
        lying on it over most of that, with finer faces on its left half."""
        v, f = flat_grid(np.linspace(0, 0.5, 26), np.linspace(0, 0.2, 5))
        if self.body is None:
            source = self.mesh('Source', v, f)
            position = v[:, 0] / 0.1
            values = np.clip(1.0 - np.abs(position[:, None] - np.arange(6)[None, :]), 0.0, 1.0)
            self.weights(source, self.RIGID_BONES, values)
            self.body = source
        xs = np.concatenate((np.linspace(0.05, 0.2, 16), np.linspace(0.23, 0.45, 5)))
        v, f = flat_grid(xs, np.linspace(0.05, 0.15, 4), 0.001 + offset)
        return self.mesh(name, v, f)

    def area_average(self, obj, values):
        vertices, triangles, _ = util.get_obj_arrs_world(obj)
        areas = transfer.vertex_areas(vertices, triangles)
        average = (areas[:, None] * values).sum(axis=0) / areas.sum()
        return average / average.sum()

    def test_rigid_gives_the_whole_mesh_one_set_of_weights(self):
        target = self.rigid_scene()
        self.select(target)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        loose = self.read(target, self.RIGID_BONES)
        self.assertGreater(np.ptp(loose, axis=0).max(), 0.5)  # an ordinary transfer varies across it
        expected = self.area_average(target, loose)

        for group in list(target.vertex_groups):
            target.vertex_groups.remove(group)
        target.magic_fit_transfer.rigid = True
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        rigid = self.read(target, self.RIGID_BONES)
        np.testing.assert_array_equal(rigid, np.tile(rigid[0], (len(rigid), 1)))  # exactly the same everywhere
        self.assertAlmostEqual(float(rigid[0].sum()), 1.0, places=5)
        np.testing.assert_allclose(rigid[0], expected, atol=1e-5)  # the area-weighted average

        # Posed, every vertex moves with the same blend of bones: one linear map for the whole mesh.
        for bone, angle in (('A', 0.4), ('C', -0.3), ('E', 0.2)):
            self.rig.pose.bones[bone].rotation_mode = 'XYZ'
            self.rig.pose.bones[bone].rotation_euler = (angle, 0.0, 0.0)
        bpy.context.view_layer.update()
        rest = np.array([v.co[:] for v in target.data.vertices])
        posed = np.array([v.co[:] for v in target.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
        homogeneous = np.column_stack((rest, np.ones(len(rest))))
        fit, *_ = np.linalg.lstsq(homogeneous, posed, rcond=None)
        self.assertLess(np.abs(homogeneous @ fit - posed).max(), 1e-5)

    def test_rigid_weights_average_by_area_and_drop_what_rounds_away(self):
        values = np.array([[0.6, 0.399, 0.001, 0.5], [0.8, 0.199, 0.001, 0.5]])
        movable = np.array([True, True, True, False])
        shares = transfer.rigid_weights(values, [3.0, 1.0], movable)
        # Weighted 3:1, and the 0.001 (under half a step of 8-bit weights) dropped.
        np.testing.assert_allclose(shares, np.array([0.65, 0.349, 0.0]) / 0.999, atol=1e-9)
        np.testing.assert_allclose(transfer.rigid_weights(values, [0.0, 0.0], movable),
                                   np.array([0.7, 0.299, 0.0]) / 0.999, atol=1e-9)
        np.testing.assert_allclose(transfer.rigid_weights(values, [1.0, 1.0], movable, limit=1), [1.0, 0.0, 0.0])
        self.assertIsNone(transfer.rigid_weights(np.zeros((2, 3)), [1.0, 1.0], np.ones(3, bool)))

    def test_rigid_respects_max_groups_and_locked_groups(self):
        target = self.rigid_scene()
        target.magic_fit_transfer.rigid = True
        self.settings.num_limit_groups = 2
        self.select(target)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        rigid = self.read(target, self.RIGID_BONES)
        self.assertEqual(np.count_nonzero(rigid[0]), 2)
        np.testing.assert_array_equal(rigid, np.tile(rigid[0], (len(rigid), 1)))
        self.assertAlmostEqual(float(rigid[0].sum()), 1.0, places=5)

        self.settings.num_limit_groups = 8
        count = len(target.data.vertices)
        spine = np.linspace(0.0, 0.3, count)
        for group in list(target.vertex_groups):
            target.vertex_groups.remove(group)
        self.weights(target, ['Spine'], spine[:, None])
        target.vertex_groups['Spine'].lock_weight = True
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target, ['Spine'])[:, 0], spine, atol=1e-6)
        rest = self.read(target, self.RIGID_BONES)
        shares = rest / rest.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(shares, np.tile(shares[0], (count, 1)), atol=1e-6)
        np.testing.assert_allclose(rest.sum(axis=1) + spine, 1.0, atol=1e-5)

    def test_rigid_is_per_mesh_with_apply_to_selected(self):
        rigid = self.rigid_scene('Buckle')
        loose = self.rigid_scene('Strap', offset=0.002)
        rigid.magic_fit_transfer.rigid = True
        self.settings.apply_to_selected = True
        self.select(rigid, loose)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        self.assertLess(np.ptp(self.read(rigid, self.RIGID_BONES), axis=0).max(), 1e-7)
        self.assertGreater(np.ptp(self.read(loose, self.RIGID_BONES), axis=0).max(), 0.5)

    def test_non_deform_group_is_not_used_from_other_rig(self):
        # A group can be a bone on one target but an unrelated mask on another.
        self.settings.seam_sync = True
        first = self.mesh('First', self.v, self.f)
        second = self.mesh('Second', self.v + [0, 0, 3], self.f)
        other = self.make_rig('Other')
        other.data.bones['C'].use_deform = False
        second.modifiers[0].object = other
        targets = []
        for obj in [first, second]:
            self.weights(obj, ['C'], np.tile([0.4], (8, 1)))
            target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
            target['weights'] = np.tile([0.8, 0.2, 0.4], (8, 1))
            transfer.stage_weights(target, ['A', 'B', 'C'], [True] * 3)
            targets.append(target)
        transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig)
        transfer.write_targets(targets)
        np.testing.assert_allclose(self.read(second, ['C']), 0.4)


class CreaseSmoothingTests(TransferScene, unittest.TestCase):
    """Weight Transfer with Crease Smoothing, the default. BlenderTests covers the matching and inpainting
    underneath with it off."""

    def setUp(self):
        super().setUp()
        self.settings.crease_smoothing = True
        self.settings.inpaint_mode = 'COMBINED'
        self.settings.max_distance = 0.05

    def crease_scene(self, gusset=False):
        """A V-shaped groove weighted to bones L and R, and a narrow strap lying in it. With `gusset`, a
        wider piece instead that lies on both walls and bridges the bottom of the groove."""
        data = bpy.data.armatures.new('CreaseRig')
        rig = bpy.data.objects.new('CreaseRig', data)
        bpy.context.collection.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        for name, x in (('L', -0.3), ('R', 0.3)):
            bone = data.edit_bones.new(name)
            bone.head = (x, 0, 0.03)
            bone.tail = (x, 0, 0.13)
        bpy.ops.object.mode_set(mode='OBJECT')
        rig.select_set(False)

        def quads(columns, rows):
            index = np.arange(columns * rows).reshape(columns, rows)
            return [[index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]]
                    for i in range(columns - 1) for j in range(rows - 1)]

        def rigged(name, v, f):
            obj = self.mesh(name, v, f, rig=False)
            obj.modifiers.new('Armature', 'ARMATURE').object = rig
            return obj

        gx, gy = np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 11), indexing='ij')
        # The walls rise at 45 degrees on both sides of x == 0.
        source = rigged('Source', np.column_stack([gx.ravel(), gy.ravel(), np.abs(gx.ravel())]), quads(21, 11))
        side = np.sign(gx.ravel())
        self.weights(source, ['L', 'R'], np.column_stack([np.where(side < 0, 1, np.where(side > 0, 0, .5)),
                                                          np.where(side > 0, 1, np.where(side < 0, 0, .5))]))
        self.body = source
        if gusset:
            # 1 mm above the walls, and flat across the groove where they're less than 2 cm high.
            sx, sy = np.meshgrid(np.linspace(-0.05, 0.05, 26), np.linspace(-0.06, 0.06, 13), indexing='ij')
            height = np.maximum(np.abs(sx.ravel()), 0.02) + 0.001
            strap = rigged('Gusset', np.column_stack([sx.ravel(), sy.ravel(), height]), quads(26, 13))
        else:
            sx, sy = np.meshgrid(np.linspace(-0.012, 0.012, 7), np.linspace(-0.06, 0.06, 13), indexing='ij')
            strap = rigged('Strap', np.column_stack([sx.ravel(), sy.ravel(), np.full(sx.size, 0.03)]),
                           quads(7, 13))
        self.select(strap)
        return rig, strap, sx.ravel()

    def posed(self, obj):
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        return np.array([vertex.co[:] for vertex in mesh.vertices])

    def test_strap_in_a_crease_does_not_twist(self):
        rig, strap, x = self.crease_scene()
        # The walls are 45 degrees off the strap. Copying them exactly, like a strap hugging the walls of
        # a cleft, lets each edge follow its own wall.
        self.settings.max_normal_angle_difference = np.radians(50)
        normals, gaps = {}, {}
        for enabled in (False, True):
            self.settings.crease_smoothing = enabled
            for pose_bone in rig.pose.bones:
                pose_bone.scale = (1, 1, 1)
            bpy.context.view_layer.update()
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
            weights = self.read(strap, ['L', 'R'])
            np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-5)
            gaps[enabled] = abs(weights[x == x.min(), 0].mean() - weights[x == x.max(), 0].mean())
            # Growing both sides of the groove pushes its walls past each other. A strap switching sides
            # across its width turns inside out.
            for pose_bone in rig.pose.bones:
                pose_bone.scale = (2, 2, 2)
            bpy.context.view_layer.update()
            posed = self.posed(strap)
            normals[enabled] = [np.cross(posed[p.vertices[1]] - posed[p.vertices[0]],
                                         posed[p.vertices[2]] - posed[p.vertices[0]])[2]
                                for p in strap.data.polygons]
        self.assertGreater(gaps[False], 0.3)
        self.assertLess(gaps[True], gaps[False] / 2)
        self.assertTrue(any(z < 0 for z in normals[False]))
        self.assertTrue(all(z > 0 for z in normals[True]))

    def test_bridge_blends_but_what_lies_on_the_body_follows_it(self):
        rig, gusset, x = self.crease_scene(gusset=True)
        bone = rig.pose.bones['L']
        bone.rotation_mode = 'XYZ'
        rest = np.array([vertex.co[:] for vertex in gusset.data.vertices])
        edges = np.array([edge.vertices[:] for edge in gusset.data.edges])
        row = np.flatnonzero(np.isclose(rest[:, 1], 0))
        row = row[np.argsort(rest[row, 0])]
        stretch = {}
        for enabled in (False, True):
            self.settings.crease_smoothing = enabled
            self.settings.max_normal_angle_difference = np.radians(50)
            bone.rotation_euler = (0, 0, 0)
            bpy.context.view_layer.update()
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
            weights = self.read(gusset, ['L', 'R'])
            np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-5)
            # Away from the groove, the gusset lies on the walls and follows them.
            far = np.abs(x) > 0.04
            np.testing.assert_allclose(weights[far & (x < 0)], np.tile([1, 0], (np.count_nonzero(far & (x < 0)), 1)),
                                       atol=0.02)
            np.testing.assert_allclose(weights[far & (x > 0)], np.tile([0, 1], (np.count_nonzero(far & (x > 0)), 1)),
                                       atol=0.02)
            if enabled:
                # Across the groove, both sides blend gradually and evenly...
                profile = weights[row, 0]
                self.assertTrue(np.all(np.diff(profile) <= 1e-4))
                self.assertLess(-np.diff(profile).min(), 0.2)
                np.testing.assert_allclose(profile + profile[::-1], 1, atol=0.05)
                # ...trusting the walls where the gusset lies on them and not where it bridges the groove.
                targets, _groups = operators.match_objects(bpy.context, self.settings, self.body, [gusset])
                trust = targets[0]['confidence']
                self.assertGreater(trust[(np.abs(x) > 0.03) & (np.abs(rest[:, 1]) < 0.03)].min(), 0.9)
                self.assertLess(trust[(np.abs(x) < 0.015) & (np.abs(rest[:, 1]) < 0.02)].max(), 0.1)
            # Turning one wall drags its side along. Where the weights switch sides abruptly, the edges
            # across the switch stretch much further.
            bone.rotation_euler = (0, np.radians(20), 0)
            bpy.context.view_layer.update()
            posed = self.posed(gusset)
            stretch[enabled] = (np.linalg.norm(posed[edges[:, 1]] - posed[edges[:, 0]], axis=1)
                                / np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)).max()
        bone.rotation_euler = (0, 0, 0)
        self.assertLess(stretch[True], stretch[False] / 3)

    def test_bound_meshes_match_in_the_rest_pose(self):
        rig, gusset, _x = self.crease_scene(gusset=True)
        self.settings.use_deformed_source = True
        self.settings.use_deformed_target = True
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        rest = self.read(gusset, ['L', 'R'])
        # A pose (or the weights the gusset has now) doesn't change what it gets.
        bone = rig.pose.bones['L']
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = (0, np.radians(35), 0)
        bone.scale = (1.5, 1.5, 1.5)
        bpy.context.view_layer.update()
        states = [(mod.name, mod.show_viewport) for obj in (self.body, gusset) for mod in obj.modifiers]
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(gusset, ['L', 'R']), rest, atol=1e-6)
        self.assertEqual([(mod.name, mod.show_viewport) for obj in (self.body, gusset) for mod in obj.modifiers],
                         states)

    def test_customize_plus_is_left_out(self):
        from magic_fit.cplus import rig as cplus_rig
        rig, gusset, _x = self.crease_scene(gusset=True)
        self.settings.use_deformed_source = True
        self.settings.use_deformed_target = True
        # An unbound copy of the gusset, matched against the body as displayed.
        loose = self.mesh('Loose', [v.co[:] for v in gusset.data.vertices],
                          [p.vertices[:] for p in gusset.data.polygons], rig=False)
        results = {}
        for label, cplus in (('plain', False), ('cplus', True)):
            if cplus:
                fake = bpy.data.objects.new('C+ CreaseRig', rig.data.copy())
                bpy.context.collection.objects.link(fake)
                fake.parent = rig
                fake[cplus_rig.MARK] = '{}'
                bpy.context.view_layer.update()
                self.assertEqual(cplus_rig.connect(rig, fake), 2)
                fake.pose.bones['L'].scale = (2, 2, 2)
                bpy.context.view_layer.update()
                states = [(mod.name, mod.show_viewport) for mod in self.body.modifiers]
            for obj in (gusset, loose):
                self.select(obj)
                self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
                results[label, obj.name] = self.read(obj, ['L', 'R'])
            if cplus:
                self.assertEqual([(mod.name, mod.show_viewport) for mod in self.body.modifiers], states)
        for name in (gusset.name, loose.name):
            np.testing.assert_allclose(results['cplus', name], results['plain', name], atol=1e-6)

    def test_split_seams_get_the_same_weights(self):
        _rig, gusset, x = self.crease_scene(gusset=True)
        # Cut the gusset into two loose parts along a column of vertices off the middle.
        rest = np.array([vertex.co[:] for vertex in gusset.data.vertices])
        faces = [list(p.vertices) for p in gusset.data.polygons]
        cut = np.unique(x)[8]
        right = rest[:, 0] >= cut - 1e-9
        copies = {int(i): len(rest) + k for k, i in enumerate(np.flatnonzero(np.isclose(rest[:, 0], cut)))}
        vertices = np.vstack([rest, rest[list(copies)]])
        split = [[copies.get(i, i) if all(right[j] for j in face) else i for i in face] for face in faces]
        target = self.mesh('Split', vertices, split, rig=False)
        target.modifiers.new('Armature', 'ARMATURE').object = gusset.modifiers[0].object
        self.select(target)
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        weights = self.read(target, ['L', 'R'])
        pairs = np.array(list(copies.items()))
        np.testing.assert_allclose(weights[pairs[:, 0]], weights[pairs[:, 1]], atol=1e-6)

    def legs_scene(self):
        """Two thighs (FFXIV bone names) under a short skirt tube whose hem lies on them."""
        data = bpy.data.armatures.new('Legs')
        rig = bpy.data.objects.new('Legs', data)
        bpy.context.collection.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        hips = data.edit_bones.new('j_kosi')
        hips.head, hips.tail = (0, 0, 0.9), (0, 0, 1.0)
        for name, x in (('j_asi_a_l', 0.08), ('j_asi_a_r', -0.08)):
            bone = data.edit_bones.new(name)
            bone.head, bone.tail = (x, 0, 0.9), (x, 0, 0.5)
            bone.parent = hips
        bpy.ops.object.mode_set(mode='OBJECT')
        rig.select_set(False)

        def tube(name, centres, radius, bottom, top, rings=9, sides=16):
            angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
            vertices, faces = [], []
            for cx in centres:
                start = len(vertices)
                for z in np.linspace(bottom, top, rings):
                    vertices += [[cx + radius * np.cos(a), 0.06 * np.sin(a), z] for a in angles]
                for ring in range(rings - 1):
                    for side in range(sides):
                        a = start + ring * sides + side
                        b = start + ring * sides + (side + 1) % sides
                        faces.append([a, b, b + sides, a + sides])
            obj = self.mesh(name, vertices, faces, rig=False)
            obj.modifiers.new('Armature', 'ARMATURE').object = rig
            return obj, np.array(vertices)

        body, points = tube('Body', (0.08, -0.08), 0.06, 0.55, 0.95)
        self.weights(body, ['j_asi_a_l', 'j_asi_a_r', 'j_kosi'],
                     np.column_stack([(points[:, 0] > 0) * np.clip((0.95 - points[:, 2]) / 0.2, 0, 1),
                                      (points[:, 0] < 0) * np.clip((0.95 - points[:, 2]) / 0.2, 0, 1),
                                      np.clip((points[:, 2] - 0.75) / 0.2, 0, 1)]))
        self.body = body
        skirt, _points = tube('Skirt', (0.0,), 0.15, 0.7, 0.95)
        self.select(skirt)
        return rig, skirt

    def test_hem_follow_runs_with_ffxiv_legs(self):
        self.settings.use_deformed_source = True
        self.settings.use_deformed_target = True
        _rig, skirt = self.legs_scene()
        pulls = []
        original = wt.hem_follow

        def record(*args):
            pulls.append(original(*args))
            return pulls[-1]
        with patch.object(transfer, 'follow_hems', wraps=transfer.follow_hems) as follow, \
                patch.object(wt, 'hem_follow', side_effect=record):
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        batch = follow.call_args.args[0]
        self.assertIn('strides', batch[0])
        self.assertEqual(len(batch[0]['strides']['poses']), 4)
        # The swinging thighs would come through the hem, so it's pulled towards them.
        self.assertGreater(max(pull.max() for pull in pulls), 0.1)
        weights = self.read(skirt, ['j_asi_a_l', 'j_asi_a_r', 'j_kosi'])
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-5)
        # Without the thighs' names, there are no strides to check.
        for bone in _rig.data.bones:
            if bone.name.startswith('j_asi'):
                bone.name = bone.name.replace('j_asi', 'leg')
        with patch.object(transfer, 'follow_hems', wraps=transfer.follow_hems) as follow:
            self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        self.assertNotIn('strides', follow.call_args.args[0][0])

    def test_customize_plus_on_the_meshes_own_armature_is_left_out(self):
        from magic_fit.cplus import rig as cplus_rig
        _rig, gusset, _x = self.crease_scene(gusset=True)
        self.settings.use_deformed_source = True
        self.settings.use_deformed_target = True
        # Gear on its own skeleton (with the same bone names), which gets a Customize+ rig of its own.
        gear_rig = bpy.data.objects.new('GearRig', gusset.modifiers[0].object.data.copy())
        bpy.context.collection.objects.link(gear_rig)
        gusset.modifiers[0].object = gear_rig
        bpy.context.view_layer.update()
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        plain = self.read(gusset, ['L', 'R'])
        fake = bpy.data.objects.new('C+ GearRig', gear_rig.data.copy())
        bpy.context.collection.objects.link(fake)
        fake.parent = gear_rig
        fake[cplus_rig.MARK] = '{}'
        bpy.context.view_layer.update()
        self.assertEqual(cplus_rig.connect(gear_rig, fake), 1)
        fake.pose.bones['L'].scale = (1.4, 1.4, 1.4)
        bpy.context.view_layer.update()
        states = [(mod.name, mod.show_viewport) for mod in gusset.modifiers]
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(gusset, ['L', 'R']), plain, atol=1e-6)
        self.assertEqual([(mod.name, mod.show_viewport) for mod in gusset.modifiers], states)

    def test_stray_vertices_keep_the_bodys_weights(self):
        _rig, gusset, x = self.crease_scene(gusset=True)
        self.settings.inpaint_mode = 'SURFACE'
        mesh = gusset.data
        mesh.vertices.add(1)
        mesh.vertices[-1].co = (-0.08, 0.0, 0.081)
        mesh.update()
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(gusset, ['L', 'R'])[-1], [1, 0], atol=1e-6)

    def test_meshes_sharing_a_solve_match_one_mesh(self):
        rig, gusset, x = self.crease_scene(gusset=True)
        rest = np.array([vertex.co[:] for vertex in gusset.data.vertices])
        faces = [list(p.vertices) for p in gusset.data.polygons]
        cut = np.unique(x)[8]
        # The gusset as one mesh with a split seam, and as two objects meeting there.
        right = [all(rest[i, 0] >= cut - 1e-9 for i in face) for face in faces]
        parts = []
        for name, keep in (('Left', [not r for r in right]), ('Right', right)):
            used = sorted({i for face, k in zip(faces, keep) if k for i in face})
            index = {v: i for i, v in enumerate(used)}
            obj = self.mesh(name, rest[used], [[index[i] for i in face] for face, k in zip(faces, keep) if k],
                            rig=False)
            obj.modifiers.new('Armature', 'ARMATURE').object = rig
            parts.append((obj, used))
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        whole = self.read(gusset, ['L', 'R'])
        self.settings.apply_to_selected = True
        self.settings.seam_sync = True
        self.settings.seam_sync_across_objects = True
        self.settings.virtual_merge = True
        self.select(parts[0][0], parts[1][0])
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        for obj, used in parts:
            np.testing.assert_allclose(self.read(obj, ['L', 'R']), whole[used], atol=0.02)

    def test_zero_max_distance_still_transfers(self):
        _rig, gusset, _x = self.crease_scene(gusset=True)
        self.settings.max_distance = 0
        self.assertEqual(bpy.ops.magic_fit.transfer_weights(), {'FINISHED'})
        np.testing.assert_allclose(self.read(gusset, ['L', 'R']).sum(axis=1), 1, atol=1e-5)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    magic_fit.unregister()
    failures = len(result.failures) + len(result.errors)
    print("RESULT:", "OK" if result.wasSuccessful() else "FAILED", failures, "failure(s)")
    sys.exit(0 if result.wasSuccessful() else 1)
