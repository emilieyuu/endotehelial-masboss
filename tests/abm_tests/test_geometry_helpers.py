import numpy as np

from src.abm.helpers.geometry import (
    axial_coord,
    lateral_coord,
    perpendicular,
    polar_mask,
    polygon_area,
    polygon_arc_lengths,
    polygon_outward_normals,
)
from tests.abm_tests.helpers_shared import ABMHelperTestCase


class TestGeometryHelpers(ABMHelperTestCase):
    def test_axial_coord_projects_single_point_onto_normalized_axis(self):
        coord = axial_coord(points=[3.0, 4.0], origin=[1.0, 1.0], axis=[2.0, 0.0])
        self.assertAlmostEqual(coord, 2.0)

    def test_axial_coord_projects_multiple_points(self):
        coords = axial_coord(
            points=np.array([[1.0, 2.0], [4.0, 2.0], [-2.0, 2.0]]),
            origin=[1.0, 2.0],
            axis=[1.0, 0.0],
        )
        np.testing.assert_allclose(coords, [0.0, 3.0, -3.0])

    def test_lateral_coord_uses_counterclockwise_perpendicular_sign(self):
        coords = lateral_coord(
            points=np.array([[0.0, 2.0], [0.0, -2.0], [0.0, 0.0]]),
            origin=[0.0, 0.0],
            axis=[1.0, 0.0],
        )
        np.testing.assert_allclose(coords, [2.0, -2.0, 0.0])

    def test_perpendicular_rotates_vector_counterclockwise(self):
        np.testing.assert_allclose(perpendicular(np.array([2.0, -5.0])), [5.0, 2.0])

    def test_polar_mask_marks_points_near_positive_and_negative_axis_as_polar(self):
        points = np.array([
            [5.0, 0.0],
            [-5.0, 0.0],
            [0.0, 5.0],
            [3.0, 3.0],
        ])
        mask = polar_mask(points, origin=[0.0, 0.0], axis=[1.0, 0.0], angle_deg=30.0)
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_polygon_area_returns_positive_area(self):
        square = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])
        self.assertAlmostEqual(polygon_area(square), 4.0)

    def test_polygon_outward_normals_for_unit_square_point_outward(self):
        square = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        normals = polygon_outward_normals(square)
        expected = np.array([
            [-np.sqrt(0.5), -np.sqrt(0.5)],
            [np.sqrt(0.5), -np.sqrt(0.5)],
            [np.sqrt(0.5), np.sqrt(0.5)],
            [-np.sqrt(0.5), np.sqrt(0.5)],
        ])
        np.testing.assert_allclose(normals, expected)

    def test_polygon_arc_lengths_assign_average_adjacent_edge_lengths(self):
        rectangle = np.array([
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [0.0, 2.0],
        ])
        arc_lengths = polygon_arc_lengths(rectangle)
        np.testing.assert_allclose(arc_lengths, [3.0, 3.0, 3.0, 3.0])
