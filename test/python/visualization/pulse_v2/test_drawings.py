# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for core modules of pulse drawer."""

from qiskit import pulse
from qiskit.visualization.pulse_v2 import drawings, types
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestDrawingObjects(QiskitTestCase):
    """Tests for DrawingObjects."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        # bits
        self.ch_d0 = [pulse.DriveChannel(0)]
        self.ch_d1 = [pulse.DriveChannel(1)]

        # metadata
        self.meta1 = {"val1": 0, "val2": 1}
        self.meta2 = {"val1": 2, "val2": 3}

        # style data
        self.style1 = {"property1": 0, "property2": 1}
        self.style2 = {"property1": 2, "property2": 3}

    def test_line_data_equivalent(self):
        """Test for LineData."""
        xs = [0, 1, 2]
        ys = [3, 4, 5]

        data1 = drawings.LineData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d0,
            meta=self.meta1,
            ignore_scaling=True,
            styles=self.style1,
        )

        data2 = drawings.LineData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d1,
            meta=self.meta2,
            ignore_scaling=True,
            styles=self.style2,
        )

        self.assertEqual(data1, data2)

    def test_line_data_equivalent_with_abstract_coordinate(self):
        """Test for LineData with abstract coordinate."""
        xs = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.TOP, types.AbstractCoordinate.BOTTOM]

        data1 = drawings.LineData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d0,
            meta=self.meta1,
            ignore_scaling=True,
            styles=self.style1,
        )

        data2 = drawings.LineData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d1,
            meta=self.meta2,
            ignore_scaling=True,
            styles=self.style2,
        )

        self.assertEqual(data1, data2)

    def test_text_data(self):
        """Test for TextData."""
        xs = [0]
        ys = [1]

        data1 = drawings.TextData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            text="test1",
            latex=r"test_1",
            channels=self.ch_d0,
            meta=self.meta1,
            ignore_scaling=True,
            styles=self.style1,
        )

        data2 = drawings.TextData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            text="test2",
            latex=r"test_2",
            channels=self.ch_d1,
            meta=self.meta2,
            ignore_scaling=True,
            styles=self.style2,
        )

        self.assertEqual(data1, data2)

    def test_text_data_with_abstract_coordinate(self):
        """Test for TextData with abstract coordinates."""
        xs = [types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.TOP]

        data1 = drawings.TextData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            text="test1",
            latex=r"test_1",
            channels=self.ch_d0,
            meta=self.meta1,
            ignore_scaling=True,
            styles=self.style1,
        )

        data2 = drawings.TextData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            text="test2",
            latex=r"test_2",
            channels=self.ch_d1,
            meta=self.meta2,
            ignore_scaling=True,
            styles=self.style2,
        )

        self.assertEqual(data1, data2)

    def test_box_data(self):
        """Test for BoxData."""
        xs = [0, 1]
        ys = [-1, 1]

        data1 = drawings.BoxData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d0,
            meta=self.meta1,
            ignore_scaling=True,
            styles=self.style1,
        )

        data2 = drawings.BoxData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d1,
            meta=self.meta2,
            ignore_scaling=True,
            styles=self.style2,
        )

        self.assertEqual(data1, data2)

    def test_box_data_with_abstract_coordinate(self):
        """Test for BoxData with abstract coordinate."""
        xs = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.BOTTOM, types.AbstractCoordinate.TOP]

        data1 = drawings.BoxData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d0,
            meta=self.meta1,
            ignore_scaling=True,
            styles=self.style1,
        )

        data2 = drawings.BoxData(
            data_type="test",
            xvals=xs,
            yvals=ys,
            channels=self.ch_d1,
            meta=self.meta2,
            ignore_scaling=True,
            styles=self.style2,
        )

        self.assertEqual(data1, data2)
