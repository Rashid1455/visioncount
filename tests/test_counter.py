from visioncount.counter import LineCounter


def make():
    return LineCounter((0, 100), (200, 100))  # horizontal line y=100


def test_counts_downward_and_upward_crossings():
    lc = make()
    lc.update(1, (50, 80), "car"); assert lc.update(1, (50, 120), "car") is not None
    lc.update(2, (60, 130), "person"); assert lc.update(2, (60, 90), "person") is not None
    assert lc.in_count == 1 and lc.out_count == 1 and lc.total == 2


def test_same_track_not_double_counted():
    lc = make()
    for y in (80, 120, 80, 120, 80):
        lc.update(7, (50, y))
    assert lc.total == 2  # one in, one out, then ignored


def test_no_count_outside_segment():
    lc = make()
    lc.update(3, (300, 80)); lc.update(3, (300, 120))
    assert lc.total == 0


def test_no_count_without_crossing():
    lc = make()
    for y in (10, 30, 60, 90):
        lc.update(4, (50, y))
    assert lc.total == 0


def test_per_class_breakdown():
    lc = make()
    lc.update(1, (50, 80), "car"); lc.update(1, (50, 120), "car")
    lc.update(2, (70, 80), "truck"); lc.update(2, (70, 120), "truck")
    assert set(lc.per_class) == {"car", "truck"}
