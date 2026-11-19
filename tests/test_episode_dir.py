from collector import Collector


def test_episode_dir_is_house_name_with_suffix_on_collision(tmp_path):
    first = Collector._episode_dir(str(tmp_path), "house_000436")
    assert first == str(tmp_path / "house_000436")
    (tmp_path / "house_000436").mkdir()
    assert Collector._episode_dir(str(tmp_path), "house_000436") == str(tmp_path / "house_000436_2")
    (tmp_path / "house_000436_2").mkdir()
    assert Collector._episode_dir(str(tmp_path), "house_000436") == str(tmp_path / "house_000436_3")
