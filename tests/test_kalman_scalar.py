def test_vbf_package_imports() -> None:
    import vbf

    assert "kalman" in vbf.__all__

