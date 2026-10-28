def test_pmf_import_has_no_optional_loss_requirement():
    from inv_framework.models.pmf import pixelMeanFlow
    assert pixelMeanFlow.__name__ == "pixelMeanFlow"


def test_public_cuda_surface():
    import inv_framework.operators.ct.radon_cuda as radon
    assert {"project", "bp", "gridrec_cuda"}.issubset(radon.__all__)
    assert "bp_linear" not in radon.__all__
    assert "fbp_astra" not in radon.__all__
