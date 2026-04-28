import numpy as np
import SimpleITK as sitk


def normalize(img):
    # taken from the original authors here:
    # https://github.com/sbonaretti/pyKNEEr/blob/master/pykneer/pykneer/sitk_functions.py#L255
    def field_correction(img):
        # creating Otsu mask
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(0)
        otsu.SetOutsideValue(1)
        otsu.SetNumberOfHistogramBins(200)
        mask = otsu.Execute(img)

        # correct field
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaskLabel(1)
        # CHANGED: reduced bins from 600 to 300
        corrector.SetNumberOfHistogramBins(300)
        corrector.SetWienerFilterNoise(10)
        corrector.SetBiasFieldFullWidthAtHalfMaximum(15)
        corrector.SetMaximumNumberOfIterations([50])
        corrector.SetConvergenceThreshold(0.001)
        img = corrector.Execute(img, mask)

        return img

    def rescale_to_range(img):
        img = sitk.GetArrayFromImage(img)
        qp = np.percentile(img, 99.9)
        img = np.clip(img / qp, 0.0, 1.0)
        return img

    # NOTE: high res data is in the range -1..1
    img = (img - img.min()) / (img.max() - img.min())
    # correct low frequency intensity non-uniformity:
    # https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    img = sitk.GetImageFromArray(img)
    img = field_correction(img)
    # linearly maps the 99.9th percentile intensity to the range `0..1`, clipping
    # anything above to `1`.
    img = rescale_to_range(img)
    return img
